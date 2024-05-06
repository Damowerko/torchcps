from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, StackDataset, Subset, TensorDataset
from torch_cluster.grid import grid_cluster
from torch_geometric.nn.pool import avg_pool_x

from torchcps.kernel.nn import sample_kernel, solve_kernel
from torchcps.kernel.rkhs import GaussianKernel, Kernel, Mixture, projection


def preprocess_aq(df_aq, df_aq_station) -> tuple[pd.DataFrame, pd.DataFrame]:
    # preprocess aq data
    df_aq = df_aq.copy()
    df_aq["utc_time"] = pd.to_datetime(df_aq["utc_time"])
    df_aq = df_aq.set_index("utc_time")
    df_aq = df_aq.sort_index()
    # rename stationId to station_id
    df_aq = df_aq.rename(columns={"stationId": "station_id"})
    df_aq_station = df_aq_station.rename(columns={"ID": "station_id"})
    # add station_index from df_aq_station
    df_aq["station_index"] = (
        df_aq.reset_index()
        .merge(
            df_aq_station.reset_index(),
            left_on="station_id",
            right_on="station_id",
            how="left",
        )
        .set_index("utc_time")
        .sort_index()["index"]
    )
    df_aq = df_aq.drop(columns=["station_id"])
    # resample hourly and forward fill up to 24 hours
    df_aq = (
        df_aq.groupby("station_index")
        .apply(
            lambda df: df.resample("3h", origin="start_day").mean().ffill(limit=8),
            include_groups=False,
        )
        .reset_index("station_index")
    )
    return df_aq, df_aq_station


def preprocess_meo(df_meo, df_meo_station) -> tuple[pd.DataFrame, pd.DataFrame]:
    # drop long lat
    df_meo = df_meo.drop(columns=["longitude", "latitude"])
    # time index
    df_meo["utc_time"] = pd.to_datetime(df_meo["utc_time"])
    df_meo = df_meo.set_index("utc_time").sort_index()
    # humidity is a percentage so use float
    df_meo["humidity"] = df_meo["humidity"].astype(float)
    # convert wind speed from polar to cartesian
    df_meo["wind_x"] = df_meo["wind_speed"] * np.cos(
        df_meo["wind_direction"] * np.pi / 180
    )
    df_meo["wind_y"] = df_meo["wind_speed"] * np.sin(
        df_meo["wind_direction"] * np.pi / 180
    )
    df_meo = df_meo.drop(columns=["wind_speed", "wind_direction"])
    # fill missing wind values with 0
    df_meo[["wind_x", "wind_y"]] = df_meo[["wind_x", "wind_y"]].fillna(0.0)
    # drop weather
    df_meo = df_meo.drop(columns=["weather"])
    # use index in df_meo_station as station_id
    df_meo["station_index"] = (
        df_meo.reset_index()
        .merge(
            df_meo_station.reset_index(),
            left_on="station_id",
            right_on="station_id",
            how="left",
        )
        .drop(columns=["station_id"])
        .set_index("utc_time")
        .sort_index()["index"]
    )
    # drop station_id
    df_meo = df_meo.drop(columns=["station_id"])
    # resample hourly and forward fill missing values within 24 hours
    df_meo = (
        df_meo.groupby("station_index")
        .apply(
            lambda df: df.resample("3h", origin="start_day").mean().ffill(limit=8),
            include_groups=False,
        )
        .reset_index("station_index")
    )
    return df_meo, df_meo_station


def parse_beijing(path: str = "./data/beijing"):
    data_dir = Path(path)
    df_aq = pd.read_csv(data_dir / "beijing_17_18_aq.csv")
    df_aq_station = pd.read_csv(
        data_dir / "Beijing_AirQuality_Stations_EN.txt",
        delimiter="\t",
        encoding="utf-16",
    )
    df_aq, df_aq_station = preprocess_aq(df_aq, df_aq_station)

    df_meo = pd.read_csv(data_dir / "beijing_17_18_meo.csv")
    df_meo_station = pd.read_csv(data_dir / "Beijing_MEO_Stations_cn.csv")
    df_meo, df_meo_station = preprocess_meo(df_meo, df_meo_station)

    common_index = df_aq.index.intersection(df_meo.index)
    df_aq = df_aq.loc[common_index]
    df_meo = df_meo.loc[common_index]

    # create feature tensors
    meo_data = (
        torch.from_numpy(
            np.stack(
                [
                    pd.pivot_table(
                        df_meo,
                        index="utc_time",
                        values=v,
                        columns="station_index",
                        dropna=False,
                    ).values
                    for v in [
                        "temperature",
                        "pressure",
                        "humidity",
                        "wind_x",
                        "wind_y",
                        # "weather",
                    ]
                ],
                axis=-1,
            )
        )
        .float()
        .transpose(0, 1)
    )

    aq_data = (
        torch.from_numpy(
            pd.pivot_table(
                df_aq,
                index="utc_time",
                values="PM2.5",
                columns="station_index",
                dropna=False,
            ).values
        )[..., None]
        .float()
        .transpose(0, 1)
    )

    # create position tensors
    meo_pos = torch.from_numpy(df_meo_station[["longitude", "latitude"]].values).float()
    aq_pos = torch.from_numpy(df_aq_station[["longitude", "latitude"]].values).float()

    return aq_data, meo_data, aq_pos, meo_pos


def project_aq(
    aq_data: torch.Tensor, aq_pos: torch.Tensor, positions: torch.Tensor, kernel: Kernel
):
    """
    Convert the given air quality data to the Reproducing Kernel Hilbert Space (RKHS) representation.

    Args:
        aq_data (S, T, 1): The air quality data to be converted.
        aq_pos (S, 2): The positions of the air quality data.
        positions (N, 2): The positions of the RKHS representation.
        kernel (Kernel): The kernel function to be used.

    Returns:
        torch.Tensor: The RKHS representation of the air quality data.
    """
    _, T, _ = aq_data.shape
    N = positions.shape[0]

    weights = torch.zeros(N, T, 1)
    mask = ~torch.isnan(aq_data).squeeze(-1)
    for t in range(T):
        aq_data_masked = aq_data[:, t][mask[:, t]].contiguous()
        aq_pos_masked = aq_pos[mask[:, t]].contiguous()

        K_xy = kernel(aq_pos_masked, positions) @ torch.eye(N)
        weights[:, t] = torch.linalg.lstsq(
            K_xy, aq_data_masked, driver="gelsd"
        ).solution
    return weights.reshape(N, T, 1)


def project_meo(
    meo_data: torch.Tensor,
    meo_pos: torch.Tensor,
    positions: torch.Tensor,
    kernel: Kernel,
    implementation: str = "lstsq",
):
    """
    Convert the given meteorological data to the Reproducing Kernel Hilbert Space (RKHS) representation.

    Args:
        meo_data (S, T, F): The meteorological data to be converted.

    Returns:
        torch.Tensor: The RKHS representation of the meteorological data.
    """
    N = positions.shape[0]
    S, T, F = meo_data.shape
    meo_pos = meo_pos.double().contiguous()
    meo_data = meo_data.reshape(S, T * F).double().contiguous()
    positions = positions.double()

    if implementation == "torch":
        K_xx = kernel(meo_pos, meo_pos) @ torch.eye(S, dtype=torch.float64)
        K_xy = kernel(meo_pos, positions) @ torch.eye(N, dtype=torch.float64)
        K_yy = kernel(positions, positions) @ torch.eye(N, dtype=torch.float64)
        # projection to RKHS at meo_pos
        weights = torch.linalg.solve(K_xx, meo_data)
        # sample at self.positions
        samples = K_xy.T @ weights
        # projection to RKHS at self.positions
        weights = torch.linalg.solve(K_yy, samples)
    elif implementation == "lstsq":
        K_xy = kernel(meo_pos, positions) @ torch.eye(N, dtype=torch.float64)
        weights = torch.linalg.lstsq(K_xy, meo_data, driver="gelsd").solution
    elif implementation == "pykeops":
        alpha = 0.01
        K_xx = kernel(meo_pos, meo_pos)
        K_yx = kernel(positions, meo_pos)
        K_yy = kernel(positions, positions)

        # projection to RKHS at meo_pos
        weights = K_xx.solve(
            meo_data,
            dtype_acc="float64",
            sum_scheme="kahan_scheme",
            alpha=alpha,
        )
        # sample at self.positions
        samples = K_yx @ weights
        # projection to RKHS at self.positions
        weights = K_yy.solve(
            samples,
            dtype_acc="float64",
            sum_scheme="kahan_scheme",
            alpha=alpha,
        )
        assert isinstance(weights, torch.Tensor)
    return weights.float().reshape(N, T, F)


class BeijingDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root: str = "./data", window_size=24, rkhs=True, sigma=0.1
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.rkhs = rkhs
        self.sigma = sigma

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        if self.train_dataset and self.val_dataset and self.test_dataset:
            return

        aq_data, meo_data, aq_pos, meo_pos = parse_beijing()

        # we use first self.window_size steps as input and the next window_size steps as target
        L = self.window_size * 2

        # valid: only contain samples where at least one of the stations is not nan
        valid = ~torch.isnan(aq_data).all(dim=0).squeeze(1)

        if self.rkhs:
            # project aq and meo data to RKHS
            positions = torch.cat([aq_pos, meo_pos], dim=0)
            kernel = GaussianKernel(0.1)

            N = positions.shape[0]
            T = aq_data.shape[1]

            aq_weights = torch.full((N, T, aq_data.shape[-1]), np.nan)
            meo_weights = torch.full((N, T, meo_data.shape[-1]), np.nan)
            aq_weights[:, valid] = project_aq(
                aq_data[:, valid], aq_pos, positions, kernel
            )
            meo_weights[:, valid] = project_meo(
                meo_data[:, valid], meo_pos, positions, kernel, implementation="lstsq"
            )

        # valid starting points after unfolding
        valid = valid.unfold(0, L, 1).all(dim=1)
        n_valid = int(valid.sum().item())

        aq_data_unfolded = aq_data.transpose(0, 1).unfold(0, L, 1)[valid]
        meo_data_unfolded = meo_data.transpose(0, 1).unfold(0, L, 1)[valid]
        aq_pos_expanded = aq_pos[None].expand(n_valid, -1, -1)
        meo_pos_expanded = meo_pos[None].expand(n_valid, -1, -1)
        original = TensorDataset(
            aq_data_unfolded, meo_data_unfolded, aq_pos_expanded, meo_pos_expanded
        )
        if self.rkhs:
            # do transpose since TensorDataset expects first dimension to be the same
            aq_weights_unfold = aq_weights.transpose(0, 1).unfold(0, L, 1)[valid]
            meo_weights_unfold = meo_weights.transpose(0, 1).unfold(0, L, 1)[valid]
            positions_expanded = positions[None].expand(n_valid, -1, -1)
            rkhs = TensorDataset(
                aq_weights_unfold, meo_weights_unfold, positions_expanded
            )
            dataset = StackDataset(original=original, rkhs=rkhs)
        else:
            original = original

        # train/val/test split is 0.7/0.1/0.2
        # split chronologically
        n_train = int(n_valid * 0.7)
        n_val = int(n_valid * 0.1)
        n_test = n_valid - n_train - n_val

        self.train_dataset = Subset(dataset, range(0, n_train))
        self.val_dataset = Subset(dataset, range(n_train, n_train + n_val))
        self.test_dataset = Subset(
            dataset, range(n_train + n_val, n_train + n_val + n_test)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=16
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=16
        )
