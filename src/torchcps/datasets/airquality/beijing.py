from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_cluster.grid import grid_cluster
from torch_geometric.nn.pool import avg_pool_x

from torchcps.kernel.nn import sample_kernel, solve_kernel
from torchcps.kernel.rkhs import GaussianKernel, Kernel, Mixture, projection


def preprocess_aq(df_aq, df_aq_station) -> tuple[pd.DataFrame, pd.DataFrame]:
    # preprocess aq data
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
    # resample hourly and forward fill up to 24 hours
    df_aq = (
        df_aq.groupby("station_index")
        .apply(
            lambda df: df.resample("h").first().ffill(limit=24), include_groups=False
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
    # convert weather to category
    df_meo["weather"] = df_meo.weather.factorize()[0]
    # use index in df_meo_station as station_id
    df_meo["station_index"] = (
        df_meo.reset_index()
        .merge(
            df_meo_station.reset_index(),
            left_on="station_id",
            right_on="station_id",
            how="left",
        )
        .set_index("utc_time")
        .sort_index()["index"]
    )
    # resample hourly and forward fill missing values within 24 hours
    df_meo = (
        df_meo.groupby("station_index")
        .apply(
            lambda df: df.resample("h").first().ffill(limit=24), include_groups=False
        )
        .reset_index("station_index")
    )
    return df_meo, df_meo_station


def temporal_intersection(df_aq, df_meo):
    common_index = df_aq.index.intersection(df_meo.index)
    return df_aq.loc[common_index], df_meo.loc[common_index]


class BeijingDataset(Dataset):
    def __init__(self, data_root: str = "./data", duration=24) -> None:
        super().__init__()
        data_dir = Path(data_root) / "beijing"
        self.duration = duration

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

        df_aq, df_meo = temporal_intersection(df_aq, df_meo)

        # create feature tensors
        self.meo_data = (
            torch.from_numpy(
                np.stack(
                    [
                        pd.pivot_table(
                            df_meo, index="utc_time", values=v, columns="station_index"
                        ).values
                        for v in [
                            "temperature",
                            "pressure",
                            "humidity",
                            "wind_x",
                            "wind_y",
                            "weather",
                        ]
                    ],
                    axis=-1,
                )
            )
            .float()
            .transpose(0, 1)
        )
        self.aq_data = (
            torch.from_numpy(
                pd.pivot_table(
                    df_aq, index="utc_time", values="PM2.5", columns="station_index"
                ).values
            )[..., None]
            .float()
            .transpose(0, 1)
        )

        # create position tensors
        self.meo_pos = torch.from_numpy(
            df_meo_station[["longitude", "latitude"]].values
        ).float()
        self.aq_pos = torch.from_numpy(
            df_aq_station[["longitude", "latitude"]].values
        ).float()

    def __len__(self):
        return len(self.aq_data) - self.duration

    def __getitem__(self, idx):
        return (
            self.aq_data[:, idx : idx + self.duration],
            self.meo_data[:, idx : idx + self.duration],
            self.aq_pos,
            self.meo_pos,
        )


class BeijingDatasetRKHS:
    def __init__(self, beijing: BeijingDataset, sigma=0.1):
        self.beijing = beijing
        self.meo_pos = beijing.meo_pos
        self.aq_pos = beijing.aq_pos
        self.kernel = GaussianKernel(sigma)

        # find positions with grid clustering
        self.positions = torch.cat([beijing.aq_pos, beijing.meo_pos], dim=0)
        # cluster = grid_cluster(positions, torch.Tensor([0.05] * 2))
        # self.positions = avg_pool_x(cluster, positions, torch.zeros(len(positions)))[0]

    def __len__(self):
        return len(self.beijing)

    def __getitem__(self, idx):
        aq_data, meo_data, _, _ = self.beijing[idx]

        aq_data = self.aq_to_rkhs(aq_data)
        meo_data = self.meo_to_rkhs(meo_data)

        return aq_data.float(), meo_data.float(), self.positions.float()

    def aq_to_rkhs(self, aq_data: torch.Tensor):
        """
        Convert the given air quality data to the Reproducing Kernel Hilbert Space (RKHS) representation.

        Args:
            aq_data (S, T, 1): The air quality data to be converted.

        Returns:
            torch.Tensor: The RKHS representation of the air quality data.
        """
        S, T, _ = aq_data.shape
        N = self.positions.shape[0]

        weights = torch.zeros(N, T, 1)
        mask = ~torch.isnan(aq_data).squeeze(-1)
        for t in range(T):
            aq_data_masked = aq_data[:, t][mask[:, t]]
            aq_pos_masked = self.aq_pos[mask[:, t]]

            K_xy = self.kernel(aq_pos_masked, self.positions) @ torch.eye(N)
            weights[:, t] = torch.linalg.lstsq(
                K_xy, aq_data_masked, driver="gelsd"
            ).solution
        return weights.reshape(N, T, 1)

    def meo_to_rkhs(
        self,
        meo_data: torch.Tensor,
        implementation="torch",
    ):
        """
        Convert the given meteorological data to the Reproducing Kernel Hilbert Space (RKHS) representation.

        Args:
            meo_data (S, T, F): The meteorological data to be converted.

        Returns:
            torch.Tensor: The RKHS representation of the meteorological data.
        """
        N = self.positions.shape[0]
        S, T, F = meo_data.shape
        meo_data = meo_data.reshape(S, T * F)

        if implementation == "torch":
            K_xx = self.kernel(self.meo_pos, self.meo_pos) @ torch.eye(S)
            K_xy = self.kernel(self.meo_pos, self.positions) @ torch.eye(N)
            K_yy = self.kernel(self.positions, self.positions) @ torch.eye(N)
            # projection to RKHS at meo_pos
            weights = torch.linalg.solve(K_xx, meo_data)
            # sample at self.positions
            samples = K_xy.T @ weights
            # projection to RKHS at self.positions
            weights = torch.linalg.solve(K_yy, samples)
        elif implementation == "lstsq":
            K_xy = self.kernel(self.meo_pos, self.positions) @ torch.eye(N)
            weights = torch.linalg.lstsq(K_xy, meo_data, driver="gelsd").solution
        elif implementation == "pykeops":
            raise NotImplementedError(
                "PyKeOps implementation does not work for illconditioned matrices."
            )
            alpha = 1e-10
            K_xx = self.kernel(self.meo_pos, self.meo_pos)
            K_yx = self.kernel(self.positions, self.meo_pos)
            K_yy = self.kernel(self.positions, self.positions)

            # projection to RKHS at meo_pos
            weights = K_xx.solve(
                meo_data,
                dtype_acc="float64",
                sum_scheme="kanhan_scheme",
                alpha=alpha,
            )
            # sample at self.positions
            samples = K_yx @ weights
            # projection to RKHS at self.positions
            weights = K_yy.solve(
                meo_data,
                dtype_acc="float64",
                sum_scheme="kanhan_scheme",
                alpha=alpha,
            )
            assert isinstance(weights, torch.Tensor)
        return weights.reshape(N, T, F)
