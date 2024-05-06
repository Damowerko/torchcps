import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from torchcps.datasets.aq import BeijingDataModule
from torchcps.kernel.nn import KNN, sample_kernel
from torchcps.kernel.rkhs import GaussianKernel, Mixture
from torchcps.utils import add_model_specific_args, make_trainer


class AQModel(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        window_size: int = 72,
        sigma: float = 0.05,
        hidden_channels: int = 512,
        n_layers: int = 4,
        n_layers_mlp: int = 2,
        hidden_channels_mlp: int = 1024,
        max_filter_kernels: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.window_size = window_size
        self.sigma = sigma
        self.lr = lr
        self.weight_decay = weight_decay
        self.knn = KNN(
            n_dimensions=2,
            in_channels=6 * self.window_size,
            hidden_channels=hidden_channels,
            out_channels=1 * self.window_size,
            n_layers=n_layers,
            n_layers_mlp=n_layers_mlp,
            hidden_channels_mlp=hidden_channels_mlp,
            sigma=sigma,
            max_filter_kernels=max_filter_kernels,
            update_positions=False,
            alpha=None,
            kernel_init="uniform",
        )

    def forward(self, batch) -> torch.Tensor:
        _, _, aq_pos, _ = batch["original"]
        aq_weights, meo_weights, positions = batch["rkhs"]
        B, S = positions.shape[:2]
        # use the first half of the data to predict the aq in the second half
        x = torch.cat(
            [
                aq_weights[..., : self.window_size],
                meo_weights[..., : self.window_size],
            ],
            dim=-2,
        )
        # flatten the temporal and feature dimensions
        x_mixture = Mixture(
            positions.view(S * B, -1).contiguous(),
            x.view(B * S, -1).contiguous(),
            batch=torch.tensor([S] * B).to(x.device),
        )
        y_hat_mixture = self.knn.forward(x_mixture)
        # sample y_hat_mixture at aq_pos
        kernel = GaussianKernel(self.sigma)
        y_hat = sample_kernel(
            kernel,
            y_hat_mixture,
            aq_pos.view(-1, 2).contiguous(),
            torch.tensor([aq_pos.shape[1]] * B).to(x.device),
        ).weights
        return y_hat

    def training_step(self, batch):
        aq_data, _, _, _ = batch["original"]
        y_hat = self(batch)
        y = aq_data[..., self.window_size :].view(y_hat.shape)
        error = y_hat - y
        # ignore missing values in aq_data
        error = error[~torch.isnan(error)]
        mse = error.pow(2).nanmean()
        mae = error.abs().nanmean()

        self.log("train/rmse", mse.sqrt(), prog_bar=True)
        self.log("train/mae", mae, prog_bar=True)

        return mse

    def validation_step(self, batch, batch_idx):
        aq_data, _, _, _ = batch["original"]
        y_hat = self(batch)
        y = aq_data[..., self.window_size :].view(y_hat.shape)
        error = y_hat - y
        # ignore missing values in aq_data
        error = error[~torch.isnan(error)]
        mse = error.pow(2).nanmean()
        mae = error.abs().nanmean()

        self.log("val/rmse", mse.sqrt(), prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)

        return mse

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def train(params: argparse.Namespace):
    model = AQModel(**vars(params))
    dm = BeijingDataModule(
        window_size=params.window_size, rkhs=params.rkhs, sigma=params.sigma
    )
    trainer = make_trainer("aq", params)
    trainer.fit(model, dm)


def test(params):
    raise NotImplementedError()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # parse arguments, first argument is either "train" or "test"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # train subparser
    train_parser = subparsers.add_parser("train")

    group = train_parser.add_argument_group("Model")
    AQModel.add_model_specific_args(group)

    group = train_parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=100)
    group.add_argument("--patience", type=int, default=20)
    group.add_argument("--profiler", type=str, default=None)
    group.add_argument("--fast_dev_run", action="store_true")
    group.add_argument("--grad_clip_val", type=float, default=0)

    # test subparser
    test_parser = subparsers.add_parser("test")

    # run code
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
