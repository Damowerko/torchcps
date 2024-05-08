import argparse
import inspect
import typing
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
from wandb.wandb_run import Run


def get_init_arguments_and_types(cls):
    """

    Args:
        cls: class to get init arguments from

    Returns:
        list of tuples (name, type, default)
    """
    parameters = inspect.signature(cls).parameters
    args = []
    for name, parameter in parameters.items():
        args.append((name, parameter.annotation, parameter.default))
    return args


def add_model_specific_args(cls, group):
    for base in cls.__bases__:
        if hasattr(base, "add_model_specific_args"):
            group = base.add_model_specific_args(group)  # type: ignore
    args = get_init_arguments_and_types(cls)  # type: ignore
    for name, type, default in args:
        if default is inspect.Parameter.empty:
            continue
        if type not in (int, float, str, bool):
            continue
        if type == bool:
            group.add_argument(f"--{name}", dest=name, action="store_true")
        else:
            group.add_argument(f"--{name}", type=type, default=default)
    return group


def make_trainer(project: str, params: argparse.Namespace, callbacks=[]) -> pl.Trainer:
    if params.no_log:
        logger = False
    else:
        # create loggers
        logger = WandbLogger(
            project=project,
            save_dir="logs",
            config=params,
            log_model=True,
            notes=params.notes,
        )
        logger.log_hyperparams(params)
        run = typing.cast(Run, logger.experiment)
        run.log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                path.endswith(".py") and "logs" not in path and ("src" in path)
            ),
        )
        callbacks += [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=f"./checkpoints/{run.id}",
                filename="best",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ]
    callbacks += [EarlyStopping(monitor="val/loss", patience=params.patience)]

    # configure profiler
    if params.profiler == "advanced":
        profiler = AdvancedProfiler(dirpath=".", filename="profile")
    elif params.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=".", export_to_chrome=True, sort_by_key="cuda_time_total"
        )
    else:
        profiler = params.profiler

    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=not params.no_log,
        precision=32,
        devices=1,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        profiler=profiler,
        fast_dev_run=params.fast_dev_run,
        gradient_clip_val=params.grad_clip_val,
    )
