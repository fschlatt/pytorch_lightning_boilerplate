import argparse
from typing import Any, List, Optional

import optuna
import pytorch_lightning as pl
from pytorch_lightning import callbacks


class OptunaArg:
    def __init__(self, value: Any):
        try:
            self.value = int(value)
        except ValueError:
            try:
                self.value = float(value)
            except ValueError:
                self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def parse(trial: optuna.Trial, key: str, args: List["OptunaArg"]):
        values = [arg.value for arg in args]
        param, *values = values
        if param == "categorical":
            func = trial.suggest_categorical
        elif param == "int":
            func = trial.suggest_int
        elif param == "uniform":
            func = trial.suggest_uniform
        elif param == "loguniform":
            func = trial.suggest_loguniform
        elif param == "discrete_uniform":
            func = trial.suggest_discrete_uniform
        else:
            raise ValueError(
                "invalid optuna parameter type, expected one of "
                "{categorical, int, uniform, loguniform, discrete_uniform}, "
                f"got {param}"
            )
        func(key, *values)

    @staticmethod
    def parse_optuna_args(trial: optuna.Trial, args: argparse.Namespace):
        args = argparse.Namespace(**vars(args))
        args_dict = vars(args)
        for key, args_list in args_dict.items():
            if isinstance(args_list, list):
                if args_list and all(isinstance(arg, OptunaArg) for arg in args_list):
                    sampled_arg = OptunaArg.parse(trial, key, args_list)
                    args_dict[key] = sampled_arg
        return args


class PyTorchLightningPruningCallback(callbacks.EarlyStopping):
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: Optional[str] = None,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
    ):
        super(PyTorchLightningPruningCallback, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        stop_training = super().on_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if isinstance(current_score, float):
            self.trial.report(current_score, step=epoch)
            if self.trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(epoch)
                raise optuna.exceptions.TrialPruned(message)
        return stop_training
