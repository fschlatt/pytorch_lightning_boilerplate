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
        return func(key, *values)

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


class OptunaPruningCallback(callbacks.EarlyStopping):
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
        super(OptunaPruningCallback, self).__init__(
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
        self.pruned = False

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        if self.pruned:
            raise optuna.exceptions.TrialPruned(
                f"Trial was pruned at step {trainer.global_step}"
            )

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        step = trainer.global_step
        self.trial.report(current.item(), step)
        should_prune = self.trial.should_prune()
        self.pruned = should_prune
        if should_prune:
            reason = f"Trial was pruned at step {step}"

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)
