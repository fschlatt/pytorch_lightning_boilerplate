import argparse

import optuna
import pytorch_lightning as pl


class OptunaArg():

    def __init__(self, val):
        try:
            self.val = int(val)
        except ValueError:
            try:
                self.val = float(val)
            except ValueError:
                self.val = val


    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    @staticmethod
    def parse(trial, key, args):
        error_msg = 'invalid number of arguments for `{}`, expected {}, got {}'
        param = args[-1].val
        args = [arg.val for arg in args[:-1]]
        if param == 'categorical':
            return trial.suggest_categorical(key, args)
        elif param == 'int':
            if len(args) == 1:
                return args[0]
            assert len(args) == 2, error_msg.format(
                'suggest_int', 2, len(args)
            )
            return trial.suggest_int(key, *args)
        elif param == 'uniform':
            if len(args) == 1:
                return args[0]
            assert len(args) == 2, error_msg.format(
                'suggest_uniform', 2, len(args)
            )
            return trial.suggest_uniform(key, *args)
        elif param == 'loguniform':
            if len(args) == 1:
                return args[0]
            assert len(args) == 2, error_msg.format(
                'suggest_loguniform', 2, len(args)
            )
            return trial.suggest_loguniform(key, *args)
        elif param == 'discrete_uniform':
            if len(args) == 1:
                return args[0]
            assert len(args) == 2, error_msg.format(
                'suggest_discrete_uniform', 3, len(args)
            )
            return trial.suggest_discrete_uniform(key, *args)
        else:
            raise ValueError(
                'invalid optuna parameter type, expected one of '
                '{categorical, int, uniform, loguniform, discrete_uniform}, '
                f'got {param}'
            )


def parse_optuna_args(trial, args):
    args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    for key, args_list in args_dict.items():
        if isinstance(args_list, list):
            if args_list and all(isinstance(arg, OptunaArg) for arg in args_list):
                sampled_arg = OptunaArg.parse(trial, key, args_list)
                args_dict[key] = sampled_arg
    return args


class PyTorchLightningPruningCallback(pl.callbacks.EarlyStopping):

    def __init__(self, trial, monitor='val_loss', min_delta=0.0,
                 patience=3, verbose=False, mode='auto', strict=True,
                 **kwargs):
        super(PyTorchLightningPruningCallback, self).__init__(
            monitor=monitor, min_delta=min_delta, patience=patience,
            verbose=verbose, mode=mode, strict=strict
        )

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, trainer, pl_module):
        stop_training = super().on_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.exceptions.TrialPruned(message)
        return stop_training
