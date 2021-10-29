import argparse

import optuna
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.utilities import argparse as pl_argparse_utils

import optuna_helpers
from data import Dataset
from model import Model


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:

    args = optuna_helpers.OptunaArg.parse_optuna_args(trial, args)
    kwargs = vars(args)

    checkpoint = pl_callbacks.ModelCheckpoint(
        monitor=args.monitor, save_last=args.save_last
    )

    early_stop = optuna_helpers.PyTorchLightningPruningCallback(trial, **kwargs)
    callbacks = [early_stop, checkpoint]
    kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(**kwargs)
    model = Model(args)

    trainer.fit(model)

    return early_stop.best_score


def main():

    parser = argparse.ArgumentParser()

    def str_to_bool(string: str):
        return str(string).lower() in ("true", "1", "y", "yes")

    group = parser.add_argument_group("optuna.Study")
    group.add_argument("--n_trials", type=int, default=1)
    group.add_argument("--pruning", type=str_to_bool, default=True)
    group.add_argument("--timeout", type=int, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = pl_argparse_utils.add_argparse_args(pl_callbacks.ModelCheckpoint, parser)
    parser = Model.add_model_specific_args(parser)
    parser = Dataset.add_argparse_args(parser)

    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction=args.direction, pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
