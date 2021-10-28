import argparse

import optuna
import optuna_helpers
import pytorch_lightning as pl
from model import Model


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:

    args = optuna_helpers.OptunaArg.parse_optuna_args(trial, args)
    kwargs = vars(args)

    early_stop = optuna_helpers.PyTorchLightningPruningCallback(trial, **kwargs)
    kwargs["early_stop_callback"] = early_stop

    trainer = pl.Trainer(**kwargs)
    model = Model(args)

    trainer.fit(model)

    return early_stop.best_score


def main():

    parser = argparse.ArgumentParser()

    def str_to_bool(string: str):
        return str(string).lower() in ("true", "1", "y", "yes")

    parser.add_argument("--direction", type=int, default="minimize")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--pruning", type=str_to_bool, default=True)
    parser.add_argument("--shuffle", type=str_to_bool, default=True)
    parser.add_argument("--timeout", type=int, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)

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
