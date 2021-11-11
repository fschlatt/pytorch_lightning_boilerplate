import argparse

import optuna
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.utilities import argparse as pl_argparse_utils

import optuna_helpers
import util
from data import Dataset
from model import Model


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:

    args = optuna_helpers.OptunaArg.parse_optuna_args(trial, args)
    kwargs = vars(args)

    checkpoint = pl_callbacks.ModelCheckpoint(
        dirpath=args.dirpath,
        filename=args.filename,
        monitor=args.monitor,
        verbose=args.verbose,
        save_last=args.save_last,
        save_top_k=args.save_top_k,
        save_weights_only=args.save_weights_only,
        mode=args.mode,
        auto_insert_metric_name=args.auto_insert_metric_name,
        every_n_train_steps=args.every_n_train_steps,
        # train_time_interval=args.train_time_interval, # TODO
        every_n_epochs=args.every_n_epochs,
        save_on_train_epoch_end=args.save_on_train_epoch_end,
        period=args.period,
        every_n_val_epochs=args.every_n_val_epochs,
    )

    early_stop = optuna_helpers.PyTorchLightningPruningCallback(
        trial,
        monitor=args.monitor,
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=args.verbose,
        mode=args.mode,
        strict=args.strict,
        check_finite=args.check_finite,
        stopping_threshold=args.stopping_threshold,
        divergence_threshold=args.divergence_threshold,
        check_on_train_epoch_end=args.check_on_train_epoch_end,
    )
    callbacks = [early_stop, checkpoint]
    kwargs["callbacks"] = callbacks

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    model = Model(args)
    dataset = Dataset(**kwargs)

    trainer.fit(model, datamodule=dataset)

    return early_stop.best_score


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    conflict_tracker = util.ArgumentConflictSolver()
    argparse._ActionsContainer.add_argument = conflict_tracker.catch_conflicting_args(
        argparse._ActionsContainer.add_argument
    )

    parser = pl.Trainer.add_argparse_args(parser)
    parser = pl_argparse_utils.add_argparse_args(pl_callbacks.ModelCheckpoint, parser)
    parser = pl_argparse_utils.add_argparse_args(pl_callbacks.EarlyStopping, parser)
    parser = Model.add_model_specific_args(parser)
    parser = Dataset.add_argparse_args(parser)

    group = parser.add_argument_group("Optuna")
    group.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help=(
            "The number of trials. If this argument is set to None, there is no"
            " limitation on the number of trials. If timeout is also set to None, the"
            " study continues to create trials until it receives a termination signal"
            " such as Ctrl+C or SIGTERM, default 1"
        ),
    )
    parser.add_argument(
        "--pruning",
        dest="pruning",
        action="store_true",
        help="if true activates experiment pruning, default false",
    )
    group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Stop study after the given number of second(s). If this argument is set"
            "to None, the study is executed without time limitation. If n_trials is"
            " also set to None, the study continues to create trials until it receives"
            " a termination signal such as Ctrl+C or SIGTERM, default None"
        ),
    )

    group = parser.add_argument_group("Global")
    conflict_tracker.resolve_conflicting_args(
        group,
        {
            "--monitor": "quantity to monitor for early stopping and checkpointing",
            "--verbose": "verbosity mode, default False",
            "--mode": (
                "one of {min, max}, dictates if early stopping and checkpointing "
                "considers maximum or minimum of monitored quantity"
            ),
        },
    )
    for option_string, actions in conflict_tracker.conflicting_args.items():
        if option_string not in parser._option_string_actions:
            raise argparse.ArgumentError(
                actions.pop(), "missing global argument for conflicting argument"
            )

    return parser


def main():

    parser = create_argument_parser()
    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="maximize" if args.mode == "max" else "minimize", pruner=pruner
    )
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
