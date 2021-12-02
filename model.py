import argparse

import pytorch_lightning as pl
import torch
import torch.utils.data

import optuna_helpers


class Model(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(Model).__init__()
        self.save_hyperparameters(args)

    def forward(self, inp: torch.Tensor):
        raise NotImplementedError()

    def training_step(self, data_batch, batch_i):
        pass

    def validation_step(self, data_batch, batch_i):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, data_batch, batch_i):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)]

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        group = parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")

        group.add_argument(
            "--lr", type=optuna_helpers.OptunaArg, nargs="+", default=1e-3
        )

        return parser
