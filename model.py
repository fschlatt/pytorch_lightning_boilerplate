import argparse

import optuna_helpers
import pytorch_lightning as pl
import torch.utils.data
import torch


class Model(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(Model).__init__()
        self.save_hyperparameters(args)

    def load_train_data(self):
        self.train_data = None

    def load_val_data(self):
        self.val_data = None

    def load_test_data(self):
        self.test_data = None

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

    def prepare_data(self):
        self.train_data = self.val_data = self.test = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, self.hparams.batch_size, shuffle=self.hparams.shuffle
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, self.hparams.val_batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, self.hparams.test_batch_size)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        group = parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")

        group.add_argument(
            "--lr", type=optuna_helpers.OptunaArg, nargs="+", default=1e-3
        )

        return parser
