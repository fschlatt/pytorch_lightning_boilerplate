import argparse

import optuna_helpers
import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model).__init__()
        self.hparams = hparams
        torch.manual_seed(hparams.seed)

    def forward(self, inp):
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
            self.train_data, self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, self.hparams.val_batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, self.hparams.test_batch_size
        )

    @staticmethod
    def add_model_specific_args(parser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument(
            '--lr', type=optuna_helpers.OptunaArg, nargs='+', default=1e-3
        )

        return parser
