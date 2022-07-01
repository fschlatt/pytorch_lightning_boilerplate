import argparse

import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics

import optuna_helpers


class LossTensor(torchmetrics.Metric):
    def __init__(self) -> None:
        self.add_state("loss", torch.tensor(0))
        self.add_state("counter", torch.tensor(0))

    def update(self, loss: torch.Tensor) -> None:
        self.loss: torch.Tensor = self.loss + loss
        self.counter: torch.Tensor = self.counter + 1

    def compute(self) -> torch.Tensor:
        return self.loss / self.counter


class Model(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)

    def forward(self, inp: torch.Tensor):
        ...

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...

    def training_step(self, data_batch, batch_i):
        loss = self.loss()

        # logging over accumulate grad batches
        self.loss_metric(loss)
        if not self.trainer.fit_loop.should_accumulate():
            self.log("loss", self.loss_metric.compute())
            self.loss_metric.reset()

    def validation_step(self, data_batch, batch_i):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, data_batch, batch_i):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = None
        scheduler_config = {
            "interval": self.hparams.lr_interval,
            "frequency": self.hparams.lr_frequency,
            "monitor": self.hparams.monitor,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **scheduler_config},
        }

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")
        # add model specific arguments here arguments

        group = parser.add_argument_group(
            f"{cls.__module__}.{cls.__qualname__}.Optimizer"
        )
        group.add_argument(
            "--learning_rate",
            type=optuna_helpers.OptunaArg.parse,
            nargs="+",
            default=1e-3,
            help="learning rate for model",
        )
        group.add_argument(
            "--lr_interval",
            type=str,
            default="epoch",
            choices=("epoch", "batch"),
            help="interval for calling lr scheduler",
        )
        group.add_argument(
            "--lr_frequency",
            type=int,
            default=1,
            help="frequency for calling lr scheduler",
        )

        return parser
