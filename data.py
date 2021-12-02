from typing import Optional
import torch.utils.data
import pytorch_lightning as pl


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        *,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        shuffle: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.shuffle = shuffle

    def prepare_data(self) -> None:
        # download dataset or copy to SSD or something
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = torch.utils.data.Dataset()
            self.val_dataset = torch.utils.data.Dataset()
        if stage in (None, "test"):
            self.test_dataset = torch.utils.data.Dataset()
        if stage in (None, "predict"):
            self.predict_dataset = torch.utils.data.Dataset()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.eval_batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.eval_batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.eval_batch_size
        )
