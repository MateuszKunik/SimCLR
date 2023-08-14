import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(
            self,
            target_dir: Path,
            transform: transforms.Compose=None
    ):
        super(CustomDataset, self).__init__()
        self.paths = list(target_dir.glob("*/*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = Image.open(self.paths[index])

        if self.transform:
            image = self.transform(image)

        return image


class CustomDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            transform: transforms.Compose=None,
            batch_size: int=32,
            num_workers: int=2
    ):
        super(CustomDataModule, self).__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(
                target_dir=self.data_dir / "Train",
                transform=self.transform
            )

            self.valid_dataset = CustomDataset(
                target_dir=self.data_dir / "Valid",
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = CustomDataset(
                target_dir=self.data_dir / "Test",
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )