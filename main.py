import torchvision
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger

import config
from callbacks import *
from engine import SimCLR
from data import CustomDataModule


if __name__ == "__main__":
    time = datetime.now().strftime("%Y%m%d%H%M")
    logger = TensorBoardLogger("tb_logs", name=f"model_{time}")

    weights = torchvision.models.ResNet18_Weights.DEFAULT
    backbone = torchvision.models.resnet18(weights=weights)
    
    backbone_transforms = weights.transforms()

    data_module = CustomDataModule(
        data_dir=config.data_dir,
        transform=backbone_transforms,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    model = SimCLR(
        backbone=backbone,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        min_epochs=1,
        max_epochs=config.max_epochs,
        precision=config.precision,
        logger=logger,
        callbacks=[
            print_callback,
            earlystopper,
            accumulator  
        ]
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)