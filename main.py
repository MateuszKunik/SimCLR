import torchvision
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.loggers import TensorBoardLogger

import config
from engine import SimCLR
from callbacks import PrintingCallback


if __name__ == "__main__":
    time = datetime.now().strftime("%Y%m%d%H%M")
    logger = TensorBoardLogger("tb_logs", name=f"model_{time}")

    weights = torchvision.models.ResNet18_Weights.DEFAULT
    backbone = torchvision.models.resnet18(weights)
    
    backbone_transforms = weights.transforms()

    # data_module = IntelDataModule(
    #     data_dir=config.DATA_DIR,
    #     train_transform=train_transforms,
    #     test_transform=backbone_transforms,
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS
    # )

    model = SimCLR()

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        logger=logger,
        callbacks=[
            PrintingCallback(),
            EarlyStopping(monitor="valid_loss")
        ]
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)