from torch.optim import SGD, Adam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import pytorch_lightning as pl
import torchvision

from augmentation import Augment
from model import AddProjectionHead
from contrastiveloss import ContrastiveLoss


def define_param_groups(model, weight_decay, optimizer_name):
    """
    Defines parameter groups for optimization, considering weight decay and layer adaptation.

    Args:
        model (nn.Module): The neural network model.
        weight_decay (float): Weight decay coefficient for regularization.
        optimizer_name (str): Name of the optimizer.

    Returns:
        list: List of parameter groups for optimization, each containing relevant parameters, weight decay, and layer adaptation flag.
    """
    param_groups = []
    
    for name, p in model.named_parameters():
        exclude = ('bn' in name) or ((optimizer_name == 'lars') and ('bias' in name))
        group = {
            'params': [p],
            'weight_decay': weight_decay if not exclude else 0.,
            'layer_adaptation': not exclude,
        }
        param_groups.append(group)
    
    return param_groups


class SimCLR(pl.LightningModule):
    """
    Initializes a SimCLR model for self-supervised learning.

    Args:
        backbone (torchvision.models): Pretrained neural network architecture.
        image_size (int, optional): Input image size for augmentations. Defaults to 224.
        embedding_size (int, optional): Dimension of the embedding space. Defaults to 128.
        head_type (str, optional): Type of projection head linearity. Defaults to "nonlinear".
        temperature (float, optional): Temperature parameter for contrastive loss. Defaults to 0.5.
        batch_size (int, optional): Training batch size. Defaults to 64.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 1000.
        optimizer_name (str, optional): Name of the optimizer. Only "adam" or "lars" are supported. Defaults to "adam".
        learning_rate (float, optional): Learning rate for optimization. Defaults to 0.2.
        weight_decay (float, optional): Weight decay coefficient for regularization. Defaults to 0.0001.
    """
    def __init__(
            self,
            backbone: torchvision.models,
            image_size: int=224,
            embedding_size: int=128,
            head_type: str="nonlinear",
            temperature: float=0.5, 
            batch_size: int=64,
            max_epochs: int=1000,
            optimizer_name: str="adam",
            learning_rate: float=0.2,
            weight_decay: float=0.0001
    ):
        self.max_epochs = max_epochs

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = AddProjectionHead(backbone, embedding_size, head_type)
        self.augment = Augment(image_size)

        self.loss = ContrastiveLoss(batch_size, temperature)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input data.
            
        Returns:
            torch.Tensor: Output embeddings.
        """
        return self.model(x)
    
    def _common_step(self, batch, batch_idx):
        x, _ = batch
        #
        x_i, x_j = self.augment(x)
        #
        z_i, z_j = self.model(x_i), self.model(x_j)

        #
        loss = self.loss(z_i, z_j)

        return loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx % 100 == 0:
            x = x[:4]
            grid = torchvision.utils.make_grid(x.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("shoes images", grid, self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)

        return loss
    
    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate scheduler.
        
        Returns:
            tuple: Tuple containing optimizer(s) and scheduler(s).
        """
        param_groups = define_param_groups(
            model=self.model,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer_name
        )
        
        if self.optimizer_name == "lars":
            # do zrobienia
            optimizer = None
        else:
            optimizer = Adam(param_groups, lr = self.learning_rate, weight_decay=self.weight_decay)

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.max_epochs, warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]