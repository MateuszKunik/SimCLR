from torch import nn
from torch.optim import SGD, Adam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import pytorch_lightning as pl
import torchvision

from augmentation import Augment
from contrastiveloss import ContrastiveLoss



class TinyLinearBlock(nn.Module):
    """
    TinyLinearBlock class for creating a linear block using a neural network.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int
    ):
        super(TinyLinearBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear_block = nn.Sequential(
            nn.Linear(
                self.in_features,
                self.out_features
            ),
            nn.BatchNorm1d(self.out_features)
        )

    def forward(self, x):
        """
        Forward pass through the TinyLinearBlock.

        Args:
            x (torch.Tensor): Input data to be passed through the linear block.

        Returns:
            torch.Tensor: Output of the linear block.
        """
        return self.linear_block(x)


class ProjectionHead(nn.Module):
    """
    ProjectionHead class tailored for the SimCLR framework.
    
    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features for nonlinear projection.
        out_features (int): Number of output features (embedding dimensions).
        head_type (str): Projection head linearity type ("linear" or "nonlinear").
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        head_type: str
    ):
        super(ProjectionHead, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.head_type = head_type

        # creating a multilayer perceptron 
        if self.head_type == "linear":
            self.mlp = TinyLinearBlock(self.in_features, self.out_features)
            
        elif self.head_type == "nonlinear":
            self.mlp = nn.Sequential(
                TinyLinearBlock(
                    self.in_features,
                    self.hidden_features
                ),
                nn.ReLU(),
                TinyLinearBlock(
                    self.hidden_features,
                    self.out_features
                ),
            )

    def forward(self, x):
        """
        Forward pass through the ProjectionHead.
        
        Args:
            x (torch.Tensor): Input data to be transformed into embeddings.
            
        Returns:
            torch.Tensor: Embeddings produced by the projection head.
        """
        x = self.mlp(x)
        return x


class AddProjectionHead(nn.Module):
    """
    A PyTorch module that combines a backbone model and a projection head for feature embedding.

    This class creates an instance that consists of a backbone neural network followed by a projection head.
    The backbone's final fully connected layer is replaced with an identity function, and its weights are frozen.
    The projection head transforms features from the backbone into embeddings of the desired dimensionality.

    Args:
        backbone (torchvision.models): A neural network model serving as the backbone for feature extraction.
        embedding_size (int, optional): The desired dimensionality of the resulting embeddings. Defaults to 128.
        head_type (str, optional): Type of projection head, either "linear" or "nonlinear". Defaults to "nonlinear".

    Attributes:
        backbone (torchvision.models): The chosen backbone model with its final fully connected layer replaced by nn.Identity().
        projection (ProjectionHead): An instance of the ProjectionHead class for transforming backbone features into embeddings.

    """
    def __init__(
            self,
            backbone: torchvision.models,
            embedding_size: int=128,
            head_type: str="nonlinear"
    ):
        super(AddProjectionHead, self).__init__()
        self.backbone = backbone
        self.backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # freezing all the weights
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # creating a projection head
        self.projection = ProjectionHead(
            self.backbone_features,
            self.backbone_features,
            embedding_size,
            head_type
        )

    def forward(self, x):
        """
        Transform input through the model.

        Args:
            x (torch.Tensor): Input tensor with dimensions (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embedding tensor with dimensions (batch_size, embedding_size).
        """
        h = self.backbone(x)
        z = self.projection(h)
        return z
    

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
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.max_epochs = max_epochs

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = AddProjectionHead(self.backbone, embedding_size, head_type)
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
        x = batch
        #
        x_i, x_j = self.augment(x)
        #
        z_i, z_j = self.model(x_i), self.model(x_j)

        #
        loss = self.loss(z_i, z_j)

        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch
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