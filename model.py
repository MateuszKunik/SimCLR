import torchvision
from torch import nn


class TinyLinearBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int
    ):
        """
        TinyLinearBlock class for creating a linear block using a neural network.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__()
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
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        head_type: str
    ):
        """
        ProjectionHead class tailored for the SimCLR framework.
        
        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features for nonlinear projection.
            out_features (int): Number of output features (embedding dimensions).
            head_type (str): Projection head linearity type ("linear" or "nonlinear").
        """
        super().__init__()
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

    def foward(self, x):
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
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # freezing all the weights
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # creating a projection head
        self.projection = ProjectionHead(
            self.backbone.fc.in_features,
            self.backbone.fc.in_features,
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