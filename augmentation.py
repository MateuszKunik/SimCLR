import torchvision.transforms as T
from torch import nn


class Augment:
    """
    A stochastic data augmentation module that generates two correlated views of the same example.

    This module applies a series of random transformations to an input data example, resulting in two versions of
    the same example that are correlated. It is designed for use in self-supervised learning method - SimCLR.

    Args:
        image_size (int): Desired square size of the output image. If int, a square image is assumed.

    Attributes:
        train_transform (nn.Sequential): A sequence of random data augmentation transformations.
    """
    def __init__(
            self,
            image_size: int
    ):
        self.train_transform = nn.Sequential(
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [
                    T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
                ], p=0.8),
            T.RandomApply(
                [
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
                ],p=0.5),
        )
        
    def __call__(self, x):
        """
        Apply the stochastic data augmentation to the input data.

        Args:
            x (torch.Tensor): Input data to be augmented.

        Returns:
            tuple: A tuple containing two augmented versions of the input data.
        """
        return self.train_transform(x), self.train_transform(x)