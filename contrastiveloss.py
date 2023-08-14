import torch
import torch.nn.functional as F
from torch import nn



def device_as(t1, t2):
   """
   Moves tensor t1 to the device of tensor t2.
   """
   return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss calculates the loss in the SimCLR method, emphasizing learning by comparing pair similarities.
    The loss minimizes differences for similar pairs and amplifies differences for dissimilar pairs.

    Args:
        batch_size (int): Batch size of input data pairs.
        temperature (float, optional): Scaling factor for the logits. Defaults to 0.5.

    Attributes:
        batch_size (int): Batch size of input data pairs.
        temperature (float): Scaling factor for the logits.
    """
    def __init__(
            self,
            batch_size: int,
            temperature: float=0.5
    ):
        super(ContrastiveLoss, self).__init__()
        # Initializing loss hyperparameters
        self.batch_size = batch_size
        self.temperature = temperature

    def _get_similarity_matrix(self, u, v):
        """
        Calculates similarity matrix for input data pairs using cosine similarity.
        """
        representations = torch.cat([u, v], dim=0)
        similiarity_matrix = F.cosine_similarity(
            x1=representations.unsqueeze(0),
            x2=representations.unsqueeze(1),
            dim=2
        )

        return similiarity_matrix
    
    def forward(self, z_i, z_j):
        """
        Compute contrastive loss based on input representations and labels such as positive or negative pair.

        Args:
            z_i (torch.Tensor): Embeddings of the first view.
            z_j (torch.Tensor): Embeddings of the second view.

        Returns:
            torch.Tensor: Calculated contrastive loss.
        """
        batch_size = z_i.shape[0]
        # performing l_2 normalization 
        z_i = F.normalize(z_i, p=2)
        z_j = F.normalize(z_j, p=2)

        similarity_matrix = self._get_similarity_matrix(z_i, z_j)

        # extracting only positive pairs to calculate the numerator
        positives = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size)
            ],
        )

        nominator = torch.exp(positives / self.temperature)

        # getting all (positive and negative) pairs to calculate the denominator
        mask = (~torch.eye(2 * batch_size, dtype=bool)).float()
        denominator = torch.sum(
            device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature),
            dim=1
        )

        # calculating l_i,j's and contrastive loss
        all_losses = -torch.log(nominator / denominator)
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss