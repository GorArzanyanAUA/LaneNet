import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiscriminativeLoss(nn.Module):
    """
    Implementation of the discriminative loss with to terms.
    """
    def __init__(self, delta_v, delta_d):
        self.delta_v = delta_v
        self.delta_d = delta_d
    

    def forward(self, embedding_tensor, instance_labels):
        H, W, D = embedding_tensor.shape


class LaneNetDataset(Dataset):
    pass


class LaneNetDataLoader(DataLoader):
    pass

