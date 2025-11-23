import torch

import torch.nn as nn
import torch.nn.functional as F

from model.common import DiscriminativeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LaneNet(nn.Module):
    """
    Takes as input the image in a transformed 512*512*3 tensor 
    
    Returns
        Binary mask: 512*512*1 tensor with 1 if lane and 0 otherwise 
        Binary Mask logits: 512*512*1 tensor before applying the sigmoid
        Instance segmentation: 512*512*C tensor with C being the number of clusters
                                Logits for each cluster 
    """
    def __init__(self):
        super(LaneNet, self).__init__()
        self._encoder = None
        self._binary_seg_decoder = None
        self._instance_seg_decoder = None

    def forward(self, x):
        encoded = self._encoder(x)
        binary_mask, binary_mask_logits = self._binary_seg_decoder(encoded)
        instance_seg_logits = self._instance_seg_decoder(encoded)

        return [binary_mask, binary_mask_logits, instance_seg_logits]



def compute_loss(binary_mask_pt, instance_seg_logits_pt, binary_mask_gt, instance_seg_gt):
    """
    Consists of 2 parts:
        1. Inverse Frequency Weighted Binary Cross Entropy
        2. Clustering Loss with Variance and Distance terms (Discriminative Loss)
    """
    delta_v = 0.5
    delta_d = 3.0

    # Weighter BCE loss
    pos_fraction = binary_mask_gt.mean()
    neg_fraction = 1- pos_fraction
    pos_weight = 1 / (pos_fraction + 1e-6)
    neg_weight = 1 / (neg_fraction + 1e-6)
    criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = criterion_bce(binary_mask_pt, binary_mask_gt)
    weights = binary_mask_gt * pos_weight + (1-binary_mask_gt) * neg_weight
    weighted_bce_loss = (bce_loss * weights).mean()    
    
    # Discriminative Loss
    criterion_clustering = DiscriminativeLoss(delta_v, delta_d)
    var_loss, dist_loss = criterion_clustering(instance_seg_logits_pt, instance_seg_gt)

    total_loss = weighted_bce_loss + var_loss + dist_loss

    return weighted_bce_loss


if __name__=="__main__":
    # pass
    tensor = torch.ones((2, 3), device=device)
    print(tensor.mean())