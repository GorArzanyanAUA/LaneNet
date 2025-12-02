import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import time

class TuSimpleLaneNetDataset(Dataset):
    """
    TuSimple lane dataset using a list file of the form:

        <img_rel_path> <seg_rel_path> e1 e2 e3 e4 e5 e6

    __getitem__ returns:
      - image: CHW float32 tensor
      - binary_mask: HW int64 tensor {0,1}
      - instance_mask: HW int64 tensor {0,1,2,...}

    Note: __getitem__ processes on the go.
    """

    def __init__(self, root, list_file, transform=None):
        """
        root:       dataset root (contains 'clips/' and 'seg_label/')
        list_file:  text file like train_gt.txt with lines:
                        img_path seg_label_path e1 e2 ...
        transform: callable(img, bin_mask, inst_mask) -> (img, bin_mask, inst_mask)
                         for resize / crop / flip etc.
        """
        self.root = root
        self.list_file = list_file
        self.transform = transform

        self.samples = self._load_list() # list of dicts

    def _load_list(self):
        samples = []
        with open(self.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                img_rel = parts[0]
                seg_rel = parts[1]

                samples.append(
                    {
                        "img_rel": img_rel,
                        "seg_rel": seg_rel,
                    }
                )
        if not samples:
            raise RuntimeError(f"No valid lines found in list file: {self.list_file}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = os.path.join(self.root, rec["img_rel"].lstrip("/"))
        seg_path = os.path.join(self.root, rec["seg_rel"].lstrip("/"))

        img = Image.open(img_path).convert("RGB")
        seg_pil = Image.open(seg_path)
        inst_np = np.array(seg_pil)

        # if paletted/RGB, drop extra channels
        if inst_np.ndim == 3:
            inst_np = inst_np[..., 0]
        inst_np = inst_np.astype(np.int64)  # H×W

        bin_np = (inst_np > 0).astype(np.uint8)  # H×W, 0/1

        inst_pil = Image.fromarray(inst_np.astype(np.uint8), mode="L")
        bin_pil = Image.fromarray(bin_np, mode="L")
        img = TF.to_tensor(img)
        
        if self.transform is not None:
            img, bin_pil, inst_pil = self.transform(img, bin_pil, inst_pil)

        bin_np = np.array(bin_pil, dtype=np.int64)
        inst_np = np.array(inst_pil, dtype=np.int64)

        binary_mask = torch.from_numpy(bin_np).long()      # (H, W)
        instance_mask = torch.from_numpy(inst_np).long()   # (H, W)

        sample = {
            "image": img,                     # (C, H, W) float32
            "binary_mask": binary_mask,       # (H, W) int64 {0,1}
            "instance_mask": instance_mask,   # (H, W) int64 {0..K}
        }
        return sample

class ReScaleTransform:
    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, img, bin_mask, inst_mask):
        img = TF.resize(img, self.size)

        # Resize masks – ALWAYS use NEAREST or labels get corrupted!
        binary_mask = TF.resize(bin_mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        instance_mask = TF.resize(inst_mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return img, binary_mask, instance_mask

class Encoder(nn.Module):
    """
    Implements the Initial, first and second stages of the 
    ENet Architechture.
    Initiial:
        Single Bottleneck Block:
            Takes input 
                3x3 stride 2 conv with 13 maps
                Maxpooling 2x2 stride 2
    first Stage:
    Second Stage:
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_block = InitialBlcok(3, 16)
        # 1st Stage
        self.bottleneck1_0 = DownsamplingBottlneck(16, 64)
        self.bottleneck1_1 = RegularBottlneck(64, 64)
        self.bottleneck1_2 = RegularBottlneck(64, 64)
        self.bottleneck1_3 = RegularBottlneck(64, 64)
        self.bottleneck1_4 = RegularBottlneck(64, 64)
        
        # # 2nd Stage
        self.bottleneck2_0 = DownsamplingBottlneck(64, 128, dropout_prob=0.1)
        self.bottleneck2_1 = RegularBottlneck(128, 128, dropout_prob=0.1)
        self.dilated2_2 = RegularBottlneck(128, 128, dropout_prob=0.1, padding=2, dilation=2)
        self.asymmetrc2_3 = RegularBottlneck(128, 128, kernel_size=5, dropout_prob=0.1, padding=2, asymmetric=True)
        self.dilated2_4 = RegularBottlneck(128, 128, dropout_prob=0.2, padding=4, dilation=4) 
        self.bottleneck2_5 = RegularBottlneck(128, 128, dropout_prob=0.1)
        self.dilated2_6 = RegularBottlneck(128, 128, dropout_prob=0.1, dilation=8, padding=8) 
        self.asymmetrc2_7 = RegularBottlneck(128, 128, kernel_size=5, dropout_prob=0.1, padding=2, asymmetric=True) 
        self.dilated2_8 = RegularBottlneck(128, 128, dropout_prob=0.1, dilation=16, padding=16) 
        
    def forward(self, x):
        x = self.initial_block(x)
        # first stage
        x, max_indices_1_0 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # second stage
        x, max_indeces_2_0 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetrc2_3(x)
        x = self.dilated2_4(x)
        x = self.bottleneck2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetrc2_7(x)
        x = self.dilated2_8(x)
        return x, [max_indeces_2_0, max_indices_1_0]

class SegmentationDecoder(nn.Module):
    """
    Implments the Binary segmentation Decoder branch.

    Consisits of the stage 3, 4, 5 of the ENet.
    """
    def __init__(self, num_classes):
        super(SegmentationDecoder, self).__init__()
        # Stage 3
        self.bottleneck3_1 = RegularBottlneck(128, 128, dropout_prob=0.1)
        self.dilated3_2 = RegularBottlneck(128, 128, dropout_prob=0.1, padding=2, dilation=2)
        self.asymmetrc3_3 = RegularBottlneck(128, 128, kernel_size=5, dropout_prob=0.1, padding=2, asymmetric=True)
        self.dilated3_4 = RegularBottlneck(128, 128, dropout_prob=0.2, padding=4, dilation=4) 
        self.bottleneck3_5 = RegularBottlneck(128, 128, dropout_prob=0.1)
        self.dilated3_6 = RegularBottlneck(128, 128, dropout_prob=0.1, dilation=8, padding=8) 
        self.asymmetrc3_7 = RegularBottlneck(128, 128, kernel_size=5, dropout_prob=0.1, padding=2, asymmetric=True) 
        self.dilated3_8 = RegularBottlneck(128, 128, dropout_prob=0.1, dilation=16, padding=16) 

        # Stage 4
        self.upbottleneck4_0 = UpsamplingBottlneck(128, 64, kernel_size=3, internal_ratio=2)
        self.bottleneck4_1 = RegularBottlneck(64, 64, dropout_prob=0.1)
        self.bottleneck4_2 = RegularBottlneck(64, 64, dropout_prob=0.1)

        # Stage 5
        self.upbottleneck5_0 = UpsamplingBottlneck(64, 16, kernel_size=3, internal_ratio=1)
        self.bottleneck5_1 = RegularBottlneck(16, 16, dropout_prob=0.1)

        # Head
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, max_indices):
        ''' torch.Size([BATCH_SIZE, 128, 64, 64]) '''
        x = self.bottleneck3_1(x)
        x = self.dilated3_2(x)
        x = self.asymmetrc3_3(x)
        x = self.dilated3_4(x)
        x = self.bottleneck3_5(x)
        x = self.dilated3_6(x)
        x = self.asymmetrc3_7(x)
        x = self.dilated3_8(x)

        # stage 4
        x = self.upbottleneck4_0(x, max_indices[0])
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # stage 5
        x = self.upbottleneck5_0(x, max_indices[1])
        x = self.bottleneck5_1(x)
        logits = self.transposed_conv(x)        
        return logits

class InstanceSegDecoder(nn.Module):
    pass

class InitialBlcok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlcok, self).__init__()
        self.activation = nn.PReLU()
        self.main_branch = nn.Conv2d(in_channels, out_channels=(out_channels-3), kernel_size=3, stride=2, padding=1)
        self.ext_branch = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat([main, ext], dim=1)
        out = self.batch_norm(out)    
        return self.activation(out)

class DownsamplingBottlneck(nn.Module):
    '''
    Implements ResNet but with downsampling. 
    In main branch downsampling is done with Maxpooling layer.
    In ext branch downsampling is done with 2*2 stride 2 convolution.

    Main Branch:
        Maxpolling with stride 2, indeces are saved

    Ext Branch:
        2*2 convolution, stride 2, also projection to reduce channels by internal ratio
        convolution 3*3
        1*1 convolution to increase channels to out_channels
        droupout

    Note: the outout is not the concatination but the sum
    '''
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.2):
        super(DownsamplingBottlneck, self).__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.PReLU 
        self.activation = activation()

        self.main_max1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2),
                                        nn.BatchNorm2d(internal_channels), 
                                        activation())
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1), 
                                       nn.BatchNorm2d(internal_channels), 
                                       activation())
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(out_channels),
                                       activation())
        self.ext_regulator = nn.Dropout2d(p=dropout_prob)



    def forward(self, x):
        main, indices = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regulator(ext)

        n, ch, h, w = ext.shape
        padding = torch.zeros((n, ch-main.shape[1], h, w), device=main.device)
        main = torch.concat((main, padding), dim=1)
        out = main + ext
        return self.activation(out), indices

class UpsamplingBottlneck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, internal_ratio=2, stride=1, padding=0, dropout_rate=0.1):
        super(UpsamplingBottlneck, self).__init__()
        internal_channels = in_channels//internal_ratio
        activation = nn.PReLU
        self.activation = activation()
        # Main
        # projection to higher channels
        self.main_proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       activation())
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Extension
        self.ext_1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(internal_channels),
                                   activation())
        self.ext_2 = nn.Sequential(nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=False),
                                   nn.BatchNorm2d(internal_channels),
                                   activation())
        self.ext_3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        self.ext_dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, max_indices):
        main = self.main_proj(x)
        main = self.main_unpool(main, max_indices)

        ext = self.ext_1(x)
        ext = self.ext_2(ext)
        ext = self.ext_3(ext)
        ext = self.ext_dropout(ext)
        
        out = main + ext
        return self.activation(out)
    
class RegularBottlneck(nn.Module):
    '''
    Implements ResNet like functionality. 
    Main Branch
        shortcut connection.
    Ext Branch
        1*1 convolution stride 1: channel projection, dim reduction
        3*3 convolution: 
        1*1 convolution stride 1: channel expansion to out_dimension
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, internal_ratio=4, dropout_prob=0.01, dilation=1, padding=1, asymmetric=False):
        super(RegularBottlneck, self).__init__()
        activation = nn.PReLU
        self.activation = activation()

        internal_channels = in_channels // internal_ratio

        self.ext_1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=1, bias=False),
                                 nn.BatchNorm2d(internal_channels), activation())
        if asymmetric:
            self.ext_2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=False, dilation=dilation),
                                       nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, bias=False),
                                       nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation),
                                       nn.BatchNorm2d(internal_channels), activation())
        
        self.ext_3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        main = x
        ext = self.ext_1(x)
        ext = self.ext_2(ext)
        ext = self.ext_3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.activation(out)

class DiscriminativeLoss(nn.Module):
    """
    Implementation of the discriminative loss with two terms.
    Using L2 norm
    """
    def __init__(self, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.delta_v = delta_v   # Variance term is activated only when is bigger than this
        self.delta_d = delta_d   # Distance term is activated only when is smaller than this

    def forward(self, embedding_tensor, instance_labels):
        B, C, H, W = embedding_tensor.shape
        var_loss = torch.tensor(0, dtype=embedding_tensor.dtype, device=embedding_tensor.device)
        dist_loss = torch.tensor(0, dtype=embedding_tensor.dtype, device=embedding_tensor.device)
        reg_loss = torch.tensor(0, dtype=embedding_tensor.dtype, device=embedding_tensor.device)
        for b in range(B):
            embedding = embedding_tensor[b]
            label_gt = instance_labels[b]
            labels = torch.unique(label_gt) # list tensor of unique elements
            labels = labels[labels!=0]   # same without element with value 0
            num_lanes = len(labels)
            if num_lanes == 0:
                # TODO
                continue

            # compute gt centroid means
            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (label_gt == lane_idx).to(device="cuda")  # binary mask for that lane 
                if not seg_mask_i.any():
                    continue
                masked_values = embedding[:, seg_mask_i]
                mean_i = torch.mean(masked_values, dim=1, keepdim=True)
                centroid_mean.append(mean_i)
                dists = torch.norm(masked_values - mean_i, dim=0)
                var_loss += torch.mean(F.relu(dists -self.delta_v) ** 2)

            var_loss = var_loss / num_lanes
            centroid_mean = torch.stack(centroid_mean).squeeze(2)
            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, C)
                centroid_mean2 = centroid_mean.reshape(1, -1, C)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d
                dist_loss += torch.sum(F.relu(self.delta_d - dist) ** 2)
            
            dist_loss = dist_loss / (2 * num_lanes * (num_lanes - 1))

        var_loss = var_loss / B
        dist_loss = dist_loss / B
        return var_loss, dist_loss

if __name__ == "__main__":
    dataset = TuSimpleLaneNetDataset("/home/student/Dev/LaneNet/data/TUSimple/train_set", "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_val_gt.txt")
