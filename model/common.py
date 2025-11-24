import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class TuSimpleLaneNetDataset(Dataset):
    """
    TuSimple lane dataset using a list file of the form:

        <img_rel_path> <seg_rel_path> e1 e2 e3 e4 e5 e6

    __getitem__ returns:
      - image: CHW float32 tensor
      - binary_mask: HW int64 tensor {0,1}
      - instance_mask: HW int64 tensor {0,1,2,...}
      - exist: L-dim float32 tensor with lane existence flags (optional use)
      - meta: dict with paths and idx
    """

    def __init__(self, root, list_file, transform=None):
        """
        root:       dataset root (contains 'clips/' and 'seg_label/')
        list_file:  text file like train_gt.txt with lines:
                    img_path seg_label_path e1 e2 ...
        joint_transform: callable(img, bin_mask, inst_mask) -> (img, bin_mask, inst_mask)
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
    pass

class BinarySegDecoder(nn.Module):
    pass

class InstanceSegDecoder(nn.Module):
    pass

class DiscriminativeLoss(nn.Module):
    """
    Implementation of the discriminative loss with to terms.
    """
    def __init__(self, delta_v, delta_d):
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding_tensor, instance_labels):
        H, W, D = embedding_tensor.shape

if __name__ == "__main__":
    dataset = TuSimpleLaneNetDataset("/home/student/Dev/LaneNet/data/TUSimple/train_set", "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_val_gt.txt")
