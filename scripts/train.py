import torch

from LaneNet.model.model import LaneNet, compute_loss
from LaneNet.model.common import TuSimpleLaneNetDataset, ReScaleTransform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(model, batch, optimizer):
    device = "cuda"
    out = model(batch["image"].to(device))
    compute_loss(binary_mask_logits=out["binary_mask_logits"], binary_mask_gt=batch["binary_mask"].to(device).float(), 
                 instance_seg_logits_pt=out["instance_seg_logits"] ,instance_seg_gt=batch["instance_mask"])
    
    # fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    # axes[0].imshow(out[0][0][0].detach().cpu(), cmap='gray')
    # axes[0].set_title("Binary Mask")
    # axes[0].axis('off')

    # # === Input channel 0 ===
    # axes[1].imshow(batch['image'][0].permute(1,2,0).detach().cpu())
    # axes[1].set_title("Input ch 0")
    # axes[1].axis('off')

    # # === Input channel 1 ===
    # axes[3].imshow(batch['binary_mask'][0].detach().cpu(), cmap='gray')
    # axes[3].set_title("Input ch 1")
    # axes[3].axis('off')

    # # # === Input channel 2 ===
    # # axes[4].imshow(inputs[idx, 2].detach().cpu(), cmap='gray')
    # # axes[4].set_title("Input ch 2")
    # # axes[4].axis('off')

    # plt.tight_layout()
    # plt.show()
    # # print(model(batch["image"].to(device))[0].shape)

def main():
    # "argument parser can be added later"
    BATCH_SIZE = 5
    LR = 0.001
    EPOCHS = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = ".checkpoints/model.pth"
    base_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set"
    train_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_gt.txt"
    val_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/val_gt.txt"

    train_dataset = TuSimpleLaneNetDataset(base_path, train_dataset_path, transform=ReScaleTransform((512, 512)))    
    val_dataset = TuSimpleLaneNetDataset(base_path, val_dataset_path, transform=ReScaleTransform((512, 512)))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
    model = LaneNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # print(model(train_dataset[0]['image'].to(device))[0].shape)
    
    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            train(model, batch, optimizer)
            return
            


if __name__=="__main__":
    main()