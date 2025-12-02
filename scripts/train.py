import torch

from LaneNet.model.model import LaneNet, compute_loss
from LaneNet.model.common import TuSimpleLaneNetDataset, ReScaleTransform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_step(model, batch, optimizer, device="cuda"):
    model.train()
    
    images = batch["image"].to(device, non_blocking=True)
    binary_mask_gt = batch["binary_mask"].to(device, non_blocking=True).float()
    isntance_seg_gt = batch["instance_mask"].to(device, non_blocking=True)

    out = model(images)
    weighted_bce_loss, var_loss, dist_loss = compute_loss(binary_mask_logits=out["binary_mask_logits"], 
                                                          binary_mask_gt=binary_mask_gt, 
                                                          instance_seg_logits_pt=out["instance_seg_logits"],
                                                          instance_seg_gt=isntance_seg_gt)
    loss = weighted_bce_loss + var_loss + dist_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return weighted_bce_loss.detach(), var_loss.detach(), dist_loss.detach()

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
    BATCH_SIZE = 4
    LR = 0.001
    EPOCHS = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = ".checkpoints/model.pth"
    base_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set"
    train_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_gt.txt"
    val_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/val_gt.txt"

    train_dataset = TuSimpleLaneNetDataset(base_path, train_dataset_path, transform=ReScaleTransform((512, 512)))    
    # print(len(train_dataset))
    # return
    val_dataset = TuSimpleLaneNetDataset(base_path, val_dataset_path, transform=ReScaleTransform((512, 512)))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
    model = LaneNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ---- Train ----
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            bce_loss, var_loss, dist_loss = train_step(model, batch, optimizer, device)
            loss = bce_loss + var_loss + dist_loss
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
            
        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(device, non_blocking=True)
                binary_mask_gt = batch["binary_mask"].to(device, non_blocking=True).float()
                instance_seg_gt = batch["instance_mask"].to(device, non_blocking=True)

                out = model(images)
                weighted_bce_loss, var_loss, dist_loss = compute_loss(
                    binary_mask_logits=out["binary_mask_logits"],
                    binary_mask_gt=binary_mask_gt,
                    instance_seg_logits_pt=out["instance_seg_logits"],
                    instance_seg_gt=instance_seg_gt,
                )
                loss = weighted_bce_loss + var_loss + dist_loss
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ".checkpoints/best_model.pth")
            print("  -> New best model saved")

if __name__=="__main__":
    main()