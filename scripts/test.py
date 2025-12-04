import torch
from LaneNet.model.model import LaneNet, compute_loss
from LaneNet.model.common import TuSimpleLaneNetDataset, ReScaleTransform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_model(model, checkpoint_path, device):
    """
    Load a pre-trained model from a checkpoint.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

def main(samples=10, checkpoint_path=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set"
    train_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_gt.txt"
    val_dataset_path = "/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/val_gt.txt"

    # Dataset and Dataloader setup
    train_dataset = TuSimpleLaneNetDataset(base_path, train_dataset_path, transform=ReScaleTransform((512, 512)))
    val_dataset = TuSimpleLaneNetDataset(base_path, val_dataset_path, transform=ReScaleTransform((512, 512)))
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    # Initialize model and load weights if checkpoint is provided
    model = LaneNet().to(device)
    if checkpoint_path:
        load_model(model, checkpoint_path, device)

    best_val_loss = float("inf")

    for sample in range(samples):
        # ---- Validation Loop ----
        running_loss = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(device, non_blocking=True)
                binary_mask_gt = batch["binary_mask"].to(device, non_blocking=True).float()
                instance_seg_gt = batch["instance_mask"].to(device, non_blocking=True)

                # Get model output
                out = model(images)
                weighted_bce_loss, var_loss, dist_loss = compute_loss(
                    binary_mask_logits=out["binary_mask_logits"],
                    binary_mask_gt=binary_mask_gt,
                    instance_seg_logits_pt=out["instance_seg_logits"],
                    instance_seg_gt=instance_seg_gt,
                )
                loss = weighted_bce_loss + var_loss + dist_loss
                val_loss += loss.item()

                # Plot the results
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                print(out["binary_mask_logits"].shape)

                # Ground truth binary mask
                axes[0].imshow(batch['binary_mask'][0].detach().cpu(), cmap='gray')
                axes[0].set_title("Ground Truth Mask")
                axes[0].axis('off')

                # Input image
                axes[1].imshow(batch['image'][0].permute(1, 2, 0).detach().cpu())
                axes[1].set_title("Input Image")
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()

        print(f"Validation Loss after sample {sample+1}: {val_loss / len(val_dataloader)}")

if __name__ == "__main__":
    checkpoint_path = "/home/student/Dev/LaneNet/best_model.pth"  # Update this to the correct path
    main(samples=5, checkpoint_path=checkpoint_path)
