import torch
from model.model import LaneNet, compute_loss
from model.common import LaneNetDataset, LaneNetDataLoader


def train(model, batch, optimizer):
    raise NotImplementedError("TO DO")


def main():
    # "argument parser can be added later"
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = ".checkpoints/model.pth"
    train_dataset_path = None
    val_dataset_path = None

    train_dataset = LaneNetDataset(train_dataset_path, transform=None)
    train_dataloader = LaneNetDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = LaneNetDataset(val_dataset_path, transform=None)
    val_dataloader = LaneNetDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LaneNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            train(model, batch, optimizer)
            


if __name__=="__main__":
    main()