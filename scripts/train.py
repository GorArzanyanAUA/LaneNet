import torch

from LaneNet.model.model import LaneNet, compute_loss
from LaneNet.model.common import TuSimpleLaneNetDataset, ReScaleTransform
from torch.utils.data import DataLoader

def train(model, batch, optimizer):
    print()
    print(len(batch))

def main():
    # "argument parser can be added later"
    BATCH_SIZE = 16
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

    import matplotlib.pyplot as plt


    for batch in train_dataloader:
        print(len(batch))
        fig = plt.figure(figsize=(10, 20))
        fig.add_subplot(1,3,1)
        plt.imshow(batch['image'][0].permute(1,2,0))
        
        fig.add_subplot(1,3,2)
        plt.imshow(batch['binary_mask'][0])
        
        fig.add_subplot(1,3,3)
        plt.imshow(batch['instance_mask'][0])
        plt.show()
        


    # model = LaneNet()
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # for epoch in range(EPOCHS):
    #     for batch in train_dataloader:
    #         train(model, batch, optimizer)
            


if __name__=="__main__":
    main()