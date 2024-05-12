import argparse
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from unet import UNet
from PIL import Image

# Global variables
dir_checkpoint = Path('./checkpoints/')
valid_percent: float = 0.1

# Step 0. Create dataset class structure
# data location directory
data_dir='./Train/'
class ISICDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    # For predict.py
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    @property
    #define a classes property that categorizes the dataset (we have 2 classes in this case, images and ground_truth)
    def classes(self):
        return self.data.classes
    
transform = transforms.Compose([
    transforms.Resize((300, 100)), # set input image sizes to be all the same
    transforms.ToTensor(),
])

dataset = ISICDataset(data_dir, transform)
print(len(dataset))

# Step 1. Split the training data into Train and Validation randomly each time. (There is no Testing)
# unit is number of images, total MUST add up to the total number of images in the dataset
# num_train need to be high enough compared to total number of images in order to make training work 
num_train = 18
num_valid = len(dataset) - num_train

train_set, valid_set = random_split(dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(0))
#print(len(train_set))

# Step 2. Create data loaders for training and validation data
def create_dataloader(batch_size):
    t_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    v_dataloader = DataLoader(valid_set, shuffle=False, batch_size=batch_size, drop_last=True)
    return t_dataloader, v_dataloader

# Step 3. Make a training model to process the data (I chose UNet)
def train_model(
        epochs,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
    ):
    
    model = UNet(n_channels=3, n_classes=2)

    # Initialize the optimizer and loss criterion
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    criterion = nn.CrossEntropyLoss()

    # Step 4. Train the model on the given data
    model.to(device)
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            #print(outputs.shape)
            #print(labels.shape)
            # Loss function is returning tensor dimension bug
            #loss = criterion(outputs, labels)
            #loss.backward()
            optimizer.step()
        #   running_loss += loss.item() * labels.size(0)
        # train_loss = running_loss / len(dataloader.dataset)
        # train_losses.append(train_loss)
            
        print(f"Epoch {epoch+1}/{num_epochs} - IoU is:")

        # Checkpoint model save file (.pth) for each epoch training
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet model on images and their corresponding target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=3, help='Number of epochs to run')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # use the line below instead if wish to utilize gpu instead of cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Had to use my laptop's cpu for computation
    #device = torch.device('cpu')

    train_dataloader, valid_dataloader = create_dataloader(args.batch_size)

    train_model(epochs=args.epochs)
