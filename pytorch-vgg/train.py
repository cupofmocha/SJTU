import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import argparse

from tempfile import TemporaryDirectory
from vgg import * # own VGG implementation folder
from pathlib import Path

# Global variables
data_dir = './data/hymenoptera_data'
dir_checkpoint = Path('./checkpoints/')
cudnn.benchmark = True # allows cuDNN to benchmark multiple convolution algorithms and select the fastest one
batch_size = 4
### Define how many object classification categories here, and make sure the data amount of each object type is relatively BALANCED and varied! ###
num_classes = 3 # bees, ants, human

def get_args():
    parser = argparse.ArgumentParser(description='Predict object classifications after performing training')
    parser.add_argument('--use-pretrained', '-u', action="store_true", help='Use Pretrained network')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=3, help='Number of epochs to run')
    return parser.parse_args()


# Data augmentation and normalization for training
# Only normalization for validation
# Using dictionary structure for the data
### Crop images to be 224 x 224, which is important to match the required input sizes of the VGG model! ###
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Obtain the images from data_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# Split the training data into Train and Validation
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu') # use this for cpu


# Make a training model that processes the data -- VGG
def train_model(num_epochs, is_fixed_feature_extractor):
    start_time = time.time() # record the starting time


    #---------------------------------Define Model---------------------------------###
    # This is for if using a pretrained VGG16 network (Transfer Learning)
    if args.use_pretrained:
        model = models.vgg16(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[6].in_features
        # The size of each output sample is set to 2.
        # classifier[6] is the Linear component of vgg
        model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    else:
    # Otherwise use my own VGG implementation (accuracy is lower..)
        model = VGG_net(num_classes, in_channels=3)
    #------------------------------------------------------------------------------###
    
    
    # If using Fixed Feature Extractor method, disable all the gradients except for final layer (model.fc)
    if (is_fixed_feature_extractor==True):
        for param in model.parameters():
            param.requires_grad = False
        # Optimizer that specifies per-layer learning rates suggested from Pytorch documentation
        ### For Fixed Feature Extractor, only the parameters of final layer (model.fc) are being optimized
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)        
    else:
        # Otherwise if only using the pretrained network, parameters of each and every layer are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Decay Learning Rate (LR) by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
    
    model = model.to(device)
    
    # Loss function that measures the difference between the predicted class probabilities and the true class labels
    criterion = nn.CrossEntropyLoss()

    # Create a temp directory to save training checkpoints
    with TemporaryDirectory() as temp_dir:
        # initialize first the directory to save the best training model so far
        best_model_params_path = os.path.join(temp_dir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0 # measure of the best accuracy achieved by the model

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 20)

            # Each epoch consists of a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() # train the model via pytorch module (nn)
                else:
                    model.eval()

                running_loss = 0.0
                running_correct_counter = 0 # counter that stores the number of correct predictions by the model in given epoch

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients which is very important to ensure the gradient does not accumulate and affect the next epoch
                    optimizer.zero_grad()

                    # if training phase, forward track history as outputs
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        ### Backward + Optimize step ###
                        if phase == 'train':
                            # loss.backward() very important!! It is used to compute mean-squared error between the input and the target (a.k.a gradients)
                            # this enables the deep learning's graph to learn and be differentiated using chain rule
                            loss.backward()
                            optimizer.step()

                    # Compute statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_correct_counter += torch.sum(predictions == labels.data)
                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_correct_counter.double() / dataset_sizes[phase] # epoch accuracy in decimals

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model that has the best accuracy thus far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            print(f'Checkpoint {epoch} saved!')
            print()

        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Load the final best model weights and return
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == '__main__':
    args = get_args()
    
    # Execute training
    model_ft = train_model(num_epochs=args.epochs, is_fixed_feature_extractor=False)