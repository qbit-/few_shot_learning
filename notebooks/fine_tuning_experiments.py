# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from PIL import Image
import os
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import logging as log
from tqdm.notebook import tqdm

DATASET_PATH = 'data'
RANDOM_SEED=42


# ### Data feeding

class ImageDFData(Dataset):
    """Loads images based on the path contained in the pd.DataFrame.
        Creates unique integer labels from class names sorted alphabetically
        Parameters:
            df: pd.DataFrame - dataframe with paths to data and class names
            transform: callable, optional - A function/transform that takes in an PIL image
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """ 
            Args:
                index: int
            Returns:
                (sample, label) where target is class_index of the target class.
            Return type:
                tuple
        """
        image = Image.open(self.df.image[index]).convert("RGB")
        label = self.df.label[index]
        if self.transform:
            image = self.transform(image)

        return image, label


# +
### Load dataset from index
# -

df = pd.read_pickle(os.path.join(DATASET_PATH, 'train_top20.p'))
label_types = pd.read_pickle(os.path.join(DATASET_PATH, 'label_types.p'))

train_df = df.sample(frac=0.9, random_state=RANDOM_SEED)
val_df = df.drop(train_df.index)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)



# +
### Define the loading pipeline
# -

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # these values are for Imagenet weights
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # these values are for Imagenet weights
    ]),
}

image_datasets = {
    'train': ImageDFData(train_df, data_transforms['train']),
    'val': ImageDFData(val_df, data_transforms['val'])
}



dataloaders = {x: DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=1) 
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# +
### Test dataloader
# -

def imshow(inp, title=None):
    """Plots normalized image Tensors"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# +
# Get a batch of training data
inputs, labels = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[label_types[x.numpy()] for x in labels])

# +
# Get a batch of validaion data
inputs, labels = next(iter(dataloaders['val']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[label_types[x.numpy()] for x in labels])


# -

# ### Training setup

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """Trains the model and saves best weights"""
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(label_types[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



# +
### Define the model
# -

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# +
# Replace last layer
model_ft.fc = nn.Linear(num_ftrs, len(label_types))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Set the optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# -

# ### Training 

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)



