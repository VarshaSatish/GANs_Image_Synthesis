#!/usr/bin/env python
# coding: utf-8

# CS 763 - Lab 8 (GANs) Part-A

from tqdm import tqdm
import time
import sys
import os
import PIL
from torch.nn import functional as F
from torch import nn, optim
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import torchvision
import random
import torch
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# Check for GPU specs
get_ipython().system('nvidia-smi')

# device set to cuda if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to fix the seed for randomisation

def fix_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

fix_seed(763)

# We will use the MNIST dataset for Part A of the lab.

# define transformations to be applied on the input images
# Input images from the dataset are PIL objects, hence we first convert them to Tensors, then we normalise them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

trainset = torchvision.datasets.MNIST(
    './data', download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST(
    './data', download=True, train=False, transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2)

# Visualising the dataset
print('Lenth of Trainset = {}'.format(len(trainset)))
print('Lenth of Testset = {}'.format(len(testset)))
print('Dimensions of each image = ({},{})'.format(
    trainset[0][0].shape[1], trainset[0][0].shape[2]))
fig, axs = plt.subplots(1, 6, sharey=True, figsize=(20, 4))
for i in range(6):
    axs[i].imshow(trainset[i][0].squeeze(), cmap='gray')
    axs[i].set_title('Class {}'.format(trainset[i][1]))
plt.suptitle('MNIST Train Images')
plt.show()

# Generator and Discriminator Architecures

# The generator takes a noise vector of size (B,100), where B is the batch size and produces a 28x28 image
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ## TODO ##

        ## TODO ##

    # forward pass
    def forward(self, x):
        ## TODO ##
        pass
        ## TODO ##

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ## TODO ##

        ## TODO ##

    # forward pass
    def forward(self, x):
        pass
        ## TODO ##

        ## TODO ##

        # Create the models
G = Generator().to(device)
D = Discriminator().to(device)

# Create a BCELoss object
## TODO ##
criterion = None
## TODO ##

# Define two separate optimizers for generator and discriminator
## TODO ##
optimizer_g = None
optimizer_d = None
## TODO ##

# Train Function
# returns the total loss for the epoch
def train(epoch):
    loss_acc_g = 0
    loss_acc_d = 0
    loss_acc = 0

    for i, (images, _) in tqdm(enumerate(trainloader), desc="Training for epoch {}".format(epoch), total=len(trainloader)):
        optimizer_d.zero_grad()
        # Discriminator training on real images
        ## TODO ##
        prediction_on_real = None
        loss_real = None
        ## TODO ##
        # Discriminator training on fake images
        ## TODO ##
        fake_samples = None
        prediction_on_fake = None  # don't forget to use detach() here
        loss_fake = None
        ## TODO ##
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        ## TODO ##
        fake_samples = None
        prediction_on_fake = None
        loss_g = None
        ## TODO ##
        loss_g.backward()
        optimizer_g.step()

        loss_acc += (loss_g + loss_d).item()
        loss_acc_g += loss_g.item()
        loss_acc_d += loss_d.item()

    return loss_acc/i, loss_acc_g/i, loss_acc_d/i

# Functions to save and load checkpoints

# Enter the path on your shared drive where the checkpoint is to be saved
## TODO ##
# For example path_to_checkpoint = '/content/drive/Shareddrives/CS763Lab8/'
## TODO ##

def save_checkpoint(e):
    if not os.path.isdir(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)
    torch.save({'e': e, 'gen': G.state_dict(), 'disc': D.state_dict(), 'optim_d': optimizer_d.state_dict(
    ), 'optim_g': optimizer_g.state_dict()}, os.path.join(path_to_checkpoint, 'checkpoint_vanilla.pth'))

def load_checkpoint():
    if not os.path.isfile(os.path.join(path_to_checkpoint, 'checkpoint_vanilla.pth')):
        return -1
    dic = torch.load(os.path.join(
        path_to_checkpoint, 'checkpoint_vanilla.pth'))
    G.load_state_dict(dic['gen'])
    D.load_state_dict(dic['disc'])
    optimizer_d.load_state_dict(dic['optim_d'])
    optimizer_g.load_state_dict(dic['optim_g'])
    return dic['e']

# The generate functions are defined here
def generate(n):
    pass
    ## TODO ##

    ## TODO ##

## TODO ##
NUM_EPOCHS = 0
## TODO ##
losses_g = []
losses_d = []

# Resume training
offset = load_checkpoint()
if os.path.exists('images'):
    if offset == -1:
        os.system('rm -rf images')
        os.mkdir('images')
else:
    os.mkdir('images')

for e in range(NUM_EPOCHS):
    if e <= offset:
        continue
    loss, loss_g, loss_d = train(e)
    losses_g.append(loss_g)
    losses_d.append(loss_d)
    G.eval()
    g_output = generate(1)[0]
    G.train()
    plt.imshow(g_output.reshape(28, 28), cmap='gray')
    plt.savefig('./images/{}.png'.format(e))
    print('loss_g = {}, loss_d = {}\n'.format(loss_g, loss_d))
    save_checkpoint(e)

# Test script
load_checkpoint()
with torch.no_grad():
    G.eval()
    fake_images = generate(40)
    fig, axs = plt.subplots(4, 10, sharey=True, figsize=(20, 10))
    for i in range(40):
        axs[i//10, i % 10].imshow(fake_images[i].reshape(28, 28), cmap='gray')
    plt.suptitle('MNIST Generated Images')
    plt.show()