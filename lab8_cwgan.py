#!/usr/bin/env python
# coding: utf-8

#
# # CS 763 - Lab 8 (WGAN-GPs) Part-B

from torch.autograd import grad
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


# We will use the Fashion-MNIST dataset for this task
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

trainset = torchvision.datasets.FashionMNIST(
    './data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(
    './data', download=True, train=False, transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2)


# Visualising the dataset
labeldic = {}
labeldic[0] = 'T-shirt/top'
labeldic[1] = 'Trouser'
labeldic[2] = 'Pullover'
labeldic[3] = 'Dress'
labeldic[4] = 'Coat'
labeldic[5] = 'Sandal'
labeldic[6] = 'Shirt'
labeldic[7] = 'Sneaker'
labeldic[8] = 'Bag'
labeldic[9] = 'Ankle boot'
print('Lenth of Trainset = {}'.format(len(trainset)))
print('Lenth of Testset = {}'.format(len(testset)))
print('Dimensions of each image = ({},{})'.format(
    trainset[0][0].shape[1], trainset[0][0].shape[2]))
fig, axs = plt.subplots(4, 10, sharey=True, figsize=(20, 10))
for i in range(40):
    axs[i//10, i % 10].imshow(trainset[i][0].squeeze(), cmap='gray')
    axs[i//10, i % 10].set_title('{}'.format(labeldic[trainset[i][1]]))
plt.suptitle('Fashion MNIST Train Images')
plt.show()


# Generator and Discriminator Architecures
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ## TODO ##

        ## TODO ##

    # forward pass
    def forward(self, x, labels):
        pass
        ## TODO ##

        ## TODO ##


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ## TODO ##

        ## TODO ##

    # forward pass
    def forward(self, x, labels):
        pass
        ## TODO ##

        ## TODO ##


        # Create the models
G = Generator().to(device)
D = Discriminator().to(device)

# Define the Optimizer
## TODO ##
optimizer_g = None
optimizer_d = None
## TODO ##

# Define the Washerstein Loss functions
lamda = 10


def get_WLoss_generator(fake_images, labels):
    pass
    ## TODO ##

    ## TODO ##


def get_WLoss_discriminator(fake_images, true_images, labels):
    pass
    ## TODO ##

    ## TODO ##


def get_gradient_regularisation(fake_images, true_images, labels):
    pass
    ## TODO ##

    ## TODO ##


def get_grad(fake_images, true_images, labels):
    epsilon = torch.FloatTensor(np.random.uniform(
        size=(fake_images.shape[0], 1, 1, 1))).to(device)
    xbar = epsilon*true_images + (1-epsilon)*fake_images
    xbar.requires_grad_()
    lipschitz_grad = grad(
        outputs=D(xbar, labels).sum(),
        inputs=xbar,
        create_graph=True,
        retain_graph=True)[0]
    return lipschitz_grad.view(xbar.shape[0], -1)

# Train Function
# returns the total loss for the epoch


def train(epoch):
    loss_acc_g = 0
    loss_acc_d = 0

    for i, (images, labels) in tqdm(enumerate(trainloader), desc="Training for epoch {}".format(epoch), total=len(trainloader)):
        # Discriminator Training
        optimizer_d.zero_grad()
        ## TODO ##

        ## TODO ##
        optimizer_d.step()

        if i % 5 == 0:
            # Generator training
            optimizer_g.zero_grad()
            ## TODO ##

            ## TODO ##
            optimizer_g.step()
            loss_acc_g += loss_g.item()
        loss_acc_d += loss_d.item()

    return (loss_acc_g/(i/5)+loss_acc_d/i), loss_acc_g/(i/5), loss_acc_d/i


# Functions to save and load checkpoints
# Enter the path on your shared drive where the checkpoint is to be saved
## TODO ##
# For example path_to_checkpoint = '/content/drive/Shareddrives/CS763Lab8/'
## TODO ##


def save_checkpoint(e):
    if not os.path.isdir(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)
    torch.save({'e': e, 'gen': G.state_dict(), 'disc': D.state_dict(), 'optim_d': optimizer_d.state_dict(
    ), 'optim_g': optimizer_g.state_dict()}, os.path.join(path_to_checkpoint, 'checkpoint_cwgan.pth'))


def load_checkpoint():
    if not os.path.isfile(os.path.join(path_to_checkpoint, 'checkpoint_cwgan.pth')):
        return -1
    dic = torch.load(os.path.join(path_to_checkpoint, 'checkpoint_cwgan.pth'))
    G.load_state_dict(dic['gen'])
    D.load_state_dict(dic['disc'])
    optimizer_d.load_state_dict(dic['optim_d'])
    optimizer_g.load_state_dict(dic['optim_g'])
    return dic['e']

# The generate functions are defined here
# Should return a nx28x28 numpy array


def generate(n, ci):
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
    li = torch.randint(0, 10, (1,))[0].item()
    g_output = generate(1, li)[0]
    G.train()
    plt.imshow(g_output.reshape(28, 28), cmap='gray')
    plt.title('Category: {}'.format(labeldic[li]))
    plt.savefig('./images/{}.png'.format(e))
    print('loss_g = {}, loss_d = {}\n'.format(loss_g, loss_d))
    save_checkpoint(e)

# Test script
load_checkpoint()
with torch.no_grad():
    G.eval()
    fig, axs = plt.subplots(4, 10, sharey=True, figsize=(20, 10))
    for i in range(40):
        fake_image = generate(1, i % 10)[0]
        axs[i//10, i % 10].imshow(fake_image.reshape(28, 28), cmap='gray')
        axs[i//10, i % 10].set_title('{}'.format(labeldic[i % 10]))
    plt.suptitle('Fashion MNIST Generated Images')
    plt.show()
