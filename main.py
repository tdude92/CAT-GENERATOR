import torch
import torch.nn as nn
from torchvision import datasets, transforms

import os
import math
import cv2
import numpy as np

# Constants
MODEL_ID        = "wgan_0"
DATA_PATH       = "data"

START_EPOCH     = 0
N_EPOCHS        = 50
N_CRITIC        = 5
LEN_Z           = 100
OUT_CHANNELS    = 3
IMAGE_DIM       = 64
BATCH_SIZE      = 64
LEARN_RATE      = 0.00005
CLIP_VALUE      = 0.01
ON_CUDA         = torch.cuda.is_available()


if ON_CUDA:
    print("GPU available. Training on GPU...")
    device = "cuda:0"
else:
    print("GPU not available. Training on CPU...")
    device = "cpu"


# Transformations on training data.
transform = transforms.Compose([
    transforms.Resize(IMAGE_DIM),
    transforms.CenterCrop(IMAGE_DIM),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# Resize images to IMAGE_DIM x IMAGE_DIM
# Apply random horizontal flip.
# Store images data in torch.Tensor
# Normalize RGB values between -1 and 1


# Load data
real_data = datasets.ImageFolder(DATA_PATH, transform = transform)
data_loader = torch.utils.data.DataLoader(real_data, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

<<<<<<< HEAD
=======

# Wasserstein loss function
def wasserstein(prediction, label):
    return torch.mean(prediction * label)


# Weight clipper (Wasserstein critic weights should be between -0.01 and 0.01)
class WeightConstraint(object):
    def __init__(self, constraint):
        self.constraint = constraint
    
    def __call__(self, module):
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(-self.constraint, self.constraint)


>>>>>>> 77bcb897f21aafb07cabbecb5c3f01546d62138a
# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.pipeline(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.pipeline = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0, bias = False)
        )

    def forward(self, x):
        output = self.pipeline(x)
        return output.view(-1)


net_G = Generator()
net_D = Discriminator()

# Load models if specified.
try:
    net_G.load_state_dict(torch.load("models/generator_" + MODEL_ID + ".pth"))
    net_D.load_state_dict(torch.load("models/discriminator_" + MODEL_ID + ".pth"))
except FileNotFoundError:
    torch.save(net_G.state_dict(), "models/generator_" + MODEL_ID + ".pth")
    torch.save(net_D.state_dict(), "models/discriminator_" + MODEL_ID + ".pth")

if ON_CUDA:
    net_G.cuda()
    net_D.cuda()

optim_G = torch.optim.RMSprop(net_G.parameters(), lr = LEARN_RATE)
optim_D = torch.optim.RMSprop(net_D.parameters(), lr = LEARN_RATE)


min_lossG = np.inf
min_lossD = np.inf
for epoch in range(START_EPOCH, N_EPOCHS):
    for batch_n, (images, label) in enumerate(data_loader):
        if ON_CUDA:
            images = images.cuda()
        
        # Generate fake batch.
        z = torch.randn(BATCH_SIZE, 100, 1, 1, device = device)
        fake_images = net_G.forward(z)

        optim_D.zero_grad()

        errD = -torch.mean(net_D(images)) + torch.mean(net_D(fake_images.detach()))
        errD.backward()
        optim_D.step()

        for module in net_D.parameters():
            module.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        if batch_n % N_CRITIC == 0:
            optim_G.zero_grad()

            errG = -torch.mean(net_D(fake_images))
            errG.backward()
            optim_G.step()

    print("Epoch", epoch)
    print("--------------------")
    print("Generator Loss:", errG.item())
    print("Critic Loss:", errD.item())

    torch.save(net_G.state_dict(), "models/generator_" + MODEL_ID + ".pth")
    torch.save(net_D.state_dict(), "models/discriminator_" + MODEL_ID + ".pth")

    # Save some Generator outputs to track progress visually.
    os.makedirs("outputs/Epoch" + str(epoch), exist_ok = True)
    for i in range(4):
        img = fake_images[i].cpu().detach().numpy().transpose(1, 2, 0) * 255
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("outputs/Epoch" + str(epoch) + "/" + "epoch_" + str(epoch) + "_" + str(i) + ".jpg", img)
