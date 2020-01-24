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
LEN_Z           = 100
OUT_CHANNELS    = 3
IMAGE_DIM       = 64
BATCH_SIZE      = 64
LEARN_RATE      = 0.00005
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


# Wasserstein loss function
def wasserstein(prediction, label):
    return torch.mean(prediction * label)


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
    for images, label in data_loader:
        if ON_CUDA:
            images = images.cuda()
        
        # Generate fake batch.
        z = torch.randn(BATCH_SIZE, 100, 1, 1, device = device)
        fake_images = net_G.forward(z)

        for _ in range(5):
            net_D.zero_grad()

            # Real batch.
            output = net_D.forward(images)
            D_x = output.mean().item()

            errD_real = wasserstein(output, 1)
            errD_real.backward()

            # Fake batch.
            output = net_D.forward(fake_images.detach())
            D_G_z1 = output.mean().item()

            errD_fake = wasserstein(output, -1)
            errD_fake.backward()

            optim_D.step()
            errD = errD_real - errD_fake
        
        # Train Generator.
        net_G.zero_grad()
        output = net_D.forward(fake_images)

        errG = wasserstein(output, -1)
        errG.backward()

        optim_G.step()

        D_G_z2 = output.mean().item()

    print("Epoch", epoch)
    print("--------------------")
    print("Generator Loss:", errG.item())
    print("Critic Loss:", errD.item())
    print("D(x) =", D_x)
    print("(1) D(G(z)) =", D_G_z1)
    print("(2) D(G(z)) =", D_G_z2)

    torch.save(net_G.state_dict(), "models/generator_" + MODEL_ID + ".pth")
    torch.save(net_D.state_dict(), "models/discriminator_" + MODEL_ID + ".pth")

    # Save some Generator outputs to track progress visually.
    os.makedirs("outputs/Epoch" + str(epoch), exist_ok = True)
    for i in range(4):
        img = fake_images[i].cpu().detach().numpy().transpose(1, 2, 0) * 255
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("outputs/Epoch" + str(epoch) + "/" + "epoch_" + str(epoch) + "_" + str(i) + ".jpg", img)
