import torch
import torch.nn as nn
import cv2
import numpy as np

ON_CUDA = torch.cuda.is_available()
if ON_CUDA:
    device = "cuda:0"
else:
    device = "cpu"

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


net_G = Generator()
net_G.load_state_dict(torch.load("models/generator_0_e1100.pth", map_location = torch.device(device)))
if ON_CUDA:
    net_G.cuda()

while True:
    z = torch.randn(1, 100, 1, 1, device = device)
    img = net_G.forward(z)
    img = (img.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((64, 64, 3))) / 2
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imshow("CAT", img)
    cv2.waitKey(1000)
