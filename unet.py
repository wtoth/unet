import torch 
from torch import nn
import torch.nn.functional as F

class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.contraction1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3),
            nn.ReLU(True)
        )
        self.contraction2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3),
            nn.ReLU(True)
        )
        self.contraction3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3),
            nn.ReLU(True)
        )
        self.contraction4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3),
            nn.ReLU(True)
        )
        self.contraction5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=3),
            nn.ReLU(True)
        )
        self.max_pool = nn.MaxPool2d(2, stride=2)
        
        # Expansion Steps
        self.up_conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.expansion1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        )
        self.expansion2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        )
        self.expansion3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        )
        self.expansion4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1)
        )


    def forward(self, x):
        # Contraction Steps
        contraction1_out = self.contraction1(x)
        x = self.max_pool(contraction1_out)
        contraction2_out = self.contraction2(x)
        x = self.max_pool(contraction2_out)
        contraction3_out = self.contraction1(x)
        x = self.max_pool(contraction3_out)
        contraction4_out = self.contraction4(x)
        x = self.max_pool(contraction4_out)
        contraction5_out = self.contraction5(x)

        # Expansion Steps 
        x = self.up_conv1(contraction5_out) # up conv
        x = x.concat(contraction4_out) # concat contraction val
        x = self.expansion1(x) # Expand

        x = self.up_conv2(x)
        x = x.concat(contraction3_out)
        x = self.expansion2(x)

        x = self.up_conv3(x)
        x = x.concat(contraction2_out)
        x = self.expansion3(x)

        x = self.up_conv4(x)
        x = x.concat(contraction1_out)
        x = self.expansion4(x)
        return x

