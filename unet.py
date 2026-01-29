import torch 
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
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
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

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
        print("1", contraction1_out.shape)
        x = self.max_pool(contraction1_out)
        print("1->2", x.shape)
        contraction2_out = self.contraction2(x)
        print("2", contraction2_out.shape)
        x = self.max_pool(contraction2_out)
        print("2->3", x.shape)
        contraction3_out = self.contraction3(x)
        print("3", contraction3_out.shape)
        x = self.max_pool(contraction3_out)
        print("3->4", x.shape)
        contraction4_out = self.contraction4(x)
        print("4", contraction4_out.shape)
        x = self.max_pool(contraction4_out)
        print("4->5", x.shape)
        contraction5_out = self.contraction5(x)
        print("5", contraction5_out.shape)

        # Expansion Steps 
        x = self.up_conv1(contraction5_out) # up conv
        print(x.shape, contraction4_out.shape)
        x = torch.cat((x, contraction4_out)) # x.concat(contraction4_out) # concat contraction val
        x = self.expansion1(x) # Expand

        x = self.up_conv2(x)
        x = torch.cat((x, contraction3_out)) # x.concat(contraction3_out)
        x = self.expansion2(x)

        x = self.up_conv3(x)
        x = torch.cat((x, contraction2_out)) # x.concat(contraction2_out)
        x = self.expansion3(x)

        x = self.up_conv4(x)
        x = torch.cat((x, contraction1_out)) # x.concat(contraction1_out)
        x = self.expansion4(x)
        return x

