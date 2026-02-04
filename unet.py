import torch 
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.contraction1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.contraction2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.contraction3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.contraction4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.contraction5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.3) # Section 3.1 "Dropout layers at teh end fo the contracting path"
        )
        self.max_pool = nn.MaxPool2d(2, stride=2)
        
        # Expansion Steps
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.expansion1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        )
        self.expansion2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        )
        self.expansion3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        )
        self.expansion4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        )


    def forward(self, x):
        # Contraction Steps
        contraction1_out = self.contraction1(x)
        x = self.max_pool(contraction1_out)
        contraction2_out = self.contraction2(x)
        x = self.max_pool(contraction2_out)
        contraction3_out = self.contraction3(x)
        x = self.max_pool(contraction3_out)
        contraction4_out = self.contraction4(x)
        x = self.max_pool(contraction4_out)
        contraction5_out = self.contraction5(x)

        # Expansion Steps 
        x = self.up_conv1(contraction5_out) # up conv
        cropped_contraction4 = contraction4_out[:,:,:x.size(2),:x.size(3)]
        x = torch.cat((x, cropped_contraction4), dim=1) # x.concat(contraction4_out) # concat contraction val
        x = self.expansion1(x) # Expand

        x = self.up_conv2(x)
        cropped_contraction3 = contraction3_out[:,:,:x.size(2),:x.size(3)]
        x = torch.cat((x, cropped_contraction3), dim=1) # x.concat(contraction3_out)
        x = self.expansion2(x)

        x = self.up_conv3(x)
        cropped_contraction2 = contraction2_out[:,:,:x.size(2),:x.size(3)]
        x = torch.cat((x, cropped_contraction2), dim=1) # x.concat(contraction2_out)
        x = self.expansion3(x)

        x = self.up_conv4(x)
        cropped_contraction1 = contraction1_out[:,:,:x.size(2),:x.size(3)]
        x = torch.cat((x, cropped_contraction1), dim=1) # x.concat(contraction1_out)
        x = self.expansion4(x)

        return x

