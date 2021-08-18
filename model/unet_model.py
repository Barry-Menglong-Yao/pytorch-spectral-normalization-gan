""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn

from model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x,c):
        x1 = self.inc(x) #64,3,32,32 ->64,64,32,32
        x2 = self.down1(x1)#64,64,32,32 ->64,128,16,16
        x3 = self.down2(x2)#64,128,16,16 ->64,256,8,8
        x4 = self.down3(x3)#64,256,8,8 -> 64,512,4,4
        x5 = self.down4(x4)# 64,512,4,4 ->  64,512,2,2
        x = self.up1(x5, x4)#64,512,2,2->64,512,4,4->64,1024,4,4->64,256,4,4
        x = self.up2(x, x3)#64,256,4,4 ->256,8,8 ->512,8,8 ->64,128,8,8
        x = self.up3(x, x2)#64,128,8,8 ->64,64,16,16
        x = self.up4(x, x1)#64,64,16,16->64,64,32,32
        logits = self.outc(x)#64,64,32,32->64,1,32,32
        return logits



class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.down1 = Down(64, 128)


        

    def forward(self, z):
        z=self.down1(z)
 
        
        return z

class Discriminator(nn.Module):
    def __init__(self,has_vae,z_dim):
        super(Discriminator, self).__init__()
        self.down1 = Down(64, 128)
         

    def forward(self, x):
        x=self.down1(x)
        mu=None 
        log_var=None 
        z=None

        return x,z,mu,log_var
 

 
         

