# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import torch
from model.spectral_normalization import SpectralNorm
import torch
channels = 3
leak = 0.1
w_g = 4


class ConvNorm(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
   
        self.conv = nn.Sequential(
      
            nn.ConvTranspose2d(in_channels, out_channels, 4,  stride=2, padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

# class Generator(nn.Module):
#     def __init__(self, z_dim):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
    
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
#             nn.Tanh())
 

#     def forward(self, z,x1,x2,x3,x4):
#         return self.model(z.view(-1, self.z_dim, 1, 1))



class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
    

        self.conv_norm1=nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv_norm2=ConvNorm(512, 256)
        self.conv_norm3=ConvNorm(256, 128)
        self.conv_norm4=ConvNorm(128, 64)
        self.conv5=nn.Sequential(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())


        

    def forward(self, z,x1,x2,x3,x4):
        z=z.view(-1, self.z_dim, 1, 1)
        up1=self.conv_norm1(z)#128,1,1->64,512,4,4
        up2=self.conv_norm2(up1)#512,4,4->256,8,8
        up3=self.conv_norm3(up2)#256,8,8->128,16,16
        up4=self.conv_norm4(up3)#128,16,16->64,32,32
        up5=self.conv5(up4)
        return up5

class UnetGenerator(nn.Module):
    def __init__(self, z_dim,inject_type,inject_layer_list,is_drop_out):
        super(UnetGenerator, self).__init__()
        self.z_dim = z_dim
    

        self.conv_norm1=nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv_norm2=ConvNorm(512, 256)
        self.conv_norm3=ConvNorm(256, 128)
        self.conv_norm4=ConvNorm(128, 64)
        self.conv5=nn.Sequential(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())
        self.inject_type=inject_type
        self.inject_layer_list=inject_layer_list
        self.drop_out1=nn.Dropout(p=0.3)
        self.drop_out2=nn.Dropout(p=0.3)
        self.drop_out3=nn.Dropout(p=0.3)
        self.drop_out4=nn.Dropout(p=0.3)
        self.is_drop_out=is_drop_out
        self.norm1=None 
        self.norm2=None
        self.norm3=None 
        self.norm4=None
        if inject_type=="fc":
            if "1" in self.inject_layer_list:
                self.inject_layer1=nn.Sequential(nn.Linear(512*4*4 , 512*4*4), nn.ReLU()) 
            if "2" in self.inject_layer_list:
                self.inject_layer2=nn.Sequential(nn.Linear(256*8*8 , 256*8*8), nn.ReLU()) 
            if "3" in self.inject_layer_list:
                self.inject_layer3=nn.Sequential(nn.Linear(128*16*16 , 128*16*16), nn.ReLU()) 
            if "4" in self.inject_layer_list:
                self.inject_layer4=nn.Sequential(nn.Linear(64*32*32 , 64*32*32), nn.ReLU()) 
        
        elif inject_type=="conv":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 512,   3,   padding=(1,1)  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 256, 3,   padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 128,  3,   padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 64,  3,   padding=(1,1)),
            
            nn.ReLU()) 
        elif inject_type=="conv_broadcast":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 512, 4  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 256, 8),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 128, 16),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 64, 32),
            
            nn.ReLU()) 
        elif inject_type=="pool":
            self.inject_layer1= nn.AvgPool2d(kernel_size = 4   )
            self.inject_layer2= nn.AvgPool2d(kernel_size = 8 )
            self.inject_layer3= nn.AvgPool2d(kernel_size = 16 )
            self.inject_layer4= nn.AvgPool2d(kernel_size = 32 )
        elif inject_type=="cat":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=(1,1)),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 
        elif inject_type=="layer":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 512, 4  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 256, 8),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 128, 16),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 64, 32),
            
            nn.ReLU())
            self.norm1=nn.LayerNorm([512,1,1])
            self.norm2=nn.LayerNorm([256,1,1]) 
            self.norm3=nn.LayerNorm([128,1,1]) 
            self.norm4=nn.LayerNorm([64,1,1]) 
        elif inject_type=="group" or inject_type=="group_in_y":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 512, 4  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 256, 8),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 128, 16),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 64, 32),
            
            nn.ReLU())
            self.norm1=nn.GroupNorm(32,512)
            self.norm2=nn.GroupNorm(16,256)
            self.norm3=nn.GroupNorm(8,128)
            self.norm4=nn.GroupNorm(4,64)
        elif inject_type=="layer_in_y":
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 512, 4  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 256, 8),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 128, 16),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 64, 32),
            
            nn.ReLU())
            self.norm1=nn.LayerNorm([512,4,4])
            self.norm2=nn.LayerNorm([256,8,8]) 
            self.norm3=nn.LayerNorm([128,16,16]) 
            self.norm4=nn.LayerNorm([64,32,32]) 
        elif inject_type=="single_channel": 
            self.inject_layer1=nn.Sequential(
            nn.Conv2d(512, 1,   3,   padding=(1,1)  ),
            
            nn.ReLU())
            self.inject_layer2=nn.Sequential(
            nn.Conv2d(256, 1, 3,   padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer3=nn.Sequential(
            nn.Conv2d(128, 1,  3,   padding=(1,1)),
            
            nn.ReLU()) 
            self.inject_layer4=nn.Sequential(
            nn.Conv2d(64, 1,  3,   padding=(1,1)),
            
            nn.ReLU()) 
        else:
            print("wrong inject_type")

    def forward(self, z,x1,x2,x3,x4):
        z=z.view(-1, self.z_dim, 1, 1)
        up1=self.conv_norm1(z)#128,1,1->64,512,4,4
        if x4!=None and "1" in self.inject_layer_list:
            if self.is_drop_out:
                x4=self.drop_out1(x4)
            up1=self.inject(up1,x4, self.inject_type,"1",self.norm1)
        
        up2=self.conv_norm2(up1)#512,4,4->256,8,8
        if x3!=None and "2" in self.inject_layer_list:
            if self.is_drop_out:
                x3=self.drop_out2(x3)
            up2=self.inject(up2,x3, self.inject_type,"2",self.norm2)
        up3=self.conv_norm3(up2)#256,8,8->128,16,16
        if x2!=None and "3" in self.inject_layer_list:
            if self.is_drop_out:
                x2=self.drop_out3(x2)
            up3=self.inject(up3,x2, self.inject_type,"3",self.norm3)
        up4=self.conv_norm4(up3)#128,16,16->64,32,32
        if x1!=None and "4" in self.inject_layer_list:
            if self.is_drop_out:
                x1=self.drop_out4(x1)
            up4=self.inject(up4,x1, self.inject_type,"4",self.norm4)
        up5=self.conv5(up4)
        return up5
    def inject(self,up,x,inject_type,inject_phase,norm_func):
        inject_func=self.get_inject_func(inject_type,inject_phase)
        if inject_type=="conv" or inject_type=="conv_broadcast" or   inject_type=="pool" or inject_type=="single_channel": 
            inject_x=inject_func(x)
            up1=up+inject_x
        elif inject_type=="fc":
            b,c,w,h=x.shape
            x=x.view(-1,c*w*h)
            inject_x=inject_func(x)
            inject_x=inject_x.view(b,c,w,h)
            up1=up+inject_x
        elif inject_type=="cat":
            cat_x = torch.cat([x, up], dim=1) 
            up1=inject_func(cat_x)
        elif inject_type=="layer" or inject_type=="group":
            inject_x=inject_func(x)
            inject_x=norm_func(inject_x)
            up1=up+inject_x
        elif inject_type=="layer_in_y" or inject_type=="group_in_y":
            inject_x=inject_func(x)
            up1=up+inject_x
            up1=norm_func(up1)
      

        else:
            print("wrong inject_type")
             
        
        return up1

    def get_inject_func(self,inject_type,inject_phase):
        if inject_phase=="1":
            return self.inject_layer1
        elif inject_phase=="2":
            return self.inject_layer2
        elif inject_phase=="3":
            return self.inject_layer3 
        else: 
            return self.inject_layer4
         

class Discriminator(nn.Module):
    def __init__(self,has_vae,z_dim):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))
        if has_vae:
            self.fc_mu = SpectralNorm(nn.Linear(w_g * w_g * 512, z_dim))
            self.fc_var = SpectralNorm(nn.Linear(w_g * w_g * 512, z_dim))
            # self.fc_mu = nn.Linear(w_g * w_g * 512, z_dim)  #TODO no spectralNorm?
            # self.fc_var = nn.Linear(w_g * w_g * 512, z_dim) 
        self.has_vae=has_vae

    def forward(self, x):
        m = x
        x1 = nn.LeakyReLU(leak)(self.conv1(m)) #64,32,32
        x2 = nn.LeakyReLU(leak)(self.conv2(x1)) #64,16,16
        x2 = nn.LeakyReLU(leak)(self.conv3(x2))#128,16
        x3 = nn.LeakyReLU(leak)(self.conv4(x2))#128,8
        x3 = nn.LeakyReLU(leak)(self.conv5(x3))#256,8
        x4 = nn.LeakyReLU(leak)(self.conv6(x3))#256,4
        x4 = nn.LeakyReLU(leak)(self.conv7(x4))#512,4
        flag_down3=x4.view(-1,w_g * w_g * 512)
        out=self.fc(flag_down3)

        if self.has_vae:
            mu = self.fc_mu(flag_down3)
            log_var = self.fc_var(flag_down3)
            z = self.reparameterize(mu, log_var)
        else:
            mu=None 
            log_var=None 
            z=None

        return out,z,mu,log_var,x1,x2,x3,x4

    def reparameterize(self, mu , logvar )  :
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu




class VaeGan( nn.Module):
    def __init__(self, discriminator ,generator ,morphing ):
        super().__init__()
        self.discriminator=discriminator
   
 
        self.generator=generator
        self.morphing=morphing
        if morphing!=None:
            self.lan_steps=self.morphing.lan_steps 
        else: 
            self.lan_steps=0
   
 
    def forward(self, real_img, real_c , **kwargs  ):
        real_logits,gen_z_of_real_img ,mu,log_var,x1,x2,x3,x4 = self.discriminator(real_img  )
        if self.lan_steps > 0:
            morphed_z=self.morphing.morph_z(gen_z_of_real_img,self.generator, self.discriminator,real_img )
            reconstructed_img=self.generator(morphed_z,x1,x2,x3,x4)
        else:
            reconstructed_img=self.generator(gen_z_of_real_img,x1,x2,x3,x4)
       
        return  reconstructed_img,mu,log_var

    def sample(self,z,real_images):
        if self.lan_steps > 0:
            morphed_z=self.morphing.morph_z(z,self.generator, self.discriminator,real_images )
            generated_img=self.generator(morphed_z,None,None,None,None)
        else:
            generated_img=self.generator(z,None,None,None,None)
        return generated_img

 
    def loss_function(self, reconstructed_img, real_img,mu,log_var,kld_weight,vae_beta=1,vae_alpha=1  ):
        recons_loss =F.mse_loss(reconstructed_img, real_img)
        weighted_recons_loss=recons_loss.mul(vae_alpha )
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        weighted_kld_loss=kld_loss.mul( vae_beta)
  
        
        vae_loss = weighted_recons_loss + kld_weight * weighted_kld_loss
        return vae_loss,recons_loss,kld_loss
         



class Gan( nn.Module):
    def __init__(self, discriminator ,generator ,morphing ):
        super().__init__()
        self.discriminator=discriminator
   
 
        self.generator=generator
        self.morphing=morphing
    

    def sample(self,z,real_images):
        if self.morphing.lan_steps > 0:
            morphed_z=self.morphing.morph_z(z,self.generator, self.discriminator,real_images  )
            generated_img=self.generator(morphed_z,None,None,None,None)
        else:
            generated_img=self.generator(z,None,None,None,None)
        return generated_img

 
         