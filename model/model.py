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
    def __init__(self, z_dim,inject_type,inject_layer_list):
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
        if inject_type=="fc":
            if "1" in self.inject_layer_list:
                self.fc1=nn.Sequential(nn.Linear(512*4*4 , 512*4*4), nn.ReLU()) 
            if "2" in self.inject_layer_list:
                self.fc2=nn.Sequential(nn.Linear(256*8*8 , 256*8*8), nn.ReLU()) 
            if "3" in self.inject_layer_list:
                self.fc3=nn.Sequential(nn.Linear(128*16*16 , 128*16*16), nn.ReLU()) 
            if "4" in self.inject_layer_list:
                self.fc4=nn.Sequential(nn.Linear(64*32*32 , 64*32*32), nn.ReLU()) 
        elif inject_type=="conv":
            self.conv1=nn.Sequential(
            nn.Conv2d(512, 512, 4  ),
            
            nn.ReLU())
            self.conv2=nn.Sequential(
            nn.Conv2d(256, 256, 8),
            
            nn.ReLU()) 
            self.conv3=nn.Sequential(
            nn.Conv2d(128, 128, 16),
            
            nn.ReLU()) 
            self.conv4=nn.Sequential(
            nn.Conv2d(64, 64, 32),
            
            nn.ReLU()) 
        else:
            self.cat_conv1=nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=(1,1)),
            
            nn.ReLU())
            self.cat_conv2=nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 
            self.cat_conv3=nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 
            self.cat_conv4=nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=(1,1)),
            
            nn.ReLU()) 

    def forward(self, z,x1,x2,x3,x4):
        z=z.view(-1, self.z_dim, 1, 1)
        up1=self.conv_norm1(z)#128,1,1->64,512,4,4
        if x4!=None and "1" in self.inject_layer_list:
            up1=self.inject(up1,x4, self.inject_type,"1")
        
        up2=self.conv_norm2(up1)#512,4,4->256,8,8
        if x3!=None and "2" in self.inject_layer_list:
            up2=self.inject(up2,x3, self.inject_type,"2")
        up3=self.conv_norm3(up2)#256,8,8->128,16,16
        if x2!=None and "3" in self.inject_layer_list:
            up3=self.inject(up3,x2, self.inject_type,"3")
        up4=self.conv_norm4(up3)#128,16,16->64,32,32
        if x1!=None and "4" in self.inject_layer_list:
            up4=self.inject(up4,x1, self.inject_type,"4")
        up5=self.conv5(up4)
        return up5
    def inject(self,up,x,inject_type,inject_phase):
        inject_func=self.get_inject_func(inject_type,inject_phase)
        if inject_type=="conv":
            inject_x=inject_func(x)
            up1=up+inject_x
        elif inject_type=="fc":
            b,c,w,h=x.shape
            x=x.view(-1,c*w*h)
            inject_x=inject_func(x)
            inject_x=inject_x.view(b,c,w,h)
            up1=up+inject_x
        else:
            cat_x = torch.cat([x, up], dim=1)
            up1=inject_func(cat_x)
        
        return up1

    def get_inject_func(self,inject_type,inject_phase):
        if inject_type=="fc":
            if inject_phase=="1":
                return self.fc1
            elif inject_phase=="2":
                return self.fc2
            elif inject_phase=="3":
                return self.fc3 
            else: 
                return self.fc4
        elif inject_type=="conv":
            if inject_phase=="1":
                return self.conv1
            elif inject_phase=="2":
                return self.conv2
            elif inject_phase=="3":
                return self.conv3 
            else: 
                return self.conv4
        else:
            if inject_phase=="1":
                return self.cat_conv1
            elif inject_phase=="2":
                return self.cat_conv2
            elif inject_phase=="3":
                return self.cat_conv3 
            else: 
                return self.cat_conv4    

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
            morphed_z=self.morphing.morph_z(gen_z_of_real_img,self.generator, self.discriminator )
            reconstructed_img=self.generator(morphed_z,x1,x2,x3,x4)
        else:
            reconstructed_img=self.generator(gen_z_of_real_img,x1,x2,x3,x4)
       
        return  reconstructed_img,mu,log_var

    def sample(self,z):
        if self.lan_steps > 0:
            morphed_z=self.morphing.morph_z(z,self.generator, self.discriminator )
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
    

    def sample(self,z):
        if self.morphing.lan_steps > 0:
            morphed_z=self.morphing.morph_z(z,self.generator, self.discriminator  )
            generated_img=self.generator(morphed_z)
        else:
            generated_img=self.generator(z)
        return generated_img

 
         