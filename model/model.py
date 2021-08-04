# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import torch
from model.spectral_normalization import SpectralNorm
import torch
channels = 3
leak = 0.1
w_g = 4

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # self.c_dim=c_

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

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
        self.has_vae=has_vae

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m=m.view(-1,w_g * w_g * 512)
        out=self.fc(m)

        if self.has_vae:
            mu = self.fc_mu(m)
            log_var = self.fc_var(m)
            z = self.reparameterize(mu, log_var)
        else:
            mu=None 
            log_var=None 
            z=None

        return out,z,mu,log_var

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
    def __init__(self, discriminator ,generator  ):
        super().__init__()
        self.discriminator=discriminator
   
 
        self.generator=generator
        

 
    def forward(self, real_img, real_c   ):
        real_logits,gen_z_of_real_img ,mu,log_var = self.discriminator(real_img  )
        reconstructed_img = self.generator(gen_z_of_real_img)
        return  reconstructed_img,mu,log_var

    def loss(self, reconstructed_img, real_img,mu,log_var,vae_beta,vae_alpha  ):
        recons_loss =F.mse_loss(reconstructed_img, real_img)
        weighted_recons_loss=recons_loss.mul(vae_alpha )
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        weighted_kld_loss=kld_loss.mul( vae_beta)
        batch_size=reconstructed_img.shape[0]
        kld_weight=batch_size/50000
        vae_loss = weighted_recons_loss + kld_weight * weighted_kld_loss
        return vae_loss,recons_loss,kld_loss
         