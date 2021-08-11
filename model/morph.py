from __future__ import division, print_function
import torch.nn as nn
 
from . import mmd
 
import torch
from torch.distributions.uniform import Uniform
from torch.autograd import Variable

class Morphing(nn.Module):
   
    def forward(self    ):
        pass
    #langevin
    def morph_z(self, z,  generator, discriminator ):
        z.requires_grad = True
        self.z=z
       
         
    

        step_lr = self.lan_step_lr
        step_lr=torch.tensor(step_lr)
        noise_std = torch.sqrt(step_lr * 2) * 0.01
        kernel = getattr(mmd, '_rbf_kernel' ) 
        self.z_l = self.z
        d_i = discriminator(self.images )[0]
        for i in range(self.lan_steps):
            self.sample_one_step(kernel,d_i,step_lr,noise_std,generator, discriminator, self.batch_size,  self.z_dim)
            
        return self.z_l
            # self.G_lan = generator(self.z_l, self.batch_size, update_collection=update_collection)
            # convert to NHWC format for sampling images
            #TODO why transpose? self.G_lan = torch.transpose(self.G_lan, [0, 2, 3, 1])


    def sample_one_step(self,kernel,d_i,step_lr,noise_std,generator, discriminator, batch_size,  z_dim ):
        current_g = generator(self.z_l )
        d_g = discriminator(current_g )[0]
        # note that we should use k(x,tf.stop_gradient(x)) instead of k(x,x), but k(x,x) also works very well
        _, kxy, _, _, = kernel(d_g, d_i)
        _, kxx, _, _, = kernel(d_g, d_g.detach())
        # KL divergence
        energy = -torch.log(torch.mean(kxy, axis=-1) + 1e-10) + torch.log(
            torch.mean(delete_diag(kxx), axis=-1) / (batch_size- 1) + 1e-10)
    
            
        z_grad = torch.autograd.grad(energy, self.z_l,grad_outputs=torch.ones(self.z_l.shape[0]).cuda())[0]
  
        # energy.backward(gradient=torch.ones(batch_size).cuda() )
        
        # z_grad=self.z_l.grad
        self.z_l = self.z_l - step_lr * z_grad
        self.z_l += torch.normal( mean=0., std=noise_std,size=( batch_size,  z_dim)).cuda()
      



    def __init__(self,  lan_step_lr,lan_steps,   batch_size,  z_dim,images):
        super().__init__()
 
        self.lan_step_lr = lan_step_lr
        self.lan_steps=lan_steps
           
        self.batch_size=batch_size
        self.z_dim=z_dim
        self.images=images
   
  
    
   
 

def delete_diag(matrix):
    return matrix -torch.diag(torch.diag(matrix))  # return matrix, while k_ii is 0  TODO only support 2D, check it 



if __name__ == '__main__':
    pass