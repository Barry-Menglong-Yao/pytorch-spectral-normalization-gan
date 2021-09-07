from __future__ import division, print_function
import torch.nn as nn
 
from . import mmd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.distributions.uniform import Uniform
from torch.autograd import Variable
import matplotlib.pyplot as plt
class Morphing(nn.Module):
   
    def forward(self    ):
        pass
    #langevin
    def morph_z(self, z,  generator, discriminator,real_images ):
        if not z.requires_grad:
            z.requires_grad = True
        self.z=z
        self.images=real_images
         
    

        step_lr = self.lan_step_lr
        step_lr=torch.tensor(step_lr)
        noise_std = torch.sqrt(step_lr * 2) * 0.01
        kernel = getattr(mmd, '_rbf_kernel' ) 
        self.z_l = self.z
        d_i = discriminator(self.images )[0]
        # history=HistoryZ()
        for i in range(self.lan_steps):
            
            self.sample_one_step(kernel,d_i,step_lr,noise_std,generator, discriminator, self.batch_size,  self.z_dim, i)
        # self.log_z_one_data(self.z_l)
        # history.draw_z(  )
        return self.z_l
            # self.G_lan = generator(self.z_l, self.batch_size, update_collection=update_collection)
            # convert to NHWC format for sampling images
            #TODO why transpose? self.G_lan = torch.transpose(self.G_lan, [0, 2, 3, 1])




    def sample_one_step(self,kernel,d_i,step_lr,noise_std,generator, discriminator, batch_size,  z_dim, i):
        current_g = generator(self.z_l,None,None,None,None )
        d_g = discriminator(current_g )[0]
        # note that we should use k(x,tf.stop_gradient(x)) instead of k(x,x), but k(x,x) also works very well
        _, kxy, _, _, = kernel(d_g, d_i)
        _, kxx, _, _, = kernel(d_g, d_g.detach())
        # KL divergence
        # energy = -torch.log(torch.mean(kxy, axis=-1) + 1e-10) + torch.log(
        #     torch.mean(delete_diag(kxx), axis=-1) / (batch_size- 1) + 1e-10)
        energy=-d_g
        energy=torch.squeeze(energy)
            
        z_grad = torch.autograd.grad(energy, self.z_l,grad_outputs=torch.ones(self.z_l.shape[0]).cuda())[0]
  
        # energy.backward(gradient=torch.ones(batch_size).cuda() )
        
        # z_grad=self.z_l.grad
        z_update=step_lr * z_grad
         
        # self.log_z_one_step( self.z_l,z_grad,z_update)
        self.z_l = self.z_l - z_update
        self.z_l += torch.normal( mean=0., std=noise_std,size=( batch_size,  z_dim)).cuda()
      
    def log_z_one_data(self, z ):
        self.writer.add_scalar("z_min_for_G",torch.min(z).item(),global_step=self.train_step)
        self.writer.add_scalar("z_max_for_G",torch.max(z).item(),global_step=self.train_step)
        self.writer.add_histogram("z_distribution_for_G",z,global_step=self.train_step)
        self.writer.close()
        self.train_step+=1

    def log_z_one_step(self, z,z_grad,z_update):
        
        self.writer.add_scalar("z_min",torch.min(z).item(),global_step=self.morph_step)
        self.writer.add_scalar("z_max",torch.max(z).item(),global_step=self.morph_step)
        self.writer.add_scalar("z_grad_min",torch.min(z_grad).item(),global_step=self.morph_step)
        self.writer.add_scalar("z_grad_max",torch.max(z_grad).item(),global_step=self.morph_step)
        self.writer.add_scalar("z_update_min",torch.min(z_update).item(),global_step=self.morph_step)
        self.writer.add_scalar("z_update_max",torch.max(z_update).item(),global_step=self.morph_step)
        self.writer.add_histogram("z_distribution",z,global_step=self.morph_step)
 
        self.writer.add_histogram("z_grad_distribution",z_grad,global_step=self.morph_step)
 
        self.writer.add_histogram("z_update_distribution",z_update,global_step=self.morph_step)
        self.writer.close()
        
        self.morph_step+=1


    def __init__(self,  lan_step_lr,lan_steps,   batch_size,  z_dim,images):
        super().__init__()
        self.writer= SummaryWriter()
        self.lan_step_lr = lan_step_lr
        self.lan_steps=lan_steps
        self.morph_step=0
        self.train_step=0
        self.batch_size=batch_size
        self.z_dim=z_dim
        self.images=images
   

    
class HistoryZ():
    def __init__(self):
        self.step=[]
        self.z_min=[] 
        self.z_max=[] 
        self.z_grad_min=[]
        self.z_update_min=[]
        self.z_grad_max=[]
        self.z_update_max=[]

    def update(self,idx,z,z_grad,z_update):
        self.step.append(idx)
        self.z_min.append(torch.min(z).item())
        self.z_max.append(torch.max(z).item())
        self.z_grad_min.append(torch.min(z_grad).item())
        self.z_grad_max.append(torch.max(z_grad).item())
        self.z_update_min.append(torch.min(z_update).item())
        self.z_update_max.append(torch.max(z_update).item())
 

    def draw_z(self ):
        step=self.step
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  
        ax1.plot(step, self.z_min, label='z_min')  
        ax1.plot(step, self.z_max, label='z_max')   
        ax2.plot(step, self.z_grad_min, label='z_grad_min')
        ax2.plot(step, self.z_grad_max, label='z_grad_max')  
        ax3.plot(step, self.z_update_min, label='z_update_min')   
        ax3.plot(step, self.z_update_max, label='z_update_max')
 
        ax1.set_xlabel('x label')  # Add an x-label to the axes.
        ax1.set_ylabel('y label')  # Add a y-label to the axes.
        ax1.set_title("Simple Plot")  # Add a title to the axes.
        ax1.legend()  # Add a legend.
        ax2.set_xlabel('x label')  # Add an x-label to the axes.
        ax2.set_ylabel('y label')  # Add a y-label to the axes.
        ax2.set_title("Simple Plot")  # Add a title to the axes.
        ax2.legend()  # Add a legend.
        ax3.set_xlabel('x label')  # Add an x-label to the axes.
        ax3.set_ylabel('y label')  # Add a y-label to the axes.
        ax3.set_title("Simple Plot")  # Add a title to the axes.
        ax3.legend()  # Add a legend.
        fig.savefig('fine_tune_vae/logs/plot/z.png')

        # plt.plot(step, self.z_min, label='z_min')  
        # plt.plot(step, self.z_max, label='z_max')   
        # plt.plot(step, self.z_grad_min, label='z_grad_min')
        # plt.plot(step, self.z_grad_max, label='z_grad_max')  
        # plt.plot(step, self.z_update_min, label='z_update_min')   
        # plt.plot(step, self.z_update_max, label='z_update_max')
        # plt.xlabel('x label')
        # plt.ylabel('y label')
        # plt.title("Simple Plot")
        # plt.legend()
        # plt.savefig('fine_tune_vae/logs/plot/z.png')
 

def delete_diag(matrix):
    return matrix -torch.diag(torch.diag(matrix))  # return matrix, while k_ii is 0  TODO only support 2D, check it 



if __name__ == '__main__':
    pass