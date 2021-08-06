from __future__ import division, print_function
 
 
from . import mmd
import tensorflow as tf
import torch
from torch.distributions.uniform import Uniform
 

class GAN(object):
   

    #langevin
    def sample(self,  images, Generator, Discriminator, update_collection=None ):
        self.images = images
        self.z=Uniform(-1., 1.).sample([self.batch_size, self.z_dim]) 
        #TODO self.z = torch.random_uniform([self.batch_size, self.z_dim], minval=-1.,
        #                            maxval=1., dtype=torch.float32, name='z')
        self.load_model(Generator, Discriminator )
        if self.config.lan_steps > 0:
            step_lr = self.config.lan_step_lr
            noise_std = torch.sqrt(step_lr * 2) * 0.01
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel) 
            self.z_l = self.z
            d_i = self.discriminator(self.images, self.real_batch_size, return_layers=False, update_collection="NO_OPS")
            for i in range(self.config.lan_steps):
                self.sample_one_step(kernel,d_i,step_lr,noise_std)
            self.G_lan = self.generator(self.z_l, self.batch_size, update_collection=update_collection)
            # convert to NHWC format for sampling images
            self.G_lan = torch.transpose(self.G_lan, [0, 2, 3, 1])


    def sample_one_step(self,kernel,d_i,step_lr,noise_std):
        current_g = self.generator(self.z_l, self.batch_size, update_collection="NO_OPS")
        d_g = self.discriminator(current_g, self.batch_size, return_layers=False, update_collection="NO_OPS")
        # note that we should use k(x,tf.stop_gradient(x)) instead of k(x,x), but k(x,x) also works very well
        _, kxy, _, _, = kernel(d_g, d_i)
        _, kxx, _, _, = kernel(d_g, tf.stop_gradient(d_g))
        # KL divergence
        energy = -tf.log(tf.reduce_mean(kxy, axis=-1) + 1e-10) + tf.log(
            tf.reduce_sum(delete_diag(kxx), axis=-1) / (self.config.batch_size - 1) + 1e-10)
    #                     energy = -tf.log(tf.reduce_mean(kxy, axis=-1)) + tf.log(tf.reduce_mean(kxx, axis=-1))
            
        z_grad = tf.gradients(energy, self.z_l)[0]
        self.z_l = self.z_l - step_lr * z_grad
        self.z_l += tf.random_normal([self.batch_size, self.z_dim], mean=0., stddev=noise_std,
                                            dtype=tf.float32, )



    def __init__(self,  config):


        output_size = config.output_size
        self.output_size = output_size
        if config.real_batch_size == -1:
            config.real_batch_size = config.batch_size
        self.config = config
        self.batch_size = config.batch_size
        self.real_batch_size = config.real_batch_size
        self.z_dim = self.config.z_dim
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.dof_dim = self.config.dof_dim
        self.c_dim = config.c_dim
  
  
    
   

    
    def load_model(self,Generator, Discriminator ):
        dbn = self.config.batch_norm & (self.config.gradient_penalty <= 0)
        gen_kw = {
                'dim': self.gf_dim,
                'c_dim': self.c_dim,
                'output_size': self.output_size,
                'use_batch_norm': self.config.batch_norm,
                'format': self.format,
                'is_train': self.config.is_train,
            }
        disc_kw = {
            'dim': self.df_dim,
            'o_dim': self.dof_dim,
            'use_batch_norm': dbn,
            'with_sn': self.config.with_sn,
            'with_learnable_sn_scale': self.config.with_learnable_sn_scale,
            'format': self.format,
            'is_train': self.config.is_train,
            'scale': self.config.pico,
        }
        self.generator = Generator(**gen_kw)
        self.discriminator = Discriminator(**disc_kw)

def delete_diag(matrix):
    return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0   
