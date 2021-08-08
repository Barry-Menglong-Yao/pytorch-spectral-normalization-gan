import argparse
from util.enums import ModelAttribute
from model.morph import Morphing
from util.image import export_sample_images, save_image
from util.metrics.metric_utils import reconstruct
from util.metrics import metric_main
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import model_resnet
from model import model
from util import dnnlib
from util.data import load_dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
from util import constants

def update_discriminator(disc_iters,args,Z_dim,optim_disc,optim_gen,discriminator,generator,data):
    # update discriminator
    for _ in range(disc_iters):
        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge':
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
        elif args.loss == 'wasserstein':
            disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
        else:
            real_logic,_,_,_=discriminator(data)
            fake_logic,_,_,_=discriminator(generator(z))
            disc_loss = nn.BCEWithLogitsLoss()(real_logic, Variable(torch.ones(args.batch_size, 1).cuda())) + \
                nn.BCEWithLogitsLoss()(fake_logic, Variable(torch.zeros(args.batch_size, 1).cuda()))
        disc_loss.backward()
        optim_disc.step()
    return disc_loss

def update_generator(args,Z_dim,optim_disc,optim_gen,discriminator,generator):
    z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
    # update generator
    optim_disc.zero_grad()
    optim_gen.zero_grad()
    if args.loss == 'hinge' or args.loss == 'wasserstein':
        gen_loss = -discriminator(generator(z)).mean()
    else:
        fake_logic,_,_,_=discriminator(generator(z))
        gen_loss = nn.BCEWithLogitsLoss()(fake_logic, Variable(torch.ones(args.batch_size, 1).cuda()))
    gen_loss.backward()
    optim_gen.step()
    return gen_loss

def update_vae(img,c,vae,optim_vae,batch_size,vae_alpha,vae_beta):
    optim_vae.zero_grad()
     
    reconstructed_img, mu,log_var    = vae(img, c  )
    vae_loss,recons_loss,kld_loss=vae.loss(reconstructed_img, img,mu,log_var,vae_beta,vae_alpha)
    vae_loss.backward()
    optim_vae.step()
    return vae_loss,recons_loss,kld_loss

def load_optim(args,discriminator,generator,vae,model_attribute):
    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    if model_attribute.dgm_type.has_vae:
        optim_vae  = optim.Adam(vae.parameters(), lr=args.lr, betas=(0.0,0.9))
        scheduler_vae = optim.lr_scheduler.ExponentialLR(optim_vae, gamma=0.99)
    else:
        optim_vae=None
        scheduler_vae=None

    return optim_gen,optim_disc,optim_vae,scheduler_g,scheduler_d,scheduler_vae




def training_loop(args,run_dir):
    Z_dim = constants.Z_dim
    #number of updates to discriminator for every update to generator 
    disc_iters = 5
    model_attribute=ModelAttribute[args.model_type]
    loader,dataset=load_data(args.batch_size)
    grid_z, grid_size,real_images=export_sample_images(dataset, run_dir,  torch.device('cuda'), Z_dim,args.batch_size)
    generator,discriminator,vae=load_model(Z_dim,args.model_type,model_attribute,args.lan_step_lr,args.lan_steps,args.batch_size ,real_images)
    optim_gen,optim_disc,optim_vae,scheduler_g,scheduler_d,scheduler_vae=load_optim(args,discriminator,generator,vae,model_attribute)
    
    start_time = time.time()
    
    # fixed_z, sampled_imgs=sample_img_and_z(args,Z_dim,dataset,run_dir)
    for epoch in range(2000):
        train(epoch,loader,args,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,scheduler_d,scheduler_g,start_time,
        model_attribute,args.batch_size,vae,optim_vae,scheduler_vae,args.vae_alpha,args.vae_beta)
        if epoch%12==0 :
            evaluate(epoch,grid_z,generator,run_dir,discriminator,args.metrics,real_images,model_attribute,grid_size,vae)
            torch.save(discriminator.state_dict(), os.path.join(run_dir, 'checkpoint/disc_{}'.format(epoch)))
            torch.save(generator.state_dict(), os.path.join(run_dir, 'checkpoint/gen_{}'.format(epoch)))

def train(epoch,loader,args,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,scheduler_d,scheduler_g,start_time,model_attribute,
batch_size,vae,optim_vae,scheduler_vae,vae_alpha,vae_beta):
    discriminator.train()
    generator.train()
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data = (data.cuda().to(torch.float32) / 127.5 - 1) 
        target =  Variable(target.cuda())

        disc_loss=update_discriminator(disc_iters,args,Z_dim,optim_disc,optim_gen,discriminator,generator,data)
        gen_loss=update_generator(args,Z_dim,optim_disc,optim_gen,discriminator,generator)
        if model_attribute.dgm_type.has_vae:
            vae_loss,recons_loss,kld_loss=update_vae( data,target,vae,optim_vae,batch_size,vae_alpha,vae_beta)
        else:
            vae_loss=0
            recons_loss=0
            kld_loss=0

        if batch_idx % 100 == 0:
            tick_end_time = time.time()
            print(f"epoch:{epoch}, disc loss : {  disc_loss.item()}, gen loss: {  gen_loss.item()}, vae_loss:{vae_loss}, recons_loss:{recons_loss}, kld_loss:{kld_loss}, time {dnnlib.util.format_time(  tick_end_time - start_time):<12s} ") 
    scheduler_d.step()
    scheduler_g.step()
    if model_attribute.dgm_type.has_vae:
        scheduler_vae.step()


def evaluate(epoch,grid_z,generator,run_dir,discriminator,metrics,real_images,model_attribute,grid_size,vae_gan):
    discriminator.eval()
    generator.eval()
    rank=0
    device = torch.device('cuda', rank) 
    num_gpus=1
    evaluate_metrics(epoch,generator,discriminator,metrics,num_gpus,rank,device, run_dir,vae_gan )
    save_image( generator,grid_z, run_dir,epoch,grid_size,real_images,discriminator,model_attribute,vae_gan)
     
 
    

def evaluate_metrics(epoch,generator,discriminator,metrics,num_gpus,rank,device, run_dir,vae_gan,snapshot_pkl=None):
    if  (len(metrics) > 0):
        if rank == 0:
            print('Evaluating metrics...')
        total_result_dict=dict()
        
        for metric in metrics: 
            result_dict = metric_main.calc_metric(metric=metric, G=generator,
                 num_gpus=num_gpus, rank=rank, device=device,D=discriminator,vae_gan=vae_gan)
   
            if rank == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
       
            total_result_dict.update(result_dict.results)
         
  

def load_data(batch_size):
    dataset=load_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
            batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return loader,dataset

def load_model(Z_dim,model_type,model_attribute,lan_step_lr,lan_steps,batch_size,images ):
    # discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
    if model_type == 'resnet':
        discriminator = model_resnet.Discriminator().cuda()
        generator = model_resnet.Generator(Z_dim).cuda()
    else:
        discriminator = model.Discriminator(model_attribute.dgm_type.has_vae,Z_dim).cuda()
        generator = model.Generator(Z_dim).cuda()
        dealed_imgs=torch.from_numpy(images)
        dealed_imgs = (dealed_imgs.cuda().to(torch.float32) / 127.5 - 1) 
        morphing=  Morphing(lan_step_lr,lan_steps,batch_size,Z_dim,dealed_imgs)
        if model_attribute.dgm_type.has_vae:
            
            vae_gan=model.VaeGan(discriminator,generator,morphing) 
        else:
            vae_gan=model.Gan(discriminator,generator,morphing) 
    return generator,discriminator,vae_gan