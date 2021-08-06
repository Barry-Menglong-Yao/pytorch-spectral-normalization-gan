import argparse
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

    

def train(epoch,loader,args,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,scheduler_d,scheduler_g,start_time,model_attribute,
batch_size,vae,optim_vae,scheduler_vae,vae_alpha,vae_beta):
    discriminator.train()
    generator.train()
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

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

def save_generated_img(epoch,fixed_z,generator,run_dir):
    samples = generator(fixed_z).cpu().data.numpy()[:64]
    img_name=f'img/fake_{str(epoch).zfill(3)}.png' 
    save_img(epoch,samples,run_dir,img_name)

def save_reconstructed_img(epoch,generator,discriminator,run_dir,sampled_imgs):
    
    reconstructed_img=reconstruct(sampled_imgs ,  generator,discriminator,None    )
    reconstructed_img=reconstructed_img.cpu().data.numpy()[:64]
    img_name=f'img/reconstruct_{str(epoch).zfill(3)}.png' 
    save_img(epoch,reconstructed_img,run_dir,img_name  )



def save_img(epoch,imgs ,run_dir,img_name):
     


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    
    plt.savefig(os.path.join(run_dir,img_name), bbox_inches='tight')
    plt.close(fig)

def evaluate(epoch,fixed_z,generator,run_dir,discriminator,metrics,sampled_imgs,model_attribute):
    discriminator.eval()
    generator.eval()
    rank=0
    device = torch.device('cuda', rank) 
    num_gpus=1
    evaluate_metrics(epoch,generator,discriminator,metrics,num_gpus,rank,device, run_dir )

    save_generated_img(epoch,fixed_z,generator,run_dir)

    if model_attribute.dgm_type.has_vae:
        save_reconstructed_img(epoch,generator,discriminator,run_dir,sampled_imgs)
    

def evaluate_metrics(epoch,generator,discriminator,metrics,num_gpus,rank,device, run_dir,snapshot_pkl=None):
    if  (len(metrics) > 0):
        if rank == 0:
            print('Evaluating metrics...')
        total_result_dict=dict()
        
        for metric in metrics: 
            result_dict = metric_main.calc_metric(metric=metric, G=generator,
                 num_gpus=num_gpus, rank=rank, device=device,D=discriminator)
   
            if rank == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
       
            total_result_dict.update(result_dict.results)
         
  

def load_data(batch_size):
    dataset=load_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader,dataset

def load_model(Z_dim,model_type,model_attribute):
    # discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
    if model_type == 'resnet':
        discriminator = model_resnet.Discriminator().cuda()
        generator = model_resnet.Generator(Z_dim).cuda()
    else:
        discriminator = model.Discriminator(model_attribute.dgm_type.has_vae,Z_dim).cuda()
        generator = model.Generator(Z_dim).cuda()
        if model_attribute.dgm_type.has_vae:
            vae=model.VaeGan(discriminator,generator) 
        else:
            vae=None
    return generator,discriminator,vae