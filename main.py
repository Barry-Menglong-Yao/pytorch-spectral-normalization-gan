import argparse
from util.image import export_sample_images 
from util.enums import ModelAttribute
from util.trainer import evaluate, load_model, load_optim,   train, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import model_resnet 
from model import  model 
from util import constants
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import time
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--outdir', type=str, default='training_runs')
    parser.add_argument('--model_type', type=str, default='SNGAN_VAE')
    parser.add_argument('--vae_alpha', type=float, default=100)
    parser.add_argument('--vae_beta', type=float, default=0.01)
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--metrics', type=str,nargs='+', default=['fid50k_full_reconstruct','fid50k_full'])
    args = parser.parse_args()
    return args




def make_running_dir(outdir,model_type,remark,args):
    run_desc=model_type
    run_desc+=f'-{args.vae_alpha:.3f}-{args.vae_beta:.6f}'
    if remark!=None:
        run_desc+="-"+remark
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists( run_dir)
    os.makedirs( run_dir)
    os.makedirs(os.path.join(run_dir,'img'))
    os.makedirs(os.path.join(run_dir,'checkpoint'))
    return run_dir




def main():
    args=parse_args()
    run_dir=make_running_dir(args.outdir,args.model_type,args.remark,args)
    if args.metrics is None:
        args.metrics = ['fid50k_full_reconstruct','fid50k_full']
 
    Z_dim = constants.Z_dim
    #number of updates to discriminator for every update to generator 
    disc_iters = 5
    model_attribute=ModelAttribute[args.model_type]
    loader,dataset=load_data(args.batch_size)
    generator,discriminator,vae=load_model(Z_dim,args.model_type,model_attribute)
    optim_gen,optim_disc,optim_vae,scheduler_g,scheduler_d,scheduler_vae=load_optim(args,discriminator,generator,vae,model_attribute)
    
    start_time = time.time()
    grid_z, grid_size,real_images=export_sample_images(dataset, run_dir,  torch.device('cuda'), Z_dim,args.batch_size)
    # fixed_z, sampled_imgs=sample_img_and_z(args,Z_dim,dataset,run_dir)
    for epoch in range(2000):
        train(epoch,loader,args,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,scheduler_d,scheduler_g,start_time,
        model_attribute,args.batch_size,vae,optim_vae,scheduler_vae,args.vae_alpha,args.vae_beta)
        if epoch%12==0 :
            evaluate(epoch,grid_z,generator,run_dir,discriminator,args.metrics,real_images,model_attribute,grid_size)
            torch.save(discriminator.state_dict(), os.path.join(run_dir, 'checkpoint/disc_{}'.format(epoch)))
            torch.save(generator.state_dict(), os.path.join(run_dir, 'checkpoint/gen_{}'.format(epoch)))


# def sample_img_and_z(args,Z_dim,dataset,run_dir ):
    


    # fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
    
    # all_indices = list(range(len(dataset)))
    # rnd = np.random.RandomState(0)
    # rnd.shuffle(all_indices)
    # grid_indices = [all_indices[i % len(all_indices)] for i in range(64)]
    # images, labels = zip(*[dataset[i] for i in grid_indices])
    # sampled_imgs=torch.stack(images).cuda()
    # saved_imgs=sampled_imgs.cpu().data.numpy()[:64]
    # img_name=f'img/real.png' 
    # save_img(0,saved_imgs,run_dir,img_name  )
    # return fixed_z, sampled_imgs




if __name__ == "__main__":
    main()  
