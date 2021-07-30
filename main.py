import argparse
from trainer import evaluate, train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='hinge')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--model', type=str, default='resnet')

    args = parser.parse_args()
    return args




def load_data(args):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return loader

def load_model(Z_dim,args):
    # discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
    if args.model == 'resnet':
        discriminator = model_resnet.Discriminator().cuda()
        generator = model_resnet.Generator(Z_dim).cuda()
    else:
        discriminator = model.Discriminator().cuda()
        generator = model.Generator(Z_dim).cuda()
    return generator,discriminator


def main():
    args=parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    Z_dim = 128
    #number of updates to discriminator for every update to generator 
    disc_iters = 5
    loader=load_data(args)
    generator,discriminator=load_model(Z_dim,args)
    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
    
    
    fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
    for epoch in range(2000):
        train(epoch,loader,args,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,scheduler_d,scheduler_g)
        evaluate(epoch,fixed_z,generator)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))

if __name__ == "__main__":
    main()  
