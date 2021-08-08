import argparse
from util.image import export_sample_images 
from util.enums import ModelAttribute
from util.trainer import evaluate, load_model, load_optim,   train, load_data, training_loop
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
    parser.add_argument('--lan_step_lr', type=float, default=0.1)
    parser.add_argument('--lan_steps', type=int, default=10)
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
    if config_kwargs['mode'] is  None or config_kwargs['mode']  !="hyper_search": 
        train_cifar(None, ctx, outdir, dry_run, config_kwargs  )
    else:
        hyper_search()

def hyper_search():
    config = {
        "alpha":  tune.loguniform(5e-1, 200 ),
        "beta":   tune.loguniform(1e-3, 1) 
    }
    gpus_per_trial = 0.5
    num_samples=10
    max_num_epochs=6
    metric_name= "fid50k_full_reconstruct"
    cpus_per_trial=4

    scheduler = ASHAScheduler(
        metric= metric_name,
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)         
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[ "reconstruct_loss", "fid50k_full" , "fid50k_full_reconstruct", "training_iteration"])                   
    result = tune.run(
        partial(train_cifar ,ctx=ctx,outdir=outdir,dry_run=dry_run,config_kwargs=config_kwargs),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False) 
        

    best_trial = result.get_best_trial(metric_name, "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final generation fid: {}".format(
        best_trial.last_result[ "fid50k_full"]))
    print("Best trial final reconstrction fid: {}".format(
        best_trial.last_result["fid50k_full_reconstruct"]))
    print("Best trial final reconstruct_loss: {}".format(
        best_trial.last_result["reconstruct_loss"]))

def train_cifar():
    args=parse_args()
    run_dir=make_running_dir(args.outdir,args.model_type,args.remark,args)
    if args.metrics is None:
        args.metrics = ['fid50k_full_reconstruct','fid50k_full']
 
    training_loop(args,run_dir)

 




if __name__ == "__main__":
    main()  
