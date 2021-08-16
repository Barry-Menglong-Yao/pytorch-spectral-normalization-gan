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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.tune.schedulers import PopulationBasedTraining
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--outdir', type=str, default='training_runs')
    parser.add_argument('--model_type', type=str, default='SNGAN_VAE')
    parser.add_argument('--vae_alpha', type=float, default=0.1)
    parser.add_argument('--vae_beta', type=float, default=0.1)
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--metrics', type=str,nargs='+', default=['fid50k_full_reconstruct','fid50k_full'])
    parser.add_argument('--lan_step_lr', type=float, default=0)
    parser.add_argument('--lan_steps', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--evaluate_interval',  type=int,
                        default=12) #5
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
    if args.mode  is  None or args.mode  !="hyper_search": 
        train_cifar(None,None, args  )
    else:
        hyper_search(args)

def hyper_search(args):
    scheduler = PopulationBasedTraining(
    perturbation_interval=5,
    hyperparam_mutations={
        "alpha":  tune.choice([0.000001,0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100]),
        "beta":   tune.choice([0.00000001,0.0000001,0.000001,0.00001, 0.0001, 0.001, 0.01,0.1]),
    })

     
    gpus_per_trial = 1
    num_samples=5
    tune_iter=6
    metric_name= "fid50k_full"
    cpus_per_trial=7
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[  "fid50k_full" , "fid50k_full_reconstruct", "training_iteration"],max_progress_rows=num_samples) 
    result = tune.run(
        partial(train_cifar ,args=args),
         
        scheduler=scheduler,
        verbose=1,
        stop={
            "training_iteration": tune_iter,
        },
        metric=metric_name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
        config={
            "alpha":  tune.choice([0.000001,0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100]),
            "beta":   tune.sample_from(lambda _: (0.1)**np.random.randint(0, 8)) ,
            
        })









    # config = {
    #     "alpha":  tune.choice([0.000001,0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100]),
    #     "beta":   tune.sample_from(lambda _: (0.1)**np.random.randint(0, 8)) 
    # }
    


    # scheduler = ASHAScheduler(
    #     metric= metric_name,
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=2,
    #     reduction_factor=2)         
                      
    # result = tune.run(
    #     partial(train_cifar ,args=args),
    #     resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     progress_reporter=reporter,
    #     checkpoint_at_end=False) 
        

    best_trial = result.get_best_trial(metric_name, "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final generation fid: {}".format(
        best_trial.last_result[ "fid50k_full"]))
    print("Best trial final reconstrction fid: {}".format(
        best_trial.last_result["fid50k_full_reconstruct"]))
    # print("Best trial final reconstruct_loss: {}".format(
    #     best_trial.last_result["reconstruct_loss"]))

def train_cifar(tuner_config, checkpoint_dir=None,args=None):

   

    update_config(args,tuner_config) 
    run_dir=make_running_dir(args.outdir,args.model_type,args.remark,args)
    if args.metrics is None:
        args.metrics = ['fid50k_full_reconstruct','fid50k_full']
    training_loop(args,run_dir,checkpoint_dir)

 
def update_config(args,tuner_config) :
    if tuner_config!=None:
        args.vae_alpha=tuner_config["alpha"]
        args.vae_beta=tuner_config["beta"]



if __name__ == "__main__":
    main()  
