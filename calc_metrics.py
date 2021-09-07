# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
from util.data import load_dataset
from util.image import   setup_snapshot_image_grid
from util.enums import ModelAttribute
import click
import json
import tempfile
import copy
import torch
from util import dnnlib
from util.trainer import   load_model
from util import constants
 
from util.metrics import metric_main
from util.metrics import metric_utils
from util.torch_utils import training_stats
from util.torch_utils import custom_ops
from util.torch_utils import misc
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir ):
    if args.verbose:
        print(f'Loading network from "{args.network_pkl}"...')
    model_attribute=ModelAttribute[args.model_type]
    batch_size=64
    dataset=load_dataset()
    args.run_dir ="testing_runs/metric"
    _, real_images, _ = setup_snapshot_image_grid(training_set=dataset,batch_size=batch_size)
    real_images =  torch.from_numpy(real_images).cuda()
    generator,discriminator,vae_gan=load_model(constants.Z_dim,args.model_type,model_attribute,args.lan_step_lr,args.lan_steps,batch_size,real_images)
    if args.epoch>=0:
        gen_pkl=args.network_pkl+"/gen_"+str(args.epoch)
        dis_pkl=args.network_pkl+"/disc_"+str(args.epoch)
    else:
        gen_pkl=args.network_pkl+"/gen"
        dis_pkl=args.network_pkl+"/disc"
    generator.load_state_dict(torch.load(gen_pkl))
    discriminator.load_state_dict(torch.load(dis_pkl),strict=False)

    dnnlib.util.Logger(should_flush=True)
     
    device = torch.device('cuda', rank) 
     
    G = copy.deepcopy(generator).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(discriminator).eval().requires_grad_(False).to(device)
    total_result_dict=dict()
    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, G=G, 
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress,D=D,vae_gan=vae_gan)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()
        total_result_dict.update(result_dict.results)
    if args.mode=="hyper_search":
             
            tune.report(**total_result_dict)
    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if   isinstance(value, list):
            return value
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
# @click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH',default="/home/barry/workspace/code/referredModels/pytorch-spectral-normalization-gan/training_runs/00043-SNGAN_VAE-0.0-0.000-/checkpoint", required=True)
# # @click.option('--epoch', help=' epoch', type=int, default=828, metavar='INT' )
# @click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full,fid50k_full_reconstruct' , show_default=True)#,fid50k_full_reconstruct
# @click.option('--model_type', help=' ',default='SNGAN_VAE', type=click.Choice(['SNGAN','SNGAN_VAE' ]))
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH',default="/home/barry/workspace/code/referredModels/pytorch-spectral-normalization-gan/training_runs/00079-SNGAN-0.100-0.100000-/checkpoint", required=True)
@click.option('--epoch', help=' epoch', type=int, default=-1, metavar='INT' )
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full' , show_default=True)#,fid50k_full_reconstruct
@click.option('--model_type', help=' ',default='SNGAN', type=click.Choice(['SNGAN','SNGAN_VAE' ]))

@click.option('--data', help='Dataset to evasluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--lan_steps', help=' epoch', type=int, default=1, metavar='INT' )
@click.option('--lan_step_lr', help='lan_step_lr', type=float, default=0.1 )
@click.option('--mode', help=' ',default='test', type=click.Choice(['test','hyper_search' ]))
def main(ctx, network_pkl,epoch, metrics, data,  gpus, verbose,model_type,lan_steps,lan_step_lr,mode):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip   \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    if mode!="hyper_search":
        calc_metric(None, network_pkl,epoch, metrics,    gpus, verbose,model_type,lan_steps,lan_step_lr,mode)
    else:
        hyper_search(ctx, network_pkl,epoch, metrics, data,  gpus, verbose,model_type,lan_steps,lan_step_lr,mode)



def hyper_search(ctx, network_pkl,epoch, metrics, data,  gpus, verbose,model_type,lan_steps,lan_step_lr,mode):
    config = {
        "lan_steps":  tune.choice([10,15,20,50,100]),#10,15,20,50,100
        "lan_step_lr":tune.choice([0.01,0.1,0.3,0.5,0.6,0.7]) #  tune.loguniform(1e-2, 1)#  tune.choice([0.00001,0.0001,0.001 ])
    }
    gpus_per_trial = 1
    num_samples=30
    max_num_epochs=1
    metric_name= "fid50k_full"
    cpus_per_trial=8

    scheduler = ASHAScheduler(
        metric= metric_name,
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)         
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[  "fid50k_full" , "training_iteration"],max_progress_rows=num_samples)                   # "fid50k_full_reconstruct",
    result = tune.run(
        partial(calc_metric , network_pkl=network_pkl,epoch=epoch, metrics=metrics,    gpus=gpus, verbose=verbose,model_type=model_type,lan_steps=lan_steps,lan_step_lr=lan_step_lr,mode=mode), #
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
    # print("Best trial final reconstrction fid: {}".format(
    #     best_trial.last_result["fid50k_full_reconstruct"]))
 



def update_config(lan_steps,lan_step_lr,tuner_config) :
    if tuner_config!=None:
        return tuner_config["lan_steps"],tuner_config["lan_step_lr"]
    else:
        return lan_steps,lan_step_lr

 
def calc_metric(tuner_config,network_pkl=None,epoch=None, metrics=None,   gpus=None, verbose=None,model_type=None,lan_steps=None,lan_step_lr=None,mode=None):#  

    lan_steps,lan_step_lr=update_config(lan_steps,lan_step_lr,tuner_config) 
 

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose,model_type=model_type,lan_steps=lan_steps,lan_step_lr=lan_step_lr,
    mode=mode,epoch=epoch)
 
     
  
    subprocess_fn(rank=0, args=args, temp_dir=None ) 
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
