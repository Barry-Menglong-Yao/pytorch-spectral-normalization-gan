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
from util.image import export_sample_images, setup_snapshot_image_grid
from util.enums import ModelAttribute
import click
import json
import tempfile
import copy
import torch
from util import dnnlib
from util.trainer import load_data, load_model
from util import constants
 
from util.metrics import metric_main
from util.metrics import metric_utils
from util.torch_utils import training_stats
from util.torch_utils import custom_ops
from util.torch_utils import misc

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir,generator,discriminator,vae_gan):
    dnnlib.util.Logger(should_flush=True)
    
    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

 
    device = torch.device('cuda', rank) 
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(generator).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(discriminator).eval().requires_grad_(False).to(device)

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

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--epoch', help=' epoch', type=int, default=1, metavar='INT' )
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full,fid50k_full_reconstruct' , show_default=True)
@click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')

@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--model_type', help=' ',default='SNGAN_VAE', type=click.Choice(['SNGAN','SNGAN_VAE' ]))
@click.option('--lan_steps', help=' epoch', type=int, default=10, metavar='INT' )
@click.option('--lan_step_lr', help='lan_step_lr', type=float, default=0.1)
def calc_metrics(ctx, network_pkl,epoch, metrics, data,  gpus, verbose,model_type,lan_steps,lan_step_lr):
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
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose,model_type=model_type,lan_steps=lan_steps,lan_step_lr=lan_step_lr)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    model_attribute=ModelAttribute[args.model_type]
    batch_size=64
    dataset=load_dataset()
    args.run_dir ="testing_runs/metric"
    _, real_images, _ = setup_snapshot_image_grid(training_set=dataset,batch_size=batch_size)
    real_images =  torch.from_numpy(real_images).cuda()
    generator,discriminator,vae_gan=load_model(constants.Z_dim,None,model_attribute,args.lan_step_lr,args.lan_steps,batch_size,real_images)
    generator.load_state_dict(torch.load(network_pkl+"/gen_"+str(epoch)))
    discriminator.load_state_dict(torch.load(network_pkl+"/disc_"+str(epoch)))
 
    
 
    
    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir,generator=generator,discriminator=discriminator,vae_gan=vae_gan)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir,generator,discriminator,vae_gan), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
