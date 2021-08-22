from util.data import load_dataset
from util.image import export_sample_images
from model.morph import Morphing
from util.enums import ModelAttribute
from util.trainer import load_model
from torch.utils.data import dataset
import yaml
import argparse
import numpy as np
import torch
from fine_tune_vae.models import *
from fine_tune_vae.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

def train_vae():
    
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='fine_tune_vae/configs/vae.yaml')
    parser.add_argument('--epoch',  
                        type=int,
                        default=-1)
                        # default=336)
    parser.add_argument('--network_pkl',  
                        default="training_runs/00074-SNGAN-0.100-0.100000-/checkpoint")
                        # default="training_runs/00045-SNGAN-0.0-0.000-/checkpoint")
    parser.add_argument('--resume',  
                        default="y")
    parser.add_argument('--evaluate_interval',  type=int,
                        default=15) #5
    parser.add_argument('--fine_tune_module',  
                        default="d_and_e")   #e, g_d_e        
    parser.add_argument('--model_type',  
                    default="SNGAN_VAE")         

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model=load_vae_model(args,config)
    
    experiment = VAEXperiment(model,
                            config['exp_params'], config['model_params']['latent_dim'],args.evaluate_interval,args.fine_tune_module)

    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                    min_epochs =1,
                    logger=tt_logger,
                    log_save_interval=100,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    early_stop_callback = False,
                    **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)


def load_vae_model(args,config):
    if config['model_params']['name']=="VaeGan":
        model_attribute=ModelAttribute[args.model_type]
        
        generator,discriminator,model = load_model(config['model_params']['latent_dim'],"dcgan",model_attribute,0,0,config['exp_params']['batch_size'],None )
        if args.resume=="y":
            if args.epoch>=0:
                gen_pkl=args.network_pkl+"/gen_"+str(args.epoch)
                dis_pkl=args.network_pkl+"/disc_"+str(args.epoch)
            else:
                gen_pkl=args.network_pkl+"/gen"
                dis_pkl=args.network_pkl+"/disc"
            generator.load_state_dict(torch.load(gen_pkl),strict=False)
            discriminator.load_state_dict(torch.load(dis_pkl),strict=False)
            if args.fine_tune_module=="d_and_e":
                generator.requires_grad_(False)
            elif args.fine_tune_module=="e":
                generator.requires_grad_(False)
                for name ,child in (discriminator.named_children()):
                    if name!='fc_mu' and name!='fc_var' :
                        child.requires_grad_(False)
               
                  
    else:
        model = vae_models[config['model_params']['name']](**config['model_params'])
    return model

train_vae()