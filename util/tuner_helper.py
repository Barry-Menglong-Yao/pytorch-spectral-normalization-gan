
import os
import torch
from ray import tune
def resume_from_checkpoint(checkpoint_dir,  netVae,optimizerG,optimizerD,optimizerVae,tuner_step,generator,discriminator):
    if checkpoint_dir is not None:
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        netVae.load_state_dict(checkpoint["netVAE"])
        generator=netVae.generator
        discriminator=netVae.discriminator
        optimizerD.load_state_dict(checkpoint["optimD"])
        optimizerG.load_state_dict(checkpoint["optimG"])
        optimizerVae.load_state_dict(checkpoint["optimVae"])
        tuner_step = checkpoint["step"]

      
    return tuner_step,generator,discriminator

def save_checkpoint(step,netVae ,optimizerG,optimizerD,optimizerVae):
    with tune.checkpoint_dir(step=step) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save({
            "netVAE": netVae.state_dict(),
             
            "optimD": optimizerD.state_dict(),
            "optimG": optimizerG.state_dict(),
            "optimVae":optimizerVae.state_dict(),
            "step": step,
        }, path)

