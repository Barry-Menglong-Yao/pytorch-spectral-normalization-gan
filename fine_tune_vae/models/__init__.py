from .base import *
from .vanilla_vae import * 
from model.model import VaeGan

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE 
vae_models = { 
              'VanillaVAE':VanillaVAE,
              'VaeGan':VaeGan }
