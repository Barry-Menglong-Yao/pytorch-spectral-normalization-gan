from enum import Enum,auto


class DgmType(Enum):
    VAE=(True)
    GAN=(False)
    GAN_VAE=(True)
    autoencoder=(False)

    def __init__(self,has_vae ):
        self.has_vae=has_vae

class ModelAttribute(Enum):
    SNGAN = (DgmType.GAN,  "training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5)
    SNGAN_VAE= (DgmType.GAN_VAE,  "training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5)
    def __init__(self, dgm_type, g_model_name,d_model_name,z_dim,model_name,disc_iters):
        self.dgm_type = dgm_type       # in kilograms
     
        self.d_model_name=d_model_name
        self.g_model_name=g_model_name
        self.z_dim=z_dim
        self.model_name=model_name
        self.disc_iters=disc_iters
     


 