from enum import Enum,auto


class DgmType(Enum):
    VAE=(True)
    GAN=(False)
    GAN_VAE=(True)
    autoencoder=(False)

    def __init__(self,has_vae ):
        self.has_vae=has_vae

class ModelAttribute(Enum):
    SNGAN = (DgmType.GAN,  "training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5," ",None)
    SNGAN_VAE= (DgmType.GAN_VAE,  "training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5," ",None)
    UNet= (DgmType.GAN_VAE,  "training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5," ",None)
    UNet_SNGAN_VAE= (DgmType.GAN_VAE,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"conv",["1","2","3","4"])
    UNet_SNGAN_VAE_fc= (DgmType.GAN_VAE,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"fc",["1","4"])
    UNet_SNGAN_VAE_cat= (DgmType.GAN_VAE,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"cat",["1"])
    UNet_SNGAN_VAE_broadcast= (DgmType.GAN_VAE,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"conv_broadcast",["1","2","3","4"])
    UNet_SNGAN_VAE_pool= (DgmType.GAN_VAE,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"pool",["1","2","3","4"])
    def __init__(self, dgm_type, g_model_name,d_model_name,z_dim,model_name,disc_iters,inject_type,inject_layer_list):
        self.dgm_type = dgm_type       # in kilograms
     
        self.d_model_name=d_model_name
        self.g_model_name=g_model_name
        self.z_dim=z_dim
        self.model_name=model_name
        self.disc_iters=disc_iters
        self.inject_type=inject_type
        self.inject_layer_list=inject_layer_list
     


 