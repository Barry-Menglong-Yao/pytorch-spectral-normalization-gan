import math
from util.check_param import ParamChecker
from util.enums import ModelAttribute
from util.image import export_sample_images
from util.trainer import evaluate, evaluate_metrics
from util.data import load_dataset
import torch
from torch import optim
from fine_tune_vae.models import BaseVAE
 
from fine_tune_vae.utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import os

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict,Z_dim ,evaluate_interval) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.first_step=True
        self.Z_dim=Z_dim
        # self.param_checker=ParamChecker(self.model.generator)
        
 
        self.evaluate_interval=evaluate_interval
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input, **kwargs) :
        self.model.generator.eval()
        return self.model(input,None, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        real_img=convert_256_to_1(real_img, self.params['dataset']  )
        reconstructed_img, mu,log_var, = self.forward(real_img, labels = labels)
        vae_loss,recons_loss,kld_loss = self.model.loss_function(reconstructed_img,real_img,mu,log_var,
                                              kld_weight = self.params['batch_size']/ self.num_train_imgs )
        loss={'loss': vae_loss , 'Reconstruction_Loss':recons_loss , 'KLD':-kld_loss }
        loss_log={'loss': vae_loss.item(), 'Reconstruction_Loss':recons_loss.item(), 'KLD':-kld_loss.item()}
        self.logger.experiment.log(loss_log)

        return loss
    # def on_after_backward(self):
    #     self.param_checker.print_grad_after_backward(self.model.generator)

    # def on_before_zero_grad(self,optimizer):
     
    #     self.param_checker.compare_params( self.model.generator)

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        if self.first_step:
            run_dir= f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            os.makedirs(os.path.join(run_dir,'img'))
            os.makedirs(os.path.join(run_dir,'checkpoint'))
            
            grid_z, grid_size,real_images=export_sample_images(load_dataset(), run_dir,  torch.device('cuda'), self.Z_dim,self.params['batch_size'])
            self.grid_z=grid_z
            self.grid_size=grid_size
            self.real_images=real_images
            
            self.first_step=False
        real_img, labels = batch
        self.curr_device = real_img.device
        real_img=convert_256_to_1(real_img, self.params['dataset']  )
        reconstructed_img,mu,log_var = self.forward(real_img, labels = labels)
        vae_loss,recons_loss,kld_loss   = self.model.loss_function(reconstructed_img,real_img,mu,log_var,
                                            kld_weight = self.params['batch_size']/ self.num_val_imgs )
        loss={'loss': vae_loss , 'Reconstruction_Loss':recons_loss , 'KLD':-kld_loss }
 
     
        return loss




    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        metrics= ['fid50k_full_reconstruct','fid50k_full']
        run_dir= f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        
        model_attribute=ModelAttribute["SNGAN_VAE"]
        
        if self.current_epoch%self.evaluate_interval==0: 
            evaluate(self.current_epoch,self.grid_z,self.model.generator,run_dir,self.model.discriminator,metrics,self.real_images,model_attribute,
            self.grid_size,self.model,"train" )
 
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def sample_images(self):
    #     # Get sample reconstruction image
    #     test_input, test_label = next(iter(self.sample_dataloader))
    #     test_input=convert_256_to_1(test_input, self.params['dataset']  )
    #     test_input = test_input.to(self.curr_device)
    #     test_label = test_label.to(self.curr_device)
    #     recons = self.model(test_input, labels = test_label)[0]
    #     vutils.save_image(recons.data,
    #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #                       f"recons_{self.logger.name}_{self.current_epoch}.png",
    #                       normalize=True,
    #                       nrow=12)

    #     # vutils.save_image(test_input.data,
    #     #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #     #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
    #     #                   normalize=True,
    #     #                   nrow=12)

    #     try:
    #         samples = self.model.sample(144,
    #                                     self.curr_device,
    #                                     labels = test_label)
    #         vutils.save_image(samples.cpu().data,
    #                           f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #                           f"{self.logger.name}_{self.current_epoch}.png",
    #                           normalize=True,
    #                           nrow=12)
    #     except:
    #         pass


    #     del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []
        params_to_update=filter_params(self.model)
        optimizer = optim.Adam(params_to_update,
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        else:
            dataset=load_dataset()
            # raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True,
                          num_workers=0)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset=CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False)
            
        else:
            dataset=load_dataset()
            # raise ValueError('Undefined dataset type')
        self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = False,
                                                 drop_last=True,num_workers=0)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform = transforms.Compose([ 
                                            transforms.ToTensor() ])
            # raise ValueError('Undefined dataset type')
        return transform



def convert_256_to_1(real_img,data_type):
    if data_type== 'cifar10':
        processed_iamges=(real_img.to(torch.float32) / 127.5 - 1)
        return processed_iamges
    else:
        return real_img
    
def filter_params( model):
    params_to_update = []
    print("\t updated params")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    return params_to_update