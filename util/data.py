 
import torch
import numpy as np
import PIL.Image
from torchvision import datasets, transforms
import os
from util import dnnlib
def load_dataset():
    data="/home/barry/workspace/code/referredModels/stylegan2-ada-pytorch/datasets/cifar10.zip"
    training_set_kwargs = dnnlib.EasyDict(class_name='util.dataset.ImageFolderDataset', path=data, use_labels=False, max_size=50000, xflip=False)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) 
    # training_set=datasets.CIFAR10('../../data/', train=True, download=False,
    #         transform=transforms.Compose([
    #             transforms.ToTensor() ]))
    return training_set
 



