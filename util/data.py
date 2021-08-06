
 
from torchvision import datasets, transforms
def load_dataset():
    dataset=datasets.CIFAR10('../../data/', train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor() ]))
    return dataset
 
