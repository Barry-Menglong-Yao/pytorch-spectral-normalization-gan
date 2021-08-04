
 
from torchvision import datasets, transforms
def load_dataset():
    dataset=datasets.CIFAR10('../../data/', train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    return dataset
 
