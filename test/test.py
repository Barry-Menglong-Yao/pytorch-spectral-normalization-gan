import torch
from torch.distributions.uniform import Uniform


def test1():
    z=Uniform(-1., 1.).sample([3, 4]) 
    print(z)

test1()