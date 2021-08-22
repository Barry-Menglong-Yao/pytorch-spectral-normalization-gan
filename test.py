import torch
from torch.distributions.uniform import Uniform


def test1():
    z=Uniform(-1., 1.).sample([3, 4]) 
    print(z)

 
from model  import mmd
 

# print_tensor(tf.range(10))
def test_kernal():
    torch.set_printoptions(threshold=10000)
    kernel = getattr(mmd, '_rbf_kernel'  )
 
   
 
    d_g=torch.range(0,7,dtype=torch.float32)
    d_i=torch.range(2,9,dtype=torch.float32)
    d_g=d_g.cuda()
    d_i=d_i.cuda()
    d_g=torch.reshape(d_g,[4,2])
    d_i=torch.reshape(d_i,[4,2])
    _, kxy, _, _, = kernel(d_g, d_i)
    with open("doo.txt","w") as f:
        print(kxy,file=f)
        print(d_g,file=f)
        print(d_i,file=f)



test_kernal()

