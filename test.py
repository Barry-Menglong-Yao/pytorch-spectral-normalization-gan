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

def test_board():
    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('../../../data/mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)

    for i in range(10):
        z = torch.randn([64, 16] ) 
        z_min=(torch.min(z).item())
        writer.add_scalar("z_min",z_min,global_step=i)
        
        z_max=(torch.max(z).item())
        writer.add_scalar("z_max",z_max,global_step=i)
        writer.close()
    writer.close()
test_board()

