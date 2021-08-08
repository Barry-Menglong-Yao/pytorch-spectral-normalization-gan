
from util import dnnlib
from util.metrics.metric_utils import reconstruct
from util.metrics import metric_main
import torch
 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
import PIL.Image

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        
def save_image( G,grid_z, run_dir,epoch,grid_size,real_images,D,model_attribute,vae_gan):
    images = G(grid_z).cpu().data.numpy()[:64]
 
    save_image_grid(images, os.path.join(run_dir, f'img/fakes{epoch:06d}.png'), drange=[-1,1], grid_size=grid_size)
    if model_attribute.dgm_type.has_vae:
        grid_reconstructed_images=reconstruct_grid(real_images,G,D,vae_gan )
        save_image_grid(grid_reconstructed_images, os.path.join(run_dir, f'img/reconstruct{epoch:06d}.png'),drange=[0,255], grid_size=grid_size)



def reconstruct_grid(grid_real_images,G,D,vae_gan ):
    # grid_reconstructed_images=[]
    # for real_images  in  grid_real_images  :
        
        
    grid_reconstructed_images=reconstruct(grid_real_images ,  G,D ,None,vae_gan )
         
        # grid_reconstructed_images.append(reconstructed_img.cpu())
    grid_reconstructed_images= grid_reconstructed_images.cpu().numpy()
    return grid_reconstructed_images












def setup_snapshot_image_grid(training_set, batch_size, random_seed=0):
    rnd = np.random.RandomState(random_seed)
     
    gh=8
    gw=batch_size//8
      
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
 
    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    
def export_sample_images(training_set, run_dir, device, Z_dim,batch_size):
    grid_size = None
    grid_z = None
    real_images = None
  
    print('Exporting sample images...')
    grid_size, real_images, _ = setup_snapshot_image_grid(training_set=training_set,batch_size=batch_size)
    save_image_grid(real_images, os.path.join(run_dir, 'img/reals.png'), drange=[0,255], grid_size=grid_size)
    grid_z = torch.randn([real_images.shape[0], Z_dim], device=device) 
    real_images =  torch.from_numpy(real_images).to(device) 
    return grid_z, grid_size,real_images










# def save_generated_img(epoch,fixed_z,generator,run_dir):
#     samples = generator(fixed_z).cpu().data.numpy()[:64]
#     img_name=f'img/fake_{str(epoch).zfill(3)}.png' 
#     save_img(epoch,samples,run_dir,img_name)

# def save_reconstructed_img(epoch,generator,discriminator,run_dir,sampled_imgs):
    
#     reconstructed_img=reconstruct(sampled_imgs ,  generator,discriminator,None    )
#     reconstructed_img=reconstructed_img.cpu().data.numpy()[:64]
#     img_name=f'img/reconstruct_{str(epoch).zfill(3)}.png' 
#     save_img(epoch,reconstructed_img,run_dir,img_name  )



# def save_img(epoch,imgs ,run_dir,img_name):
     


#     fig = plt.figure(figsize=(8, 8))
#     gs = gridspec.GridSpec(8, 8)
#     gs.update(wspace=0.05, hspace=0.05)

#     for i, sample in enumerate(imgs):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    
#     plt.savefig(os.path.join(run_dir,img_name), bbox_inches='tight')
#     plt.close(fig)