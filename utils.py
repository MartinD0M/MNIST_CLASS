import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from math import sqrt,  floor


def show_batch(batched_images : torch.Tensor,
               labels : torch.Tensor = None ):
    '''
    Displays passed images with passed labels
    '''
    n_images = batched_images.shape[0]
    side = floor(sqrt(n_images))
    fig, axs = plt.subplots(side,side)
    fig.tight_layout()
    
    for img_idx in range(side**2):
        img = batched_images[img_idx,:,:,:].squeeze(dim=0)
        axs[img_idx//side][img_idx%side].imshow(to_pil_image(img), cmap = 'gray', vmin= 0, vmax = 255)
        if labels is not None:
            axs[img_idx//side][img_idx%side].set_xlabel(labels[img_idx])
    plt.show()
