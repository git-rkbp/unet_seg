import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.ndimage import zoom



from model import UNET

import matplotlib
matplotlib.use('TkAgg')  # Change to a different backend if necessary


# elastic deformation for data augmentation.
from scipy.ndimage import map_coordinates, gaussian_filter



def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid for indices
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Distort meshgrid indices
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    # Map coordinates from distorted indices to original image
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(shape), random_state
def aply_elastic_def_im_ms(image,mask,alpha, sigma):
    im_deformed, state = elastic_transform(image,  alpha=alpha, sigma=sigma)
    mk_deformed, state = elastic_transform(mask,  alpha=alpha, sigma=sigma, random_state=state)

    return im_deformed, mk_deformed


def load_images_and_masks(folder = "./data_n"):
    images = []
    masks = []


    folder_images = folder + "/images"
    folder_masks  = folder + "/masks"  

    # Count the number of image files
    num_files = len([name for name in os.listdir(folder_images) if name.startswith('img') and name.endswith('.npy')])


    for i in range(num_files):
        # Construct file paths for image and mask
        img_path = os.path.join(folder_images, f"img{i}.npy")
        msk_path = os.path.join(folder_masks, f"msk{i}.npy")

        # Load the image and mask and append them to the lists
        images.append(np.load(img_path))
        masks.append(np.load(msk_path))

    # Convert lists to NumPy arrays
    images_array = np.array(images)
    masks_array = np.array(masks)

    return images_array, masks_array
def plot_images(images_list):
    
    """
    Plots all the images in the given list.
    
    Args:
    - images_list (list): A list containing individual image arrays.
    """
    
    num_images = len(images_list)
    
    # Calculate number of rows and columns for the grid of images
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axarr = plt.subplots(rows, cols, figsize=(8, 8))
    
    # Ensure axarr is always a 2D array
    if rows == 1 and cols == 1:
        axarr = np.array([[axarr]])
    elif rows == 1 or cols == 1:
        axarr = axarr.reshape(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                axarr[i, j].imshow(images_list[idx], cmap='jet')
                axarr[i, j].axis('off')
            else:
                axarr[i, j].axis('off')  # Hide axes if there are no more images
                
    
    plt.tight_layout()
    plt.show()
class CustomDataset(Dataset):
    def __init__(self, images, masks, alpha, sigma,transform=True):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.alpha = alpha
        self.sigma = sigma


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        alpha = self.alpha
        sigma = self.sigma
        

        if self.transform:
            if torch.rand(1).item()  < 0.75:
                image, mask  = aply_elastic_def_im_ms(image, mask, alpha, sigma)
                # print("activated")


        image = np.expand_dims(image, axis=0)

        # Ensure that the mask has the correct shape [batch_size, height, width]
        if mask.shape[0] == 1:
            mask = mask[0]

        image = torch.from_numpy(image).float()
        mask  = torch.from_numpy(mask).long()

        return image, mask



if __name__ == '__main__':

    # getting the data 
    images, masks = load_images_and_masks()


    im = images[25]
    ms = masks[25]

    alpha = 1200
    sigma = 30



    dataset = CustomDataset(images, masks, alpha= alpha, sigma = sigma )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
    #     # print(f"Batch {batch_idx}")
        print("Data: ", np.shape(data))
        print("Labels: ", np.shape(labels))


        if batch_idx == 25:

            t_image  = data.squeeze().numpy()
            t_mask   = labels.squeeze().numpy() 
            plot_images([t_image, t_mask])
            break
            # print
        # Here you can process your data, feed it into a model, etc.





# im_d, ms_d = aply_elastic_def_im_ms(im, ms, alpha, sigma)


# plot_images([im,
#              ms,
#              im_d,
#              ms_d])












