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

# from utils import *
from model import UNET

import matplotlib
matplotlib.use('TkAgg')  # Change to a different backend if necessary

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
                axarr[i, j].imshow(images_list[idx], cmap='viridis')
                axarr[i, j].axis('off')
            else:
                axarr[i, j].axis('off')  # Hide axes if there are no more images
                
    
    plt.tight_layout()
    plt.show()
def plot_images_1(images_list, class_colors):
    """
    Plots all the images in the given list with consistent colors for each class.
    
    Args:
    - images_list (list): A list containing individual image arrays.
    - class_colors (dict): A dictionary mapping class labels to colors.
    """
    
    num_images = len(images_list)
    
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axarr = plt.subplots(rows, cols, figsize=(8, 8))

    if rows == 1 and cols == 1:
        axarr = np.array([[axarr]])
    elif rows == 1 or cols == 1:
        axarr = axarr.reshape(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                # Map each pixel value to the corresponding color
                colored_image = np.array([class_colors[value] for value in images_list[idx].flatten()]).reshape(images_list[idx].shape + (3,))
                axarr[i, j].imshow(colored_image)
                axarr[i, j].axis('off')
            else:
                axarr[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        mask = self.masks[idx]

        image = np.expand_dims(image, axis=0)

        # Ensure that the mask has the correct shape [batch_size, height, width]
        if mask.shape[0] == 1:
            mask = mask[0]


        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['mask'] = torch.from_numpy(sample['mask']).long()

        return sample['image'], sample['mask']
class BasicDataset_V2(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, scale: float = 1.0):
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.images = images
        self.masks = masks

        # Ensure the number of images and masks are the same
        assert len(images) == len(masks), 'Number of images and masks must be equal'

        # Determine unique mask values
        self.mask_values = list(sorted(np.unique(np.concatenate(masks), axis=None).tolist()))

    def __len__(self):
        return len(self.images)

    @staticmethod
    def preprocess(mask_values, np_img, scale, is_mask):
        # Determine new width and height
        h, w = np_img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        # Resize the image or mask
        if np_img.ndim == 2:  # Grayscale image or mask
            np_img = zoom(np_img, (newH / h, newW / w), order=0 if is_mask else 3)
        else:  # Color image
            np_img = zoom(np_img, (newH / h, newW / w, 1), order=3)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[np_img == v] = i
            return mask
        else:
            if np_img.ndim == 2:
                np_img = np_img[np.newaxis, ...]  # Add channel dimension
            else:
                np_img = np_img.transpose((2, 0, 1))  # Change HWC to CHW

            if (np_img > 1).any():
                np_img = np_img / 255.0  # Normalize

            return np_img

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # Preprocess the image and mask
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return  torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask.copy()).long().contiguous()
def do_inference(model, dataloader, idx, DEVICE = 'cuda:0'):
    model.eval()

    for i, (input, target) in enumerate(dataloader):
        if i == idx:
            inference_image = input

    with torch.no_grad():
        X = inference_image
        X = X.to(DEVICE)
        predictions = model(X) 


        predictions = torch.nn.functional.softmax(predictions, dim=1)
        pred_labels = torch.argmax(predictions, dim=1) 
        pred_labels = pred_labels.cpu().numpy()
        pred_labels = np.squeeze (pred_labels)

    mask = pred_labels 
    return mask


# checking the cuda 
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')



model_path = "./save/model_checkpoint_epoch_18.pth"


# getting the data 
images, masks = load_images_and_masks()




dataset = CustomDataset(images, masks)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

model = UNET(in_channels=1, classes=5).to(DEVICE).train()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


val = 20


im = do_inference(model, dataloader, val)



plot_images([im, masks[val],images[val]])

# plt.imshow(pred_labels)
# plt.show()






# print(unet)


