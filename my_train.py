import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.ndimage import zoom


# data augmentation
from my_data_augmentation import elastic_transform,aply_elastic_def_im_ms
from scipy.ndimage import map_coordinates, gaussian_filter

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
                axarr[i, j].imshow(images_list[idx], cmap='gray')
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

# code functions 
# training
def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    model.train()


    total_loss = 0
    total_samples = 0

    data = tqdm(data)
    
    for batch_idx, (data, labels) in enumerate(data):
        X, y = data.to(device), labels.to(device)
        preds = model(X)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)  # Multiply the loss by batch size
        total_samples += X.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def validate_function(data, model, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients for validation
        for batch_idx, (data, labels) in enumerate(tqdm(data)):
            X, y = data.to(device), labels.to(device)
            preds = model(X)

            loss = loss_fn(preds, y)

            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

    average_loss = total_loss / total_samples
    return average_loss



MODEL_PATH = './save'
LOAD_MODEL = False 
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
EPOCHS = 20
ALPHA = 1200
SIGMA = 30

MODEL_DIR = './save'
MODEL_FILENAME = 'model_checkpoint.pth'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Create the directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)






# getting the data 
images, masks = load_images_and_masks()

dataset = CustomDataset(images, masks, alpha=ALPHA, sigma=SIGMA)

# split the data set
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)  # 80% for training
validation_size = dataset_size - train_size  # 20% for validation

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])



# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# 

# checking the cuda 
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')





# ############## ŧrainig loop ##############
epoch = 0
LOSS_VALS = []
VALIDATION_LOSS_VALS = []
LEARNING_RATE_VALS = []

# getting the dataset and the data loader



# Defining the model, optimizer and loss function
unet = UNET(in_channels=1, classes=5).to(DEVICE).train()
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=255) 
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Learning rate scheduler



# Loading a previous stored model from MODEL_PATH variable
if LOAD_MODEL == True:
    checkpoint = torch.load(MODEL_PATH)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']+1
    LOSS_VALS = checkpoint['loss_values']
    print("Model successfully loaded!") 


#Training the model for every epoch. 
for e in range(epoch, EPOCHS):

    print("--------------------------------------------------------------------")
    print(f'Epoch: {e}')

    # Print current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current Learning Rate: {current_lr}')

    # traing
    loss_val = train_function(train_dataloader, unet, optimizer, loss_function, DEVICE)
    LOSS_VALS.append(loss_val) 
    print(f'Epoch Loss: {loss_val}')


    # mesure validation loss
    validation_loss = validate_function(validation_dataloader, unet, loss_function, DEVICE)
    VALIDATION_LOSS_VALS.append(validation_loss)
    print(f'Epoch: {e}, Validation Loss: {validation_loss}')



    scheduler.step()

    if e ==18:
        # Save checkpoint for each epoch with a unique file name
        epoch_model_path = os.path.join(MODEL_DIR, f'model_checkpoint_epoch_{e}.pth')
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, epoch_model_path)
        print(f"Epoch {e} completed and model successfully saved at {epoch_model_path}!")

    
   

    print("--------------------------------------------------------------------")



np.save('./save/training_loss.npy', np.array(LOSS_VALS))
np.save('./save/validation_loss.npy', np.array(validation_loss))

# ############## ŧrainig loop ##############






# checking the inputs and outputs of the dataloader
# for i, (input, target) in enumerate(dataloader):
#     print(f"Batch {i}")
#     print(f"Input Shape: {input.shape}")
#     print(f"Target Shape: {target.shape}")
#     print(f"Input Type: {input.dtype}")
#     print(f"Target Type: {target.dtype}")

#     # print(f"Batch {i}")
#     print(f"Input Max: {input.max().item()}")
#     print(f"Input Min: {input.min().item()}")
    
#     # Convert target tensor to numpy for np.unique
#     target_numpy = target.numpy()
#     print(f"Unique Target Values: {np.unique(target_numpy)}")

#     if i == 0:  # Adjust this number based on how many batches you want to check
#         break