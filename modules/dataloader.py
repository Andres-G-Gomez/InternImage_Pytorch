import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, images_path, masks_path, target_size = (224, 224), transform=True):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, images_path)
        self.mask_folder = os.path.join(root_dir, masks_path)
        self.target_size = target_size
        self.train_transform = transform 

        # Get a list of image filenames
        self.image_filenames = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_folder, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')  # Ensure the image is in RGB format

        mask_name = os.path.join(self.root_dir, self.mask_folder, self.image_filenames[idx])
        mask = Image.open(mask_name)  

        image, mask = self.transform_data(image, mask)
        return image, mask, self.image_filenames[idx]

    def transform_data(self, image, mask):
        # Define the transformation
        image_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255.0)  
        ])
        image = image_transforms(image)
        
        mask_transforms = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)),
            transforms.Lambda(lambda x: F.one_hot(x, num_classes=4).permute(2, 0, 1).to(torch.float))  # One-hot encoding
        ])
        mask = mask_transforms(mask).to(torch.float)
        
        
        return image, mask