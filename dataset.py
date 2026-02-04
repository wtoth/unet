import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class ISBIDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.image_directory = pd.read_csv(dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("images")]
        mask_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("labels")]
        image = Image.open(image_path)
        mask = Image.open(mask_path)


        if self.transform:
            image, mask = self.transform(image, mask)

        # normalize images
        image = image.float() / 255.0
        mask = mask.float() / 255.0 
        return image, mask