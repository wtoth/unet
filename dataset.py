import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class ISBIDataset(Dataset):
    def __init__(self, dataset="processed_data/train_dataset.csv", transform=None):
        self.image_directory = pd.read_csv(dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("images")]
        print(image_path)
        label_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("labels")]
        image = Image.open(image_path)
        label = Image.open(label_path)


        if self.transform:
            image = self.transform(image)
        pil_to_tensor = v2.PILToTensor()
        image = pil_to_tensor(image)
        label = pil_to_tensor(label)
        return image, label