import pandas as pd
import numpy as np
from scipy import ndimage
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class ISBIDataset(Dataset):
    def __init__(self, dataset, spatial_transforms=None, color_transforms=None, w0=10, sigma=5):
        self.image_directory = pd.read_csv(dataset)
        self.spatial_transforms = spatial_transforms
        self.color_transforms = color_transforms
        self.w0 = w0
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("images")]
        mask_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("labels")]
        image = Image.open(image_path)
        mask = Image.open(mask_path)


        if self.spatial_transforms:
            # spatial transforms applied to both
            image, mask = self.spatial_transforms(image, mask)

        if self.color_transforms:
            # color transforms applied only to image
            image = self.color_transforms(image)

        # normalize images
        image = image.float() / 255.0
        mask = mask.float() / 255.0 

        weight_map = self.compute_weight_map(mask)

        return image, mask, weight_map
    
    def compute_weight_map(self, mask: torch.Tensor) -> torch.Tensor:
        mask_np = mask.squeeze().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)

        # w_c(x) - weight each pixel inversely proportional to its class frequency
        pos_ratio = binary.mean()
        neg_ratio = 1.0 - pos_ratio
        w_c = np.where(binary, 1.0 / (pos_ratio + 1e-6), 1.0 / (neg_ratio + 1e-6))
        w_c = w_c / w_c.mean() #

        labeled, num_objects = ndimage.label(binary)
        if num_objects < 2:
            return torch.from_numpy(w_c).float().unsqueeze(0)

        # compute distance to every object
        distances = np.zeros((num_objects, *mask_np.shape), dtype=np.float32)
        for i in range(num_objects):
            obj = (labeled == (i + 1))
            distances[i] = ndimage.distance_transform_edt(~obj)

        # get 2 shortest distances
        distances.sort(axis=0)
        d1 = distances[0]
        d2 = distances[1]

        # w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2 / 2sigma^2) from paper
        w_border = self.w0 * np.exp(-((d1 + d2) ** 2) / (2 * self.sigma ** 2))

        weight_map = w_c + w_border
        return torch.from_numpy(weight_map).float().unsqueeze(0)