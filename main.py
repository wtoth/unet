import torch
from train import UNetModel
from torchvision.transforms import v2
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main():
    root_directory = "processed_data"

    if torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")

    # Hyperparams
    num_epochs = 40
    batch_size = 4 # keeping small to save in memory
    learning_rate = 0.001
    momentum = 0.99
    weight_decay = 5e-4

    spatial_transforms = v2.Compose([   
    v2.RandomAffine(degrees=180, translate=(0.1, 0.1)),
    v2.PILToTensor(),
    v2.ElasticTransform(alpha=50.0, sigma=5.0)
    ])

    color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    validation_transforms = v2.Compose([
        v2.PILToTensor(),
    ])

    alexnet_model = UNetModel(device, log=True)
    alexnet_model.train(root_directory, num_epochs, batch_size, learning_rate, momentum, weight_decay, spatial_transforms, color_transforms, validation_transforms)

if __name__ == "__main__":
    main()