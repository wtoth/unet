import torch
from train import UNetModel
from torchvision.transforms import v2

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
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.PILToTensor(),
    ])
    validation_transforms = v2.Compose([
        v2.PILToTensor(),
    ])

    alexnet_model = UNetModel(device, log=False)
    alexnet_model.train(root_directory, num_epochs, batch_size, learning_rate, momentum, weight_decay, transforms, validation_transforms)

if __name__ == "__main__":
    main()