import torch
from train import AlexNetModel
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
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.PILToTensor(),
        #v2.Resize((256, 256)),
        #v2.RandomCrop((224, 224)), # Get a random 3x224x224 subset of the image
    ])
    validation_transforms = v2.Compose([
        v2.PILToTensor(),
        #v2.Resize((256, 256)),
        #v2.CenterCrop((224, 224)), # Get a random 3x224x224 subset of the image
    ])

    alexnet_model = AlexNetModel(device, log=False)
    alexnet_model.train(root_directory, num_epochs, batch_size, learning_rate, momentum, weight_decay, transforms, validation_transforms)

if __name__ == "__main__":
    main()