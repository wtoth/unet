import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataset import ISBIDataset
from unet import UNet
import wandb


class UNetModel:
    def __init__(self, device,log=True):
        self.model = UNet().to(device)
        self.device = device
        self.log = log

    def train(self, root_directory, num_epochs, batch_size, learning_rate, momentum, weight_decay, transforms, validation_transforms):
        if self.log:
            wandb_log = self.init_logging()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        
        # Create Datasets 
        training_dataset = ISBIDataset(dataset=f"{root_directory}/train_dataset.csv", transform=transforms)
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = ISBIDataset(dataset=f"{root_directory}/test_dataset.csv", transform=validation_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float("inf")
        global_steps = 0
        for epoch in range(num_epochs):

            # begin training loop 
            self.model.train()
            running_loss = 0.0
            for i, (input, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                assert self.model.training, 'make sure the network is in train mode with `.train()`'

                optimizer.zero_grad() # zero out the gradient from previous batches
                
                input = input.to(self.device)
                labels = labels.squeeze(1).long().to(self.device)

                output = self.model(input) # Predictions

                # class_weights = torch.tensor([0.3, 0.7]).to(self.device) 
                # loss = F.cross_entropy(output, labels, weight=class_weights)
                loss = F.cross_entropy(output, labels)

                loss.backward() # backprop step
                optimizer.step() # SGD optimizing step

                running_loss += loss.item()
                if self.log and (i % 10 == 0):
                    wandb_log.log({"eval/loss": loss.item()}, step=global_steps)
                
                global_steps += 1
            
            validation_loss, validation_pixel_accuracy = self.validation(val_dataloader) 

            print(f"Epoch: {epoch+1} Training Loss: {running_loss/len(train_dataloader)} Validation Loss: {validation_loss} Validation Pixel Accuracy: {validation_pixel_accuracy}")
            if self.log:
                wandb_log.log({
                    "validation_loss": validation_loss,
                    "validation_pixel_accuracy": validation_pixel_accuracy
                })
            if validation_loss < best_loss:
                print("***saving best weights***")
                torch.save(self.model.state_dict(), "model_weights/best_val_loss.pt")
                best_loss = validation_loss

    def validation(self, val_dataloader):
        total_loss = 0
        total_correct = 0
        total_pixels = 0

        self.model.eval() # set model to eval mode so the weights won't get changed
        with torch.no_grad():
            for input, labels in tqdm(val_dataloader, desc="Validation Run"):
                input = input.to(self.device)
                labels = labels.squeeze(1).long().to(self.device)

                outputs = self.model(input) 

                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                # get accuracy
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_pixels += labels.numel()

        avg_loss = total_loss / len(val_dataloader)  
        pixel_accuracy = total_correct / total_pixels

        return avg_loss, pixel_accuracy
    
    def init_logging(self, learning_rate, momentum, weight_decay, num_epochs):
        wandb_log = wandb.init(
            entity="wtoth21",
            # Set the wandb project where this run will be logged.
            project="unet",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "unet",
                "dataset": "EM stacks challenge - ISBI 2012",
                "learning_rate": learning_rate,
                "momentum": momentum,
                "l2 regularization": weight_decay,
                "epochs": num_epochs,
            })
        return wandb_log