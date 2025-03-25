import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet
from torch.utils.data import DataLoader
from data import SegmentationDataset, transform_img 

transform = transform_img()

train_dataset = SegmentationDataset("DUTS-TR-Image", "DUTS-TR-Mask", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = SegmentationDataset("DUTS-TE-Image", "DUTS-TE-Mask", transform=transform)
test_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in dataloader:
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            total_correct += (preds==masks).sum().item()
            total_pixels += torch.numel(preds)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels
    return avg_loss, accuracy

num_epochs = 2
total_correct = 0
total_pixels = 0

train_loss_lst = []
train_accuracy_lst = []
test_loss_lst = []
test_accuracy_lst = []

for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    model.train()
    epoch_loss = 0
    
    for images, masks in train_dataloader:
        
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(outputs) > 0.5
        total_correct += (preds==masks).sum().item()
        total_pixels += torch.numel(preds)

        epoch_loss += loss.item()

    train_accuracy = total_correct / total_pixels        
    avg_train_loss = epoch_loss/len(train_dataloader)
    print(f"Train loss at {epoch+1} epoch: {avg_train_loss}")
    print(f"Train accuracy at {epoch+1} epoch: {train_accuracy}")
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Test loss at {epoch+1} epoch: {test_loss}")
    print(f"Test accuracy at {epoch+1} epoch: {test_accuracy}")
    train_loss_lst.append(avg_train_loss)
    test_loss_lst.append(test_loss)
    train_accuracy_lst.append(train_accuracy)
    test_accuracy_lst.append(test_accuracy)