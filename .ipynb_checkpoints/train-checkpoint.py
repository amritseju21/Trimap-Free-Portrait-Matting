import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import HumanMattingDataset
from UNet import UNet
from tqdm import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cuda:1'

#Prepare Dataloaders
train_dataset = HumanMattingDataset('dataset/train', device)
test_dataset = HumanMattingDataset('dataset/test', device)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)

#Load Model
model = UNet(in_channels=3, out_channels=1)
model.to(device)

#Training Loop
lr = 1e-5
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 5 


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for x, masks in tqdm(train_dataloader):

        x = x.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(loss.item())

    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')