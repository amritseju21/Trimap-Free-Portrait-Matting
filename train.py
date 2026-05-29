import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import HumanMattingDataset
from UNet import UNet
from tqdm import tqdm
from utils import eval_loss
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cuda:1'

#Prepare Dataloaders
train_dataset = HumanMattingDataset('dataset/train', device)
test_dataset = HumanMattingDataset('dataset/test', device, mode='test')

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8)

#Load Model
model = UNet(in_channels=3, out_channels=1)
model.to(device)
model = nn.DataParallel(model)

#Training Loop
show_progress = False
lr = 1e-5
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 50

for epoch in range(num_epochs):
    #Train Phase
    model.train()
    running_loss = 0.0
    train_loss_list = []
    test_loss_list = []

    for x, masks in tqdm(train_dataloader, disable=not show_progress):
        
        x = x.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Eval Phase    
    model.eval()
    test_loss = eval_loss(model, test_dataloader, criterion, device)
    
    test_loss_list.append(test_loss)
    average_loss = running_loss / len(train_dataloader)
    train_loss_list.append(average_loss)
    print(f'\nEpoch [{epoch + 1}/{num_epochs}]\tTrain Loss: {average_loss:.4f}\tTest Loss: {test_loss:.4f}')

    #Save Model 
    torch.save(model.state_dict(), 'checkpoints/unet_model.pth')


#Save Loss Curves
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.savefig('logs/loss_curves.png')

plt.imshow(outputs[0][0].detach().cpu())
plt.savefig('logs/random_output.png')