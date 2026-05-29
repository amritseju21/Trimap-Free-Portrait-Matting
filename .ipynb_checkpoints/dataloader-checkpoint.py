import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class HumanMattingDataset(Dataset):
    def __init__(self, data_root, device):
        self.device = device
        
        self.img_dir = os.path.join(data_root,'img')
        self.trimap_dir = os.path.join(data_root,'trimaps')
        self.mask_dir = os.path.join(data_root,'mask')

        self.transform = transforms.ToTensor()
#         self.transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize()
# ])
        
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        image = Image.open(image_path)
        image = self.transform(image)
        
        # trimap_path = os.path.join(self.trimap_dir, str(idx)+'.png')
        # trimap = Image.open(trimap_path)
        # trimap = self.transform(trimap)
        
        mask_path = os.path.join(self.mask_dir, str(idx)+'.png')
        mask = Image.open(mask_path)
        mask = self.transform(mask)
        
        # x = torch.cat((image, trimap), dim=0) #Concat along channel
        # x = x.to(self.device)
        # mask = mask.to(self.device)
        return image, mask