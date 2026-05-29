import cv2
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import random
from utils import viola_jones
import numpy as np
from pymatting import load_image

class HumanMattingDataset(Dataset):
    def __init__(self, data_root, device, mode='train'):
        self.device = device
        
        self.img_dir = os.path.join(data_root,'img')
        self.mask_dir = os.path.join(data_root,'mask')

        self.transform = transforms.ToTensor()
        self.mode = mode
        
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.mode=='test': idx+=27540
        image_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        image = Image.open(image_path)
        image = self.transform(image)
        
        mask_path = os.path.join(self.mask_dir, str(idx)+'.png')
        mask = Image.open(mask_path)
        mask = self.transform(mask)
        return image, mask
    



class ViT_Dataset(Dataset):
    def __init__(self, data_root, device, mode='test'):
        self.device = device
        
        self.img_dir = os.path.join(data_root,'img')
        self.trimap_dir = os.path.join(data_root,'trimaps')
        self.mask_dir = os.path.join(data_root,'mask')

        self.transform = transforms.ToTensor()
        self.mode = mode
        
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.mode=='test': idx+=27540
        image_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        mask_path = os.path.join(self.mask_dir, str(idx)+'.png')
        trimap_path = os.path.join(self.trimap_dir, str(idx)+'.png')


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        trimap = cv2.imread(trimap_path)

        mask = Image.open(mask_path)
        mask = self.transform(mask)

        return image, trimap, mask


# def Alpha_Dataset(data_root):
#     img_dir = os.path.join(data_root,'img')
#     trimap_dir = os.path.join(data_root,'trimaps')
#     mask_dir = os.path.join(data_root,'mask')

#     n = len(os.listdir(img_dir))

#     for i in range(n):
#         k = i + 27540
#         image_path = os.path.join(img_dir, str(k)+'.jpg')
#         trimap_path = os.path.join(trimap_dir, str(k)+'.png')
#         mask_path = os.path.join(mask_dir, str(k)+'.png')

#         image = load_image(image_path)
#         trimap = load_image(trimap_path)
#         mask = cv2.imread(mask_path)
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


#         yield image, trimap, mask//200


class Alpha_Dataset(Dataset):
    def __init__(self, data_root, mode='test'):
        
        self.img_dir = os.path.join(data_root,'img')
        self.trimap_dir = os.path.join(data_root,'trimaps')
        self.mask_dir = os.path.join(data_root,'mask')
        self.mode = mode
        
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.mode=='test': idx+=27540

        image_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        trimap_path = os.path.join(self.trimap_dir, str(idx)+'.png')
        mask_path = os.path.join(self.mask_dir, str(idx)+'.png')

        image = load_image(image_path)
        trimap = load_image(trimap_path)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return image, trimap, mask
