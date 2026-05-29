import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

def get_trimap(mask):
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=3)
    dilated = cv2.dilate(mask, kernel, iterations=3)

    trimap = np.full(mask.shape, 128, dtype=np.uint8)  # Initilize trimap (gray)
    trimap[eroded == 255] = 255  # Set foreground (white)
    trimap[dilated == 0] = 0     # Set background (black)

    return trimap

if __name__ == "__main__":
    #Train split
    if not os.path.isdir('dataset/train/trimaps'):
        os.mkdir('dataset/train/trimaps')
        
    for mask_name in tqdm(os.listdir('dataset/train/mask')):
        mask_path = os.path.join('dataset/train/mask', mask_name)
        mask = cv2.imread(mask_path)
        trimap = get_trimap(mask)
        tri_img = Image.fromarray(trimap).convert('L')
        save_path = os.path.join('dataset/train/trimaps', mask_name)
        tri_img.save(save_path)

    #Test split
    if not os.path.isdir('dataset/test/trimaps'):
        os.mkdir('dataset/test/trimaps')
        
    for mask_name in tqdm(os.listdir('dataset/test/mask')):
        mask_path = os.path.join('dataset/test/mask', mask_name)
        mask = cv2.imread(mask_path)
        trimap = get_trimap(mask)
        tri_img = Image.fromarray(trimap).convert('L')
        save_path = os.path.join('dataset/test/trimaps', mask_name)
        tri_img.save(save_path)

    

