import os 
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from PIL import Image
import cv2
from transformers import AutoImageProcessor, VitMatteForImageMatting

sys.path.append(os.path.abspath(".."))

from dataloader import ViT_Dataset
from utils import threshold, accuracy, SAD, MSE, gradient_loss

#Load Test dataset
test_dir = '../dataset/test'

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ViT_Dataset(test_dir, device, mode='test')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

processor = AutoImageProcessor.from_pretrained("hustvl/vitmatte-base-distinctions-646")
model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-base-distinctions-646")

model.to(device)


t=0.7 #Threshold
mse_list = []
sad_list = []
acc_list = []
grd_list = []
infer_time = 0
for imgs, trimaps, gt_masks in tqdm(dataloader):

    for i in range(len(imgs)):
        x = trimaps[i].numpy()
        trimap = Image.fromarray(x).convert('L')

        start_time = time.time()
        pixel_values = processor(images=imgs[i], trimaps=trimap, return_tensors="pt").pixel_values
        outputs = model(pixel_values.to(device))
        outputs = outputs.alphas.flatten(0, 2)
        infer_time += time.time() - start_time
        outputs = cv2.resize(outputs.detach().cpu().numpy(), trimap.size)

        
        pred_masks = threshold(outputs, t)
        gt = gt_masks[i][0].detach().cpu().numpy()
        acc_list.append(accuracy(pred_masks, gt))
        mse_list.append(MSE(outputs, gt))
        sad_list.append(SAD(outputs, gt))
        grd_list.append(gradient_loss(outputs,gt))

acc_score = sum(acc_list)/len(acc_list)
mse_score = sum(mse_list)/len(mse_list)
sad_score = sum(sad_list)/len(sad_list)
grd_score = sum(grd_list)/len(grd_list)



print(f'MSE: {mse_score}\t SAD: {sad_score}\t Accuracy: {acc_score}\t Gradient Error: {grd_score}')
print(f"Inference Time per sample: {1000*infer_time/len(dataloader)} ms", )