import os 
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf 
from pymatting.alpha.estimate_alpha_knn import estimate_alpha_knn
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.append(os.path.abspath(".."))

from dataloader import Alpha_Dataset
from utils import threshold, accuracy, SAD, MSE, gradient_loss


#Load Test dataset
test_dir = '../dataset/test'
t=0.7 #Threshold

dataset = Alpha_Dataset(test_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

mse_list = []
sad_list = []
acc_list = []
grd_list = []
infer_time = 0
for images, trimaps, gt_masks in tqdm(dataloader):
    for i in range(len(images)):
        image = np.array(images[i])
        trimap = np.array(trimaps[i])
        gt_mask = np.array(gt_masks[i])

        start_time = time.time()
        # output = estimate_alpha_cf(image, trimap)
        output = estimate_alpha_knn(image, trimap)
        infer_time += time.time() - start_time
        
        pred_masks = threshold(output, t)
        acc_list.append(accuracy(pred_masks.astype(np.int32), (gt_mask//200).astype(np.int32)))
        mse_list.append(MSE(output.astype(np.int32), (gt_mask//200).astype(np.int32)))
        sad_list.append(SAD(output.astype(np.int32), (gt_mask//200).astype(np.int32)))
        grad = gradient_loss(output, gt_mask)
        grd_list.append(grad)
    

acc_score = sum(acc_list)/len(acc_list)
mse_score = sum(mse_list)/len(mse_list)
sad_score = sum(sad_list)/len(sad_list)
grd_score = sum(grd_list)/len(grd_list)



print(f'MSE: {mse_score}\t SAD: {sad_score}\t Accuracy: {acc_score}\t Gradient Error: {grd_score}')
print(f"Inference Time per sample: {1000*infer_time/len(dataloader)} ms", )