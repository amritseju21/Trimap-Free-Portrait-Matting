import os 
import sys
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))

from UNet import UNet
from dataloader import HumanMattingDataset
from utils import threshold, MSE, SAD, accuracy, gradient_loss


#Load Test dataset
test_dir = '../dataset/test'

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HumanMattingDataset(test_dir, device, mode='test')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)


ckpt_path = '../checkpoints/final_model.pth'

state_dict = torch.load(ckpt_path, weights_only=False)
model = UNet(in_channels=3, out_channels=1)
model = torch.nn.DataParallel(model) #Weights Stored in DataParallel format
model.load_state_dict(state_dict)
model.to(device)
model.eval()
model = model.module 


t=0.7 #Threshold
mse_list = []
sad_list = []
acc_list = []
grd_list = []
infer_time = 0
for x, gt_masks in tqdm(dataloader):
    x, gt_masks = x.to(device), gt_masks.to(device)
    start_time = time.time()
    outputs = model(x) #batch_size, 1, x, y
    infer_time += time.time() - start_time
    outputs = outputs.detach().cpu().numpy()

    for i in range(len(outputs)):
        pred_masks = threshold(outputs, t)
        conf_maps = outputs[i][0]
        pred = pred_masks[i][0]
        gt = gt_masks[i][0].detach().cpu().numpy()

        acc_list.append(accuracy(pred, gt))
        sad_list.append(SAD(conf_maps, gt))
        mse_list.append(MSE(conf_maps, gt))
        grd_list.append(gradient_loss(conf_maps,gt))


acc_score = sum(acc_list)/len(acc_list)
mse_score = sum(mse_list)/len(mse_list)
sad_score = sum(sad_list)/len(sad_list)
grd_score = sum(grd_list)/len(grd_list)



print(f'MSE: {mse_score}\t SAD: {sad_score}\t Accuracy: {acc_score}\t Gradient Error: {grd_score}')
print(f"Inference Time per sample: {1000*infer_time/len(dataloader)} ms", )
