from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from skimage import img_as_float
from scipy.ndimage import gaussian_gradient_magnitude

def threshold(x, t):
    return (x>t).astype(np.uint8)


def eval_loss(model, dataloader, criterion, device):
    running_loss=0
    for x, masks in dataloader:
        x, masks = x.to(device), masks.to(device)
        outputs = model(x) #batch_size, 1, x, y
        loss = criterion(outputs, masks)
        running_loss += loss.item()
    return running_loss/len(dataloader)


def gradient_error(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)

    grad_x = cv2.Sobel(pred_mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(pred_mask, cv2.CV_64F, 0, 1, ksize=3)
    grad_x_gt = cv2.Sobel(gt_mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_gt = cv2.Sobel(gt_mask, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag_gt = np.sqrt(grad_x_gt**2 + grad_y_gt**2)
    grad_mag_gt[grad_mag_gt < 1e-8] = 1e-8 #Zero division

    grad_diff_x = (grad_x - grad_x_gt) / grad_mag_gt
    grad_diff_y = (grad_y - grad_y_gt) / grad_mag_gt

    grad_error = np.sqrt(grad_diff_x ** 2 + grad_diff_y ** 2)
    error = np.mean(grad_error)

    return error


def connectivity_error(pred_mask, gt_mask, threshold=0.1):
    # alpha_bin = (alpha > threshold).astype(np.uint8)
    # alpha_gt_bin = (alpha_gt > threshold).astype(np.uint8)

    connectivity_error = np.sum(np.abs(pred_mask - gt_mask))

    return connectivity_error

def MSE(mask1, mask2):
    return np.mean((mask1 - mask2) ** 2)
    
def accuracy(mask1, mask2):
    return 1 - np.mean(np.abs(mask1 - mask2))


def SAD(mask1, mask2):
    abs_diff = np.abs(mask1 - mask2)
    return np.sum(abs_diff) 


def viola_jones(img):
    img = np.array(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    height, width = img.shape[:2]

    cropped_imgs = []
    dims_list = []
    for face in faces:
        (x, y, w, h) = face
        new_w = int(w*1.4) #Increase width by x2
        new_h = int(h*2.0) #Increase height by x2.4
        
        # Ensure it stays within image bounds
        new_x = max(0, x-(new_w-w)//2)
        new_y = max(0, y-(new_h-h)//2)
        
        new_x = min(new_x, width-new_w)
        new_y = min(new_y, height-new_h)

        cropped = img[new_y:new_y + new_h, new_x:new_x + new_w]
        cropped = Image.fromarray(cropped)
        
        dims =  (int(new_x), int(new_w), int(new_y), int(new_h))
        
        cropped_imgs.append(cropped)
        dims_list.append(dims)

    return cropped_imgs, dims_list

def gradient_loss(pred, target, sigma=1.4):
    pred = img_as_float(pred)
    target = img_as_float(target)

    pred_amp = gaussian_gradient_magnitude(pred, sigma=sigma)
    target_amp = gaussian_gradient_magnitude(target, sigma=sigma)

    error_map = (pred_amp - target_amp) ** 2

    loss = np.sum(error_map.astype(np.float32))
    
    return loss