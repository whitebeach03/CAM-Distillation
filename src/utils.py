import random
import torch
import numpy as np
import os
import cv2

class EarlyStopping:
    def __init__(self, patience, verbose):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
            return False

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_param(es, loss, model, model_size, epoch, batch_size, num_class):
    if es(loss):
        return True
    else:
        torch.save(model.state_dict(), './logs/' + model_size + '/' + str(epoch) + '_' + str(batch_size) + '_' + str(num_class) + '_param.pth')
        return False

def create_CAM(images, cam, epoch, model_size):
    index = 2
    c = cam[index] 
    img = images[index].numpy()
    img = img.reshape(28,28)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 色をRGBに変換
    cv2.imwrite("cam_visualization/img.png",img*255)
    jetcam = cv2.applyColorMap(np.uint8(255 * c), cv2.COLORMAP_JET) # 着色
    jetcam = (np.float32(jetcam)/2 + img*122 ) # 元の画像に合成
    cv2.imwrite('cam_visualization/' + str(epoch) + '_' + str(model_size) + '_cam.png', jetcam)