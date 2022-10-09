import random

import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, list), 'transforms must be a list'
        self.transforms = transforms
        
    def __call__(self, lr, hr):
        for transform in self.transforms:
            lr, hr = transform(lr, hr)
            
        return lr, hr
    
    
class ToTensor:
    def __call__(self, lr, hr):
        lr = torch.tensor(lr).permute(2, 0, 1).float()
        hr = torch.tensor(hr).permute(2, 0, 1).float()
        return lr, hr
    
    
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, lr, hr):
        if random.random() < self.p:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
            
        if random.random() < self.p:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        
        return lr.copy(), hr.copy()
    
    
class RandomRotate:        
    def __call__(self, lr, hr):
        angle = random.choice([0, 1, 2, 3])
        lr = np.rot90(lr, angle)
        hr = np.rot90(hr, angle)
        
        return lr.copy(), hr.copy()
            
            
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, lr, hr):
        num_channels = lr.shape[-1]
        for i in range(num_channels):
            lr[:, :, i] = (lr[:, :, i] - self.mean[i]) / self.std[i]
            hr[:, :, i] = (hr[:, :, i] - self.mean[i]) / self.std[i]
        
        return lr.copy(), hr.copy()
    
    
class CenterCrop:
    def __init__(self, lr_crop_size, hr_crop_size):
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = hr_crop_size
        
    def __call__(self, lr, hr):
        lr_h, lr_w, _ = lr.shape
        lr_mid_h, lr_mid_w = lr_h//2, lr_w//2
        lr_crop_h, lr_crop_w = self.lr_crop_size//2, self.lr_crop_size//2
        lr = lr[lr_mid_h-lr_crop_h:lr_mid_h+lr_crop_h, lr_mid_w-lr_crop_w:lr_mid_w+lr_crop_w, :]
        
        hr_h, hr_w, _ = hr.shape
        hr_mid_h, hr_mid_w = hr_h//2, hr_w//2
        hr_crop_h, hr_crop_w = self.hr_crop_size//2, self.hr_crop_size//2
        hr = hr[hr_mid_h-hr_crop_h:hr_mid_h+hr_crop_h, hr_mid_w-hr_crop_w:hr_mid_w+hr_crop_w, :]
        
        return lr.copy(), hr.copy()
    

class Identity:
    def __call__(self, lr, hr):
        return lr, hr