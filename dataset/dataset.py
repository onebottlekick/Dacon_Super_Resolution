import os

import cv2
from torch.utils.data import Dataset

from dataset import data_augmentation as T


def get_img_list(path):
    return sorted([os.path.join(path, x) for x in os.listdir(path)])


def get_transform(args):
    if args.eval:
        _transform = T.Compose([
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.ToTensor()
        ])
    else:
        _transform = T.Compose([
            T.CenterCrop(args.lr_size, args.hr_size) if args.crop else T.Identity(),
            T.RandomFlip(args.augment_prob),
            T.RandomRotate(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.ToTensor()
        ])
    
    return _transform


class TrainDataset(Dataset):
    def __init__(self, args, transform=None):
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform(args)
            
        self.hr_list = get_img_list(args.hr_dataset_dir)
        self.lr_list = get_img_list(args.lr_dataset_dir)
        
    def __len__(self):
        return len(self.hr_list)
    
    def __getitem__(self, idx):
        hr = cv2.imread(self.hr_list[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = hr/255.0
        
        lr = cv2.imread(self.lr_list[idx])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = lr/255.0
        
        lr, hr = self.transform(lr, hr)
        
        item = {'hr': hr, 'lr': lr}
        
        return item