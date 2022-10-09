import argparse
import os

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def _make_patches(img_path, patch_size, save_dir='hi'):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(img_path).split('.')[0]
    
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(patches.shape[0], -1, patch_size, patch_size)

    for i in range(patches.shape[1]):
        patch = transforms.ToPILImage()(patches[:, i, :, :])
        patch.save(os.path.join(save_dir, f'{base_name}_{i:04d}.png'))
        

def hr_patch_extraction(hr_path='./train/hr', patch_size=512, save_dir='hr_patches'):
    hrs = os.listdir(hr_path)
    for hr in tqdm(hrs, desc='HR PATCH'):
        _make_patches(os.path.join(hr_path, hr), patch_size, save_dir)
        

def lr_patch_extraction(lr_path='./train/lr', patch_size=128, save_dir='lr_patches'):
    lrs = os.listdir(lr_path)
    for lr in tqdm(lrs, desc='LR PATCH'):
        _make_patches(os.path.join(lr_path, lr), patch_size, save_dir)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Patch Preprocessing')
    parser.add_argument('--hr_path', type=str, default='./train/hr', help='HR image path')
    parser.add_argument('--hr-patch-size', type=int, default=512, help='HR patch size')
    parser.add_argument('--hr-save-dir', type=str, default='hr_patches', help='HR patch save directory')
    parser.add_argument('--lr_path', type=str, default='./train/lr', help='LR image path')
    parser.add_argument('--lr-patch-size', type=int, default=128, help='LR patch size')
    parser.add_argument('--lr-save-dir', type=str, default='lr_patches', help='LR patch save directory')
    args = parser.parse_args()
    
    hr_patch_extraction(args.hr_path, args.hr_patch_size, args.hr_save_dir)
    lr_patch_extraction(args.lr_path, args.lr_patch_size, args.lr_save_dir)