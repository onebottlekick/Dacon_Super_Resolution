import argparse
import os
import zipfile

from tqdm import tqdm


def make_submission_file(path='test_results', submission_file_name='submission.zip'):
    cur_dir = os.path.abspath(os.curdir)
    path = os.path.join(path, 'save_results')
    os.makedirs('submission', exist_ok=True)
    save_dir = os.path.join(os.path.abspath(os.curdir), 'submission')
    os.chdir(path)
    
    zip_file = zipfile.ZipFile(os.path.join(save_dir, submission_file_name), 'w')
    for img in tqdm(os.listdir(os.curdir), desc='Making submission file'):
        if 'bicubic' not in img:
            zip_file.write(img)
            
    os.chdir(cur_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make submission file')
    parser.add_argument('path', type=str, help='submission img path')
    parser.add_argument('name', type=str, help='submission file name')
    args = parser.parse_args()
    
    make_submission_file(args.path, args.name)