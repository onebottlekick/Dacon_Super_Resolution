import zipfile
from tqdm import tqdm


def extract(zip_path='./open.zip', out_path='./'):
    with zipfile.ZipFile(zip_path, 'r') as f:
        for name in tqdm(iterable=f.namelist(), total=len(f.namelist())):
            f.extract(member=name, path=out_path)
            
            
if __name__ == '__main__':
    extract()