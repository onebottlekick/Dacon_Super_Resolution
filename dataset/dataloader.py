import torch
from torch.utils.data import DataLoader, random_split

from dataset.dataset import TrainDataset


def split_dataset(dataset, args):
    train_dataset, val_dataset = random_split(dataset, (int(len(dataset)*args.train_val_split), len(dataset) - int(len(dataset)*args.train_val_split)), \
                                              generator=torch.Generator().manual_seed(args.seed))
    
    return train_dataset, val_dataset

def get_dataloader(args):
    dataset = TrainDataset(args)
    train_data, val_data = split_dataset(dataset, args)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    dataloader = {'train': train_loader, 'val': val_loader}
    
    return dataloader