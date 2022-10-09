import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from configs import args
from dataset.dataloader import get_dataloader
from model import BaseModel
from trainer import Trainer
from utils import mkExpDir
from loss.loss import VGGLoss, AdversarialLoss

if __name__ == '__main__':
    _logger = mkExpDir(args)
    _dataloader = get_dataloader(args) if not args.test else None
    _model = BaseModel(args=args)

    _reconstruction_loss = nn.L1Loss() if args.reconstruction_loss_type == 'l1' else nn.MSELoss()
    _criterion = {
        'reconstruction': _reconstruction_loss if args.reconstruction_loss_weight > 0 else None,
        'perceptual': VGGLoss(args) if args.perceptual_loss_weight > 0 else None,
        'adversarial': AdversarialLoss(args) if args.adversarial_loss_weight > 0 else None
    }
    _optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)

    trainer = Trainer(args, _logger, _dataloader, _model, _criterion, _optimizer)

    if args.test:
        trainer.load(checkpoint_path=args.checkpoint_path)
        trainer.test()

    elif args.eval:
        trainer.load(checkpoint_path=args.checkpoint_path)
        trainer.eval()

    else:
        if args.resume:
            trainer.load(checkpoint_path=args.checkpoint_path)
        elif args.load_pretrained:
            trainer.load(checkpoint_path=args.checkpoint_path)

        for epoch in range(args.start_epoch, args.num_epochs+1):
            trainer.train(cur_epoch=epoch)
            if epoch%args.val_every == 0:
                trainer.eval(cur_epoch=epoch)
