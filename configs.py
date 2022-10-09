import argparse


parser = argparse.ArgumentParser(description='Super-Resolution')

# dataset config
parser.add_argument('--hr-dataset-dir', type=str, default='dataset/train/hr', help='hr dataset path')
parser.add_argument('--lr-dataset-dir', type=str, default='dataset/train/lr', help='lr dataset path')
parser.add_argument('--crop', action='store_true', help='crop or not')
parser.add_argument('--lr-size', type=int, default=64, help='low resolution crop size')
parser.add_argument('--hr-size', type=int, default=256, help='high resolution crop size')
parser.add_argument('--augment-prob', type=float, default=0.5, help='probability of applying augmentation')
parser.add_argument('--train-val-split', type=float, default=0.95, help='ratio of training dataset')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers')

# logger config
parser.add_argument('--reset', action='store_true', help='reset logger')
parser.add_argument('--log-file-name', type=str, default='Model.log', help='Log file name')
parser.add_argument('--logger-name', type=str, default='Model', help='Logger name')
parser.add_argument('--save-dir', type=str, default='save_dir', help='Directory to save log, arguments, checkpoints')

# device config
parser.add_argument('--gpu', action='store_true', help='use gpu')

# model config
parser.add_argument('--lr-scale', type=int, default=4, help='lr scale')
parser.add_argument('--num-blocks', type=int, default=3, help='number of blocks')
parser.add_argument('--num-local-blocks', type=int, default=3, help='number of local blocks')
parser.add_argument('--num-features', type=int, default=64)

# loss config
parser.add_argument('--reconstruction-loss-type', type=str, default='mse', help='reconstruction loss type [default: MSE]')
parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0, help='l1 loss weight')
parser.add_argument('--perceptual-loss-weight', type=float, default=0.0, help='vgg loss weight')
parser.add_argument('--adversarial-loss-weight', type=float, default=0.0, help='adversarial loss weight')

# adversarial config
parser.add_argument('--gan-optimizer', type=str, default='adam', help='GAN optimizer')
parser.add_argument('--gan-learning-rate', type=float, default=1e-4, help='GAN optimizer learning rate')
parser.add_argument('--gan-betas', type=float, nargs='+', default=(0.9, 0.999), help='GAN optimizer betas')
parser.add_argument('--gan-eps', type=float, default=1e-8, help='GAN optimizer eps')
parser.add_argument('--gan-type', type=str, default='vanilla', help='gan type[gan, relativistic]')

# training config
parser.add_argument('--batch-size', type=int, default=20, help='batch size')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr-decay-rate', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr-decay-epoch', type=int, default=100, help='learning rate decay epoch')
parser.add_argument('--start-epoch', type=int, default=1, help='start epoch')
parser.add_argument('--num-epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--print-every', type=int, default=300, help='status print frequency')
parser.add_argument('--val-every', type=int, default=20, help='validation frequency')
parser.add_argument('--save-every', type=int, default=1, help='checkpoint save frequency')

# main config
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--checkpoint-path', type=str, default=None, help='checkpoint path')
parser.add_argument('--metric-checkpoint-path', type=str, default=None, help='metric checkpoint path')
parser.add_argument('--discriminator-checkpoint-path', type=str, default=None, help='discriminator path')
parser.add_argument('--load-pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--eval', action='store_true', help='evaluate model')
parser.add_argument('--eval-save-results', action='store_true', help='save results on evaluation')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--test-img-path', type=str, default=None, help='test image path')
parser.add_argument('--save-bicubic', action='store_true', help='save bicubic results')


args = parser.parse_args()

if __name__ == '__main__':
    print(args)