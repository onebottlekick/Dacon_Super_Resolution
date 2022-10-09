import logging
import os
import shutil

import matplotlib.pyplot as plt


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if args.reset:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    if ((not args.eval) and (not args.test)):
        os.makedirs(os.path.join(args.save_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'metric'), exist_ok=True)
        if args.adversarial_loss_weight > 0.0:
            os.makedirs(os.path.join(args.save_dir, 'discriminator'), exist_ok=True)

    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'), exist_ok=True)

    args_file = open(os.path.join(args.save_dir, f'{args.log_file_name.split(".log")[0]}_args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name),
        logger_name=args.logger_name).get_log()

    return _logger


def plot_activations(img, model, blocks, layer_num, n_cols=8, n_rows=8):
    hook = BlockHook(blocks, layer_num)
    model(img)
    activations = hook.features
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 30))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for row in range(n_rows):
        for column in range(n_cols):
            axis = axes[row][column]
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.imshow(activations[0][row*n_rows + column], cmap='gray')

    plt.show()


class Logger:
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
    
    
class BlockHook:
    features = []
    def __init__(self, blocks, block_num):
        self.hook = blocks[block_num].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu().numpy()
        
    def remove(self):
        self.hook.remove()