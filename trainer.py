import os

import cv2
import numpy as np
import torch

from metrics import calc_psnr_and_ssim


def lr_decay(cur_epoch, learning_rate, lr_decay_rate, lr_decay_epoch):
        lr = learning_rate*(lr_decay_rate**(cur_epoch//lr_decay_epoch))
        
        return lr


class Trainer:
    def __init__(self, args, logger, dataloader, model, criterion, optimizer):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if self.args.gpu else 'cpu'
        self.dataloader = dataloader
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.use_reconstruction = bool(self.criterion['reconstruction'])
        self.use_perceptual = bool(self.criterion['perceptual'])
        self.use_adversarial = bool(self.criterion['adversarial'])

        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, checkpoint_path=None):
        if checkpoint_path:
            self.logger.info('load_model_path' + checkpoint_path)
            model_state_dict_save = {k:v for k, v in torch.load(checkpoint_path, map_location=self.device)['model'].items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)
            
            if not (self.args.test or self.args.eval):
                optimizer_state_dict_save = {k:v for k, v in torch.load(checkpoint_path, map_location=self.device)['optimizer'].items()}
                optimizer_state_dict = self.optimizer.state_dict()
                optimizer_state_dict.update(optimizer_state_dict_save)
                self.optimizer.load_state_dict(optimizer_state_dict)
            
            if self.args.metric_checkpoint_path:
                metric_checkpoint = torch.load(self.args.metric_checkpoint_path)
                self.max_psnr = metric_checkpoint['max_psnr']
                self.max_psnr_epoch = metric_checkpoint['max_psnr_epoch']
                self.max_ssim = metric_checkpoint['max_ssim']
                self.max_ssim_epoch = metric_checkpoint['max_ssim_epoch']
        
            if self.use_adversarial:
                discriminator_path = self.args.discriminator_checkpoint_path
                if discriminator_path is None:
                    print('No discriminator path provided')
                else:
                    self.logger.info('load_discriminator_path' + discriminator_path)
                    discriminator_state_dict_save = {k:v for k, v in torch.load(discriminator_path, map_location=self.device)['discriminator'].items()}
                    discriminator_state_dict = self.criterion['adversarial'].discriminator.state_dict()
                    discriminator_state_dict.update(discriminator_state_dict_save)
                    self.criterion['adversarial'].discriminator.load_state_dict(discriminator_state_dict)

                    disc_optimizer_state_dict_save = {k:v for k, v in torch.load(discriminator_path, map_location=self.device)['optimizer'].items()}
                    disc_optimizer_state_dict = self.criterion['adversarial'].optimizer.state_dict()
                    disc_optimizer_state_dict.update(disc_optimizer_state_dict_save)
                    self.criterion['adversarial'].optimizer.load_state_dict(disc_optimizer_state_dict)
            
        else:
            raise ValueError('No checkpoint path provided')

    def to_device(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)

        return sample_batched    

    def train(self, cur_epoch=0):
        self.model.train()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_decay(cur_epoch, self.args.learning_rate, self.args.lr_decay_rate, self.args.lr_decay_epoch)
        self.logger.info(f'Current epoch learning rate: {self.optimizer.param_groups[0]["lr"]}')

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.to_device(sample_batched)
            lr = sample_batched['lr']
            hr = sample_batched['hr']

            sr = self.model(lr)

            is_print = ((i_batch + 1) %self.args.print_every == 0)

            loss = torch.zeros(1).to(self.device)
            
            if self.use_reconstruction:
                reconstruction_loss = self.criterion['reconstruction'](sr, hr)
                loss += reconstruction_loss*self.args.reconstruction_loss_weight
            
            if self.use_perceptual:
                perceptual_loss = self.criterion['perceptual'](sr, hr)
                loss += perceptual_loss*self.args.perceptual_loss_weight
                
            if self.use_adversarial:
                adversarial_loss = self.criterion['adversarial'](sr, hr)
                loss += adversarial_loss*self.args.adversarial_loss_weight
            
            if (is_print):
                if self.use_reconstruction:
                    self.logger.info(f'Epoch: {cur_epoch}\tbatch: {i_batch + 1}')
                    self.logger.info(f'reconstruction loss: {reconstruction_loss.item():.4f}')
                if self.use_perceptual:
                    self.logger.info(f'perceptual loss: {perceptual_loss.item():.4f}')
                if self.use_adversarial:
                    self.logger.info(f'adversarial loss: {adversarial_loss.item():.4f}')
                self.logger.info(f'toal_loss: {loss.item():.4f}')                
            
            loss.backward()
            self.optimizer.step()
            
        checkpoints = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.use_adversarial:
            disc_checkpoints = {
                'discriminator': self.criterion['adversarial'].discriminator.state_dict(),
                'optimizer': self.criterion['adversarial'].optimizer.state_dict(),
            }  
        if cur_epoch%self.args.save_every == 0:
            self.logger.info('saving model...')
            model_name = self.args.save_dir.strip('/') + '/model/' + self.args.log_file_name.strip('.log') + '_' + str(cur_epoch).zfill(5) + '.pth.tar'
            torch.save(checkpoints, model_name)
            if self.use_adversarial:
                torch.save(disc_checkpoints, os.path.join(self.args.save_dir, 'discriminator', 'discriminator.pth.tar'))

    def eval(self, cur_epoch=0):
        self.logger.info(f'Epoch {cur_epoch} evaluation')
        self.model.eval()
        with torch.no_grad():
            psnr, ssim, cnt = 0., 0., 0
            for i_batch, sample_batched in enumerate(self.dataloader['val']):
                cnt += 1
                sample_batched = self.to_device(sample_batched)
                lr = sample_batched['lr']
                hr = sample_batched['hr']

                sr = self.model(lr)

                if self.args.eval_save_results:
                    sr_save = (sr + 1.)*127.5
                    sr_save = cv2.cvtColor(np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5) + '.png'), sr_save)

                _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                psnr += _psnr
                ssim += _ssim

            avg_psnr = psnr/cnt
            avg_ssim = ssim/cnt
            self.logger.info(f'PSNR (now): {avg_psnr:.3f} \t SSIM (now): {avg_ssim:.3f}')
            if avg_psnr > self.max_psnr:
                self.max_psnr = avg_psnr
                self.max_psnr_epoch = cur_epoch
            if avg_ssim > self.max_ssim:
                self.max_ssim = avg_ssim
                self.max_ssim_epoch = cur_epoch
                
            metric_checkpoint = {
                'max_psnr': self.max_psnr,
                'max_psnr_epoch': self.max_psnr_epoch,
                'max_ssim': self.max_ssim,
                'max_ssim_epoch': self.max_ssim_epoch,
            }
            self.logger.info(f'PSNR (max): {self.max_psnr:.3f} on epoch: {self.max_psnr_epoch} \t SSIM (max): {self.max_ssim:.3f} on epoch: {self.max_ssim_epoch}')
            if not (self.args.eval or self.args.test):
                torch.save(metric_checkpoint, os.path.join(self.args.save_dir, 'metric', 'metric.pth.tar'))

        self.logger.info('Eval Done')

    def test(self):
        self.logger.info('Test')
        self.logger.info(f'LR path {self.args.test_img_path}')
        
        lrs = [os.path.join(self.args.test_img_path, img) for img in os.listdir(self.args.test_img_path)]
        for _lr in lrs:
            lr = cv2.imread(_lr)
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            
            if self.args.save_bicubic:
                lr_bicubic = cv2.resize(lr, (lr.shape[1]*self.args.lr_scale, lr.shape[0]*self.args.lr_scale), interpolation=cv2.INTER_CUBIC)
                bicubic_save_path = os.path.join(self.args.save_dir, 'save_results', f'{os.path.basename(_lr).split(".")[0]}_bicubic.png')
                cv2.imwrite(bicubic_save_path, cv2.cvtColor(lr_bicubic, cv2.COLOR_RGB2BGR))
            
            lr = lr/255.0
            for i in range(3):
                lr[:, :, i] = (lr[:, :, i] - 0.5) / 0.5
            lr = torch.tensor(lr).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                sr = self.model(lr)
                sr = (sr + 1.)*127.5
                sr_save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(_lr))
                sr = cv2.imwrite(sr_save_path, cv2.cvtColor(sr.squeeze().round().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
                self.logger.info(f'Save {sr_save_path} Done')
                
        self.logger.info('Test Done')
