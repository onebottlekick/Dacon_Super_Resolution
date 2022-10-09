import torch
import torch.nn as nn
from torchvision.models import vgg19

from loss.discriminator import Discriminator


class VGGLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = 'cuda' if args.gpu else 'cpu'
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.require_grad = False

    def forward(self, x, y):
        x = self.vgg(x)
        y = self.vgg(y)

        return self.loss(x, y)
    
    
class AdversarialLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.device = 'cuda' if args.gpu else 'cpu'
        self.discriminator = Discriminator(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(args.gan_betas), eps=args.gan_eps)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, gen, real):
        self.optimizer.zero_grad()
        d_fake = self.discriminator(gen.detach())
        d_real = self.discriminator(real)

        valid = torch.ones(real.shape[0], 1).to(self.device)
        fake = torch.zeros(real.shape[0], 1).to(self.device)

        if self.args.gan_type == 'vanilla':
            real_loss = self.bce_loss(d_real, valid)
            fake_loss = self.bce_loss(d_fake, fake)
            
        elif self.args.gan_type == 'relativistic':
            real_loss = self.bce_loss(d_real - d_fake.mean(0, keepdim=True), valid)
            fake_loss = self.bce_loss(d_fake - d_real.mean(0, keepdim=True), fake)
        
        else:
            raise NotImplementedError('Use [vanilla | relativistic]')

        d_loss = (real_loss + fake_loss) / 2.

        d_loss.backward()
        self.optimizer.step()

        g_loss = self.bce_loss(self.discriminator(gen) - self.discriminator(real).detach().mean(0, keepdim=True), valid)

        return g_loss
