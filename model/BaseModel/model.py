import torch
import torch.nn as nn

from model.BaseModel.modules import LocalBlock, UpsampleBlock


class BaseModel(nn.Module):
    def __init__(self, num_features=64, num_blocks=3, num_local_blocks=3, args=None):
        super().__init__()
        self.num_blocks = num_blocks
        if args is not None:
            num_features = args.num_features
            self.num_blocks = args.num_blocks
            num_local_blocks = args.num_local_blocks
        
        self.to_features = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        )
        self.blocks = nn.ModuleList([LocalBlock(num_features, num_local_blocks) for i in range(self.num_blocks)])
        self.conv1x1s = nn.ModuleList([nn.Conv2d(num_features*i, num_features, kernel_size=1, stride=1, padding=0) for i in range(2, self.num_blocks+2)])
        self.upsample = UpsampleBlock(num_features)
        self.to_rgb = nn.Conv2d(num_features, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.to_features(x)
        b = []
        c = [x]
        o = [x]
        for i in range(self.num_blocks):
            _b = self.blocks[i](o[-1])
            b.append(_b)
            _c = torch.cat([c[-1], b[-1]], dim=1)
            c.append(_c)
            _o = self.conv1x1s[i](_c)
            o.append(_o)
        x = x + o[-1]
        x = self.upsample(x)
        x = self.to_rgb(x)
        
        return x