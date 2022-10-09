import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(num_features, num_features, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.Conv2d(num_features, num_features, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(True),
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0),
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(num_features, num_features, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(num_features, num_features, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(True),
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = x + x1 + x2 + x3 + x4
        x = nn.ReLU(True)(x)
        
        return x
    
    
    
class LocalBlock(nn.Module):
    def __init__(self, num_features=64, num_local_blocks=3):
        super().__init__()
        self.num_local_blocks = num_local_blocks
        
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_features) for _ in range(num_local_blocks)])
        self.conv1x1s = nn.ModuleList([nn.Conv2d(num_features*i, num_features, kernel_size=1, stride=1, padding=0) for i in range(2, num_local_blocks+2)])
        
    def forward(self, x):
        r = []
        c = [x]
        o = [x]
        for i in range(self.num_local_blocks):
            _r = self.residual_blocks[i](o[-1])
            r.append(_r)
            _c = torch.cat([c[-1], r[-1]], dim=1)
            c.append(_c)
            _o = self.conv1x1s[i](_c)
            o.append(_o)
            
        return o[-1]


class UpsampleBlock(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(num_features, num_features*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(num_features, num_features*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        
        return x