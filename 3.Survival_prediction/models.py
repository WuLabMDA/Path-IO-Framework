"""Outer survival head exactly matches the original Path-IO CNN training."""
import torch
from torch import nn
from Macro_networks_new import resnet18, resnet34, resnet50, resnext50_32x4d

FACTORIES = {'R18': resnet18, 'R34': resnet34,
             'R50': resnet50, 'RX50': resnext50_32x4d}

class Net(nn.Module):
    def __init__(self, architecture, dropout=0.3):
        super().__init__()
        if architecture not in FACTORIES:
            raise ValueError(f'Unknown architecture {architecture}')
        self.m = FACTORIES[architecture](pretrained=False, in_channels=7,
                                         feature_dim=32, dropout=dropout)
        # Preserve original training code, including its unused maxpool attribute.
        self.m.conv1 = nn.Conv2d(7, 64, 3, 1, 1, bias=False)
        self.m.maxpool = nn.Identity()
        self.h = nn.Linear(32, 1)

    def patch_risk(self, x):
        embedding, _ = self.m(x)
        return self.h(embedding).squeeze(1)

    def forward(self, x):
        if x.ndim == 5:
            if x.shape[0] != 1:
                raise ValueError('Expected DataLoader batch size 1 patient')
            x = x.squeeze(0)
        return self.patch_risk(x).mean()
