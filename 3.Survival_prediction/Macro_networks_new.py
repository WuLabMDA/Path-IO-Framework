"""Checkpoint-compatible custom GroupNorm ResNets for Path-IO.
Only four architectures used in this release. The internal `linear` layer
is retained for state-dict compatibility, even though Path-IO uses `Net.h`.
"""
import torch
from torch import nn

def norm(c):
    g = min(8, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)

def conv3(i, o, s=1, g=1, d=1):
    return nn.Conv2d(i, o, 3, s, padding=d, groups=g, bias=False, dilation=d)

def conv1(i, o, s=1):
    return nn.Conv2d(i, o, 1, s, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, p, s=1, down=None, groups=1, width=64, dilation=1):
        super().__init__()
        self.c1 = conv3(inp, p, s, groups, dilation)
        self.n1 = norm(p)
        self.c2 = conv3(p, p)
        self.n2 = norm(p)
        self.relu = nn.ReLU(inplace=True)
        self.down = down
    def forward(self, x):
        identity = x
        x = self.relu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.relu(x + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inp, p, s=1, down=None, groups=1, width=64, dilation=1):
        super().__init__()
        w = int(p * (width / 64.)) * groups
        self.c1 = conv1(inp, w)
        self.n1 = norm(w)
        self.c2 = conv3(w, w, s, groups, dilation)
        self.n2 = norm(w)
        self.c3 = conv1(w, p * 4)
        self.n3 = norm(p * 4)
        self.relu = nn.ReLU(inplace=True)
        self.down = down
    def forward(self, x):
        identity = x
        x = self.relu(self.n1(self.c1(x)))
        x = self.relu(self.n2(self.c2(x)))
        x = self.n3(self.c3(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.relu(x + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, feature_dim=32,
                 dropout=.5, in_channels=7, groups=1, width=64):
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.width = width
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.norm1 = norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(block, 64, layers[0])
        self.layer2 = self._layer(block, 128, layers[1], 2)
        self.layer3 = self._layer(block, 256, layers[2], 2)
        self.layer4 = self._layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, feature_dim)
        self.relu2 = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, num_classes)
        self._init()
    def _layer(self, block, p, n, s=1):
        down = None
        if s != 1 or self.inplanes != p * block.expansion:
            down = nn.Sequential(conv1(self.inplanes, p * block.expansion, s),
                                 norm(p * block.expansion))
        layers = [block(self.inplanes, p, s, down, self.groups, self.width)]
        self.inplanes = p * block.expansion
        for _ in range(1, n):
            layers.append(block(self.inplanes, p, groups=self.groups, width=self.width))
        return nn.Sequential(*layers)
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)  # Original code uses `pool`, not `maxpool`.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        hazard = self.linear(self.drop(self.relu2(embedding)))
        return embedding, hazard

def resnet18(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError('Pretrained ImageNet weights are not supported in this release; use Path-IO checkpoints.')
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError('Pretrained ImageNet weights are not supported in this release; use Path-IO checkpoints.')
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError('Pretrained ImageNet weights are not supported in this release; use Path-IO checkpoints.')
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width'] = 4
    if pretrained:
        raise ValueError('Pretrained ImageNet weights are not supported in this release; use Path-IO checkpoints.')
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
