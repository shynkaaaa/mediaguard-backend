import torch
import torch.nn as nn
from torchvision import models
from .config import Config


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_dim),
            Swish(),
            nn.Linear(reduced_dim, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        expanded_dim = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))
        padding = (kernel_size - 1) // 2
        layers = []

        if expand_ratio != 1:
            layers += [nn.Conv2d(in_channels, expanded_dim, 1, bias=False),
                       nn.BatchNorm2d(expanded_dim, momentum=0.01, eps=1e-3), Swish()]

        layers += [nn.Conv2d(expanded_dim, expanded_dim, kernel_size,
                             stride=stride, padding=padding,
                             groups=expanded_dim, bias=False),
                   nn.BatchNorm2d(expanded_dim, momentum=0.01, eps=1e-3), Swish()]

        layers.append(SqueezeExcitation(expanded_dim, reduced_dim))
        layers += [nn.Conv2d(expanded_dim, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)]
        self.block = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        survival = 1 - self.drop_connect_rate
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + survival
        return x * torch.floor(mask) / survival

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = self._drop_connect(out) + x
        return out


class DeepfakeDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, Config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.backbone(x)
