"""CIFAR-style ResNet (He et al. 2015, section 4.2) and an analytic FLOP counter.

These are the small 3-stage ResNets built for 32x32 inputs — ResNet-20/32/56 —
not the ImageNet ResNet-18/50 family. ResNet-20 reaches ~91.5% and ResNet-56
~93.5% on CIFAR-10, both in minutes, which is what makes time-to-accuracy
experiments cheap enough to repeat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, num_blocks: int, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# depth = 6n + 2
_DEPTHS = {20: 3, 32: 5, 44: 7, 56: 9, 110: 18}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if not name.startswith("resnet"):
        raise ValueError(f"unknown model {name!r}")
    depth = int(name.removeprefix("resnet"))
    if depth not in _DEPTHS:
        raise ValueError(f"depth {depth} not a CIFAR ResNet; pick from {sorted(_DEPTHS)}")
    return CifarResNet(_DEPTHS[depth], num_classes)


@torch.no_grad()
def count_flops_per_sample(model: nn.Module, input_shape=(3, 32, 32)) -> int:
    """Multiply-accumulate FLOPs for one forward pass of one sample.

    Counted by hooking Conv2d/Linear and using output spatial size, which is
    the standard convention (2 FLOPs per MAC). Elementwise ops (BN, ReLU, adds)
    are excluded — they're a rounding error against the convolutions and
    including them would make MFU incomparable to published numbers, which
    also count only matmul/conv.
    """
    total = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal total
        out_elems = output.numel() / output.shape[0]
        macs_per_out = (
            module.in_channels
            // module.groups
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        total += int(2 * out_elems * macs_per_out)

    def linear_hook(module, inputs, output):
        nonlocal total
        total += int(2 * module.in_features * module.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)

    for h in hooks:
        h.remove()
    return total
