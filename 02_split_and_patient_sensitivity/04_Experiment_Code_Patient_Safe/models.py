from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        attention = torch.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)
        value = self.value(x).flatten(2)
        attended = torch.bmm(value, attention.transpose(1, 2)).view(b, c, h, w)
        return x + self.gamma * attended


def down_block(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True), nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Pix2PixGenerator(nn.Module):
    """U-Net generator with a bottleneck attention block for the revised main experiment."""
    def __init__(self, base: int = 64):
        super().__init__()
        self.d1 = down_block(1, base, normalize=False)
        self.d2 = down_block(base, base * 2)
        self.d3 = down_block(base * 2, base * 4)
        self.d4 = down_block(base * 4, base * 8)
        self.d5 = down_block(base * 8, base * 8)
        self.attention = SelfAttention2d(base * 8)
        self.u1 = up_block(base * 8, base * 8, 0.5)
        self.u2 = up_block(base * 16, base * 4, 0.5)
        self.u3 = up_block(base * 8, base * 2)
        self.u4 = up_block(base * 4, base)
        self.out = nn.Sequential(nn.ConvTranspose2d(base * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.attention(self.d5(d4))
        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))
        return self.out(torch.cat([u4, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()

        def spectral_conv(in_c, out_c, stride=2, norm=True):
            layers: list[nn.Module] = [nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *spectral_conv(4, base, norm=False), *spectral_conv(base, base * 2),
            *spectral_conv(base * 2, base * 4), *spectral_conv(base * 4, base * 8, stride=1),
            nn.utils.spectral_norm(nn.Conv2d(base * 8, 1, 4, 1, 1)),
        )

    def forward(self, bmode: torch.Tensor, ceus: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([bmode, ceus], dim=1))


def ssim_index(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Differentiable global-window SSIM for tensors in [-1, 1]."""
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.0001 * 4.0, 0.0009 * 4.0
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def _expand_first_conv(conv: nn.Conv2d, in_channels: int = 6) -> nn.Conv2d:
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        if conv.weight.shape[1] == 3 and in_channels == 6:
            new.weight[:, :3] = conv.weight / 2
            new.weight[:, 3:] = conv.weight / 2
        else:
            mean = conv.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(mean.repeat(1, in_channels, 1, 1) / max(in_channels / conv.weight.shape[1], 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class MultiTaskClassifier(nn.Module):
    """Same six-channel capacity for every input condition; unused modality channels are zeroed."""
    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, dropout: float = 0.4):
        super().__init__()
        from torchvision import models

        weights = "DEFAULT" if pretrained else None
        if backbone in {"resnet18", "resnet50"}:
            constructor = getattr(models, backbone)
            base = constructor(weights=weights)
            base.conv1 = _expand_first_conv(base.conv1)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.gradcam_layer = base.layer4[-1]
        elif backbone in {"efficientnet_b0", "efficientnet_b7"}:
            base = getattr(models, backbone)(weights=weights)
            base.features[0][0] = _expand_first_conv(base.features[0][0])
            feature_dim = base.classifier[-1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
            self.gradcam_layer = base.features[-1]
        elif backbone in {"vit_b_16", "vit_l_32"}:
            base = getattr(models, backbone)(weights=weights)
            base.conv_proj = _expand_first_conv(base.conv_proj)
            feature_dim = base.heads.head.in_features
            base.heads = nn.Identity()
            self.backbone = base
            self.gradcam_layer = base.conv_proj
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.embedding = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({
            "Internal_Echo": nn.Linear(256, 4),
            "Morphology": nn.Linear(256, 1), "Boundary": nn.Linear(256, 1),
            "Solid": nn.Linear(256, 1), "Separation": nn.Linear(256, 1),
            "Nipple": nn.Linear(256, 1), "Blood_Flow": nn.Linear(256, 1),
            "Malignant": nn.Linear(256, 1),
        })

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.backbone(x))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {name: head(features) for name, head in self.heads.items()}


class MultiTaskCriterion(nn.Module):
    def __init__(self, mode: str = "unweighted", echo_weights=None, pos_weights=None, gamma: float = 2.0):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        self.register_buffer("echo_weights", torch.as_tensor(echo_weights, dtype=torch.float32) if echo_weights is not None else None)
        pos_weights = pos_weights or {}
        self.pos_weights = {k: float(v) for k, v in pos_weights.items()}
        self.echo_map = {0: 0, 1: 1, 3: 2, 4: 3}

    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        echo_target = torch.tensor([self.echo_map[int(x)] for x in targets["Internal_Echo"].detach().cpu().tolist()],
                                   device=outputs["Internal_Echo"].device)
        echo = F.cross_entropy(outputs["Internal_Echo"], echo_target,
                               weight=self.echo_weights if self.mode == "class_weighted" else None)
        losses = []
        for task, logits in outputs.items():
            if task == "Internal_Echo":
                continue
            target = targets[task].float().view_as(logits)
            pos_weight = None
            if self.mode == "class_weighted":
                pos_weight = torch.tensor(self.pos_weights.get(task, 1.0), device=logits.device)
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
            if self.mode == "focal":
                probability = torch.sigmoid(logits)
                pt = probability * target + (1 - probability) * (1 - target)
                raw = (1 - pt).pow(self.gamma) * raw
            losses.append(raw.mean())
        return echo + torch.stack(losses).mean()

