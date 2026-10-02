from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNet18UNet(nn.Module):
    """ImageNet-pretrained ResNet-18 encoder with a paired-translation decoder."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.d4 = DecoderBlock(512, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d2 = DecoderBlock(128, 64, 64)
        self.d1 = DecoderBlock(64, 64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh(),
        )

    def encoder_parameters(self):
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from module.parameters()

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        x = gray.repeat(1, 3, 1, 1)
        e0 = self.stem(x)
        e1 = self.layer1(self.pool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return self.final(self.d1(self.d2(self.d3(self.d4(e4, e3), e2), e1), e0))


class SwinTinyUNet(nn.Module):
    """Swin-Tiny ImageNet encoder with a U-Net decoder for paired translation."""

    def __init__(self, image_size: int = 256, pretrained: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Swin-Tiny baseline requires timm; install requirements_additional.txt") from exc
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained, features_only=True, img_size=image_size
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) != 4:
            raise RuntimeError(f"Unexpected Swin feature pyramid: {channels}")
        self.channels = channels
        self.d3 = DecoderBlock(channels[3], channels[2], 384)
        self.d2 = DecoderBlock(384, channels[1], 192)
        self.d1 = DecoderBlock(192, channels[0], 96)
        self.up2 = DecoderBlock(96, 0, 64)
        self.up1 = DecoderBlock(64, 0, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    @staticmethod
    def _nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Cannot identify channel dimension in feature shape {tuple(feature.shape)}")

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        features = self.encoder(gray.repeat(1, 3, 1, 1))
        f0, f1, f2, f3 = [self._nchw(x, c) for x, c in zip(features, self.channels)]
        return self.final(self.up1(self.up2(self.d1(self.d2(self.d3(f3, f2), f1), f0))))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [4, 64, 128, 256, 512]
        layers = []
        for index in range(4):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], 4, 2 if index < 3 else 1, 1))
            if index:
                layers.append(nn.InstanceNorm2d(channels[index + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(512, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, bmode: torch.Tensor, ceus: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([bmode, ceus], dim=1))


def build_generator(name: str, image_size: int, pretrained: bool = False) -> nn.Module:
    if name == "resnet18_unet_transfer":
        return ResNet18UNet(pretrained=pretrained)
    if name == "swin_tiny_unet":
        return SwinTinyUNet(image_size=image_size, pretrained=pretrained)
    raise ValueError(f"Unknown comparison model: {name}")


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def ssim_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x, mu_y = F.avg_pool2d(x, 11, 1, 5), F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.mean()
