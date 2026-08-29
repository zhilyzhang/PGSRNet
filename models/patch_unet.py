"""Lightweight dual-branch patch refiner used by PGSRNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # torchvision < 0.13
    ResNet18_Weights = None
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise ImportError("torchvision is required by DualBranchLiteUNetV2") from exc


def init_kaiming(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SobelHF(nn.Module):
    """Fixed Sobel edge magnitude for a single-channel map."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(dtype=x.dtype)
        ky = self.ky.to(dtype=x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class CoordAtt(nn.Module):
    """Coordinate attention with separate height and width gates."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat((x_h, x_w), dim=2))))
        x_h, x_w = torch.split(y, (height, width), dim=2)
        gate_h = torch.sigmoid(self.conv_h(x_h))
        gate_w = torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))
        return x * gate_h * gate_w


def _build_resnet18(use_imagenet: bool) -> nn.Module:
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
        return resnet18(weights=weights)
    return resnet18(pretrained=use_imagenet)


class ResNet18_SmallObj_Encoder(nn.Module):
    """ResNet-18 encoder retaining full resolution until layer2."""

    def __init__(self, use_imagenet: bool = True, layer3_blocks: int = 1) -> None:
        super().__init__()
        if layer3_blocks < 1:
            raise ValueError("layer3_blocks must be at least 1")

        backbone = _build_resnet18(use_imagenet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem.apply(init_kaiming)

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = nn.Sequential(*list(backbone.layer3.children())[:layer3_blocks])
        self.layer3[0].conv1.stride = (1, 1)
        if self.layer3[0].downsample is not None:
            self.layer3[0].downsample[0].stride = (1, 1)

        # Kept for checkpoint compatibility with the original implementation.
        self.layer4 = nn.Identity()
        self.fc = nn.Identity()

    def forward(self, x: torch.Tensor):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return c1, c2, c3, c4


class SimpleConcatFuse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((first, second), dim=1))


class PixelShuffleUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CoordAtt(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv_fuse(torch.cat((x, skip), dim=1))


class SimpleBMSRRefiner(nn.Module):
    """Patch-level boundary refinement using prediction, intensity, and edge cues."""

    def __init__(self, image_channels: int = 3, mid_channels: int = 64) -> None:
        super().__init__()
        self.to_gray = nn.Conv2d(image_channels, 1, 1, bias=False)
        nn.init.constant_(self.to_gray.weight, 1.0 / image_channels)
        self.to_gray.requires_grad_(False)
        self.hf = SobelHF()
        self.fuse = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid_channels, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.fuse.apply(init_kaiming)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        gray = self.to_gray(image)
        delta = self.out(self.fuse(torch.cat((logits, gray, self.hf(gray)), dim=1)))
        return logits + self.alpha * delta


class DualBranchLiteUNetV2(nn.Module):
    """Refine a five-channel patch: RGB image + coarse probability + edge.

    The context branch consumes the last image channel, coarse probability, and
    edge. Infrared RGB inputs normally contain three replicated gray channels,
    so this is equivalent to gray + probability + edge while retaining the
    original checkpoint-compatible five-channel interface.
    """

    def __init__(self, use_boundary_refine: bool = True, use_imagenet: bool = True) -> None:
        super().__init__()
        self.use_boundary_refine = bool(use_boundary_refine)
        self.enc_img = ResNet18_SmallObj_Encoder(use_imagenet=use_imagenet)
        self.enc_ctx = ResNet18_SmallObj_Encoder(use_imagenet=use_imagenet)

        self.fuse_c1 = SimpleConcatFuse(128, 64)
        self.fuse_c2 = SimpleConcatFuse(128, 64)
        self.fuse_c3 = SimpleConcatFuse(256, 128)
        self.fuse_c4 = SimpleConcatFuse(512, 256)
        self.merge_bottom = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up_block = PixelShuffleUpBlock(128, 64, 64)
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CoordAtt(64),
        )
        self.head = nn.Conv2d(64, 1, 1)
        self.boundary_refiner = SimpleBMSRRefiner() if self.use_boundary_refine else None

        # Initialize only newly introduced layers; do not overwrite ImageNet weights.
        for module in (
            self.fuse_c1,
            self.fuse_c2,
            self.fuse_c3,
            self.fuse_c4,
            self.merge_bottom,
            self.up_block,
            self.final_conv,
            self.head,
        ):
            module.apply(init_kaiming)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 5:
            raise ValueError(f"expected input [B,5,H,W], got {tuple(x.shape)}")
        image = x[:, :3]
        context = x[:, -3:]

        img_feats = self.enc_img(image)
        ctx_feats = self.enc_ctx(context)
        fused = [
            layer(img_feat, ctx_feat)
            for layer, img_feat, ctx_feat in zip(
                (self.fuse_c1, self.fuse_c2, self.fuse_c3, self.fuse_c4),
                img_feats,
                ctx_feats,
            )
        ]
        f1, f2, f3, f4 = fused
        x = self.merge_bottom(torch.cat((f4, f3), dim=1))
        x = self.up_block(x, f2)
        logits_raw = self.head(self.final_conv(torch.cat((x, f1), dim=1)))
        logits_ref = (
            self.boundary_refiner(image, logits_raw)
            if self.boundary_refiner is not None
            else logits_raw
        )
        return {"logits_raw": logits_raw, "logits_ref": logits_ref}


if __name__ == "__main__":
    model = DualBranchLiteUNetV2(use_imagenet=False)
    output = model(torch.randn(2, 5, 64, 64))
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {params / 1e6:.3f} M")
    print({name: tuple(value.shape) for name, value in output.items()})
