"""Model architectures used by the handwriting screening pipeline.

The definitions are extracted from the training notebooks that produced the
checked-in checkpoints. They are image-level, three-class classifiers; the
existing segmentation pipeline supplies the handwriting crops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return x * torch.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        attention = torch.cat(
            [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True).values],
            1,
        )
        return x * torch.sigmoid(self.conv(attention))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class FeatureAlign(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x, target_size):
        x = self.proj(x)
        if x.shape[-2:] == target_size:
            return x
        return F.interpolate(x, target_size, mode="bilinear", align_corners=False)


class MultiLevelFusion(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int] = (40, 40, 96),
        fusion_channels: int = 128,
    ):
        super().__init__()
        self.align1 = FeatureAlign(channels[0], fusion_channels)
        self.align2 = FeatureAlign(channels[1], fusion_channels)
        self.align3 = FeatureAlign(channels[2], fusion_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * 3, fusion_channels * 2, 1, bias=False),
            nn.BatchNorm2d(fusion_channels * 2),
            nn.Hardswish(inplace=True),
            nn.Conv2d(fusion_channels * 2, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Hardswish(inplace=True),
        )
        self.cbam = CBAM(fusion_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(fusion_channels * 3, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
        )

    def forward(self, features):
        f1, f2, f3 = features
        target_size = f1.shape[-2:]
        merged = torch.cat(
            [
                self.align1(f1, target_size),
                self.align2(f2, target_size),
                self.align3(f3, target_size),
            ],
            1,
        )
        return F.relu(
            self.cbam(self.fuse(merged)) + self.residual(merged), inplace=True
        )


class SpatialPyramidPooling(nn.Module):
    def __init__(self, channels: int = 128, pool_sizes: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        return torch.cat(
            [
                F.adaptive_avg_pool2d(x, (size, size)).flatten(1)
                for size in self.pool_sizes
            ],
            1,
        )


class StandardInceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = (
            out_channels // 3,
            out_channels // 3,
            out_channels - 2 * (out_channels // 3),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[0], 1, bias=False),
            nn.BatchNorm2d(branch_channels[0]),
            nn.Hardswish(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels[1]),
            nn.Hardswish(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[2], 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels[2]),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], 1)


class InceptionSEResidual(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.inception = StandardInceptionBlock(channels, channels)
        reduced = max(channels // ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.inception(x)
        return F.hardswish(y * self.se(self.pool(y)) + x, inplace=True)


class ModelDV4(nn.Module):
    """Model D v4: MobileNetV3-Small trunk + inception/SE fusion + CBAM."""

    def __init__(self, pretrained: bool = False, num_classes: int = 3):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_small
        except Exception as exc:
            raise RuntimeError(
                "A compatible torchvision installation is required for Model D v4."
            ) from exc
        backbone = mobilenet_v3_small(weights=None)
        first = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            first.kernel_size,
            first.stride,
            first.padding,
            bias=False,
        )
        self.features = backbone.features
        self.inception_40_a = InceptionSEResidual(40)
        self.inception_40_b = InceptionSEResidual(40)
        self.inception_96 = InceptionSEResidual(96)
        self.fusion = MultiLevelFusion(channels=(40, 40, 96), fusion_channels=128)
        self.spp = SpatialPyramidPooling(channels=128, pool_sizes=(1, 2, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 21, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        features = [None, None, None]
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 4:
                features[0] = self.inception_40_a(x)
            elif index == 6:
                features[1] = self.inception_40_b(x)
            elif index == 9:
                features[2] = self.inception_96(x)
        return self.classifier(self.spp(self.fusion(features)))


class MobileNetV3Small(nn.Module):
    """Model A: grayscale MobileNetV3-Small with a three-class head."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_small
        except Exception as exc:
            raise RuntimeError(
                "A compatible torchvision installation is required for "
                "MobileNetV3-Small."
            ) from exc
        backbone = mobilenet_v3_small(weights=None)
        first = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            first.kernel_size,
            first.stride,
            first.padding,
            dilation=first.dilation,
            groups=first.groups,
            bias=False,
        )
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, num_classes
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ViTB16(nn.Module):
    """ViT-B/16 classifier with the torchvision state-dict layout."""

    def __init__(self, num_classes: int = 3, image_size: int = 224):
        super().__init__()
        try:
            from torchvision.models import vit_b_16
        except Exception as exc:
            raise RuntimeError(
                "A compatible torchvision installation is required for ViT-B/16."
            ) from exc
        backbone = vit_b_16(weights=None)
        backbone.heads.head = nn.Linear(
            backbone.heads.head.in_features, num_classes
        )

        # Keep the original torchvision attribute names so the checkpoint's
        # class_token/conv_proj/encoder/heads keys load without renaming.
        self.class_token = backbone.class_token
        self.conv_proj = backbone.conv_proj
        self.encoder = backbone.encoder
        self.heads = backbone.heads
        self.hidden_dim = backbone.hidden_dim
        self.patch_size = backbone.patch_size
        self.image_size = image_size

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"ViT-B/16 expects {self.image_size}x{self.image_size} input; "
                f"got {h}x{w}."
            )
        n_h, n_w = h // p, w // p
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        return x.permute(0, 2, 1)

    def forward(self, x):
        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.encoder(x)
        return self.heads(x[:, 0])


ARCHITECTURE_BUILDERS = {
    "model_d_v4": ModelDV4,
    "vit_b16": ViTB16,
    "mobilenet_v3_small": MobileNetV3Small,
}


def build_architecture(architecture: str, num_classes: int = 3) -> nn.Module:
    """Construct a supported architecture without loading any weights."""
    try:
        builder = ARCHITECTURE_BUILDERS[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(
            f"Unknown architecture {architecture!r}. Supported: {supported}."
        ) from exc
    return builder(num_classes=num_classes)
