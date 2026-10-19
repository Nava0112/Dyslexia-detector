from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASS_NAMES, IMAGE_SIZE


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
        return x * torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        attention = torch.cat(
            [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True).values], 1
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
        return F.interpolate(x, target_size, mode="bilinear", align_corners=False) if x.shape[-2:] != target_size else x


class MultiLevelFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.align = nn.ModuleList([FeatureAlign(channels) for channels in (40, 40, 96)])
        self.fuse = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.Hardswish(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.Hardswish(inplace=True),
        )
        self.cbam = CBAM(128)
        self.residual = nn.Sequential(nn.Conv2d(384, 128, 1, bias=False), nn.BatchNorm2d(128))

    def forward(self, features):
        target_size = features[0].shape[-2:]
        merged = torch.cat([layer(feature, target_size) for layer, feature in zip(self.align, features)], 1)
        return F.relu(self.cbam(self.fuse(merged)) + self.residual(merged), inplace=True)


class SpatialPyramidPooling(nn.Module):
    def forward(self, x):
        return torch.cat([F.adaptive_avg_pool2d(x, (size, size)).flatten(1) for size in (1, 2, 4)], 1)


class InceptionSEResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        parts = (channels // 3, channels // 3, channels - 2 * (channels // 3))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, parts[0], 1, bias=False), nn.BatchNorm2d(parts[0]), nn.Hardswish(inplace=True)),
            nn.Sequential(nn.Conv2d(channels, parts[1], 3, padding=1, bias=False), nn.BatchNorm2d(parts[1]), nn.Hardswish(inplace=True)),
            nn.Sequential(nn.Conv2d(channels, parts[2], 5, padding=2, bias=False), nn.BatchNorm2d(parts[2]), nn.Hardswish(inplace=True)),
        ])
        reduced = max(channels // 8, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, reduced, 1), nn.ReLU(inplace=True), nn.Conv2d(reduced, channels, 1), nn.Sigmoid())

    def forward(self, x):
        y = torch.cat([branch(x) for branch in self.branches], 1)
        return F.hardswish(y * self.se(y) + x, inplace=True)


class ModelDV4(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_small
        except Exception as exc:
            raise RuntimeError("A compatible torchvision installation is required for Model D v4.") from exc
        backbone = mobilenet_v3_small(weights=None)
        first = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride, first.padding, bias=False)
        self.features = backbone.features
        self.inception_40_a = InceptionSEResidual(40)
        self.inception_40_b = InceptionSEResidual(40)
        self.inception_96 = InceptionSEResidual(96)
        self.fusion = MultiLevelFusion()
        self.spp = SpatialPyramidPooling()
        self.classifier = nn.Sequential(nn.Linear(128 * 21, 1024), nn.Hardswish(inplace=True), nn.Dropout(0.5), nn.Linear(1024, num_classes))

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


def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    side = max(gray.shape)
    square = np.zeros((side, side), dtype=gray.dtype)
    y_offset, x_offset = (side - gray.shape[0]) // 2, (side - gray.shape[1]) // 2
    square[y_offset:y_offset + gray.shape[0], x_offset:x_offset + gray.shape[1]] = gray
    resized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    normalized = (enhanced.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(normalized).unsqueeze(0)


class Classifier:
    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelDV4().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, crops):
        if not crops:
            return []
        batch = torch.stack([preprocess_crop(crop.image).squeeze(0) for crop in crops]).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        results = []
        for crop, probs in zip(crops, probabilities):
            index = int(np.argmax(probs))
            results.append({
                "crop_id": crop.crop_id, "x1": crop.x1, "y1": crop.y1, "x2": crop.x2, "y2": crop.y2,
                "width": crop.x2 - crop.x1, "height": crop.y2 - crop.y1,
                "predicted_class": CLASS_NAMES[index], "confidence": round(float(probs[index]), 4),
                "probabilities": {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)},
            })
        return results
