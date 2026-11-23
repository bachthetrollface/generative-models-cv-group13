"""The ×4 networks in Ledig et al., Figure 4."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return x + self.branch(x)


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, 64, 9, padding=4), nn.PReLU(64))
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(16)))
        self.trunk = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64))
        upsample = []
        for _ in range(2):
            upsample.extend([nn.Conv2d(64, 256, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(64)])
        self.output = nn.Sequential(*upsample, nn.Conv2d(64, 3, 9, padding=4))

    def forward(self, lr):
        first = self.head(lr)
        return self.output(first + self.trunk(self.blocks(first)))


class Discriminator(nn.Module):
    """Return logits. Softplus/BCE-with-logits includes the paper's sigmoid."""
    def __init__(self):
        super().__init__()
        layers = []
        inputs = 3
        for i, outputs in enumerate((64, 64, 128, 128, 256, 256, 512, 512)):
            layers.append(nn.Conv2d(inputs, outputs, 3, stride=1 + i % 2, padding=1, bias=i == 0))
            if i:
                layers.append(nn.BatchNorm2d(outputs))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            inputs = outputs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1),
        )

    def forward(self, image):
        return self.classifier(self.features(image))


class VGGContent(nn.Module):
    def __init__(self, layer):
        super().__init__()
        end = {"vgg22": 9, "vgg54": 36}[layer]  # inclusive ReLU indices 8 and 35
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:end].eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, image):
        # Generator/target are BGR on the HR intensity scale [-1, 1].
        rgb = (image.flip(1) + 1) * 0.5
        return self.features((rgb - self.mean) / self.std) / 12.75

    def forward(self, prediction, target):
        with torch.no_grad():
            reference = self.encode(target)
        return F.mse_loss(self.encode(prediction), reference)
