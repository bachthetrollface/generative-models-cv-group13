"""The Ledig et al. and BasicSR ×4 super-resolution networks."""
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


class ResidualBlockNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class MSRResNet(nn.Module):
    """BasicSR's ×4 generator, operating on RGB in the [0, 1] intensity scale."""
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlockNoBN() for _ in range(16)))
        self.upconv1 = nn.Conv2d(64, 256, 3, padding=1)
        self.upconv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv_hr = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight)
                    layer.weight.mul_(0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, lr):
        features = self.body(F.leaky_relu(self.conv_first(lr), 0.1))
        for conv in (self.upconv1, self.upconv2):
            features = F.leaky_relu(F.pixel_shuffle(conv(features), 2), 0.1)
        residual = self.conv_last(F.leaky_relu(self.conv_hr(features), 0.1))
        return F.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False) + residual


def make_generator(arch):
    return {"srresnet": SRResNet, "msrresnet": MSRResNet}[arch]()


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


class VGGStyleDiscriminator(nn.Sequential):
    """BasicSR's discriminator for 128×128 RGB crops."""
    def __init__(self):
        layers = []
        inputs = 3
        for channels in (64, 128, 256, 512, 512):
            layers.append(nn.Conv2d(inputs, channels, 3, padding=1, bias=inputs == 3))
            if inputs != 3:
                layers.append(nn.BatchNorm2d(channels))
            layers.extend([nn.LeakyReLU(0.2, inplace=True),
                           nn.Conv2d(channels, channels, 4, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True)])
            inputs = channels
        super().__init__(*layers, nn.Flatten(), nn.Linear(512 * 4 * 4, 100),
                         nn.LeakyReLU(0.2, inplace=True), nn.Linear(100, 1))


class VGGContent(nn.Module):
    def __init__(self, layer, *, msr=False):
        super().__init__()
        self.msr = msr
        end = {"vgg22": 9, "vgg54": 36, "conv5_4": 35}[layer]
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:end].eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, image):
        rgb = image if self.msr else (image.flip(1) + 1) * 0.5
        features = self.features((rgb - self.mean) / self.std)
        return features if self.msr else features / 12.75

    def forward(self, prediction, target):
        with torch.no_grad():
            reference = self.encode(target)
        criterion = F.l1_loss if self.msr else F.mse_loss
        return criterion(self.encode(prediction), reference)
