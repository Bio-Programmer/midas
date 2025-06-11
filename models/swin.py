import torch.nn as nn
import timm

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)