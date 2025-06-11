# models/baseline_model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class BaselineModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # load pre-trained efficient net B3 with weights
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b3(weights=weights)
        # replace classifier with 8-class output
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)