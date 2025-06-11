# models/fusion_model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class FusionModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes=8):
        super().__init__()

        weights = EfficientNet_B3_Weights.DEFAULT
        self.backbone = efficientnet_b3(weights=weights)
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()

        # We'll infer image feature dim in forward()
        self.fusion_head = nn.Sequential(
            nn.Linear(1536 + num_metadata_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, metadata):
        img_feat = self.backbone(img)  # (B, 1536)
        fused = torch.cat([img_feat, metadata], dim=1)
        return self.fusion_head(fused)