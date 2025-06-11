import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

class MLPFusionModel(nn.Module):
    def __init__(self, num_classes=8, metadata_dim=3, hidden_dim=256, p1=0.5, p2=0.5):
        super().__init__()
        self.backbone = efficientnet_b3(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(in_features + metadata_dim, hidden_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, metadata):
        features = self.backbone(images)
        fused = torch.cat([features, metadata], dim=1)
        x = self.fusion(fused)
        return self.classifier(x)
