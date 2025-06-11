import torch.nn as nn
import timm  # EfficientNet-B3 backbone

class BaselineMLPHead(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.5, hidden_dim=256):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True)
        self.backbone.classifier = nn.Identity()  # remove default head

        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
