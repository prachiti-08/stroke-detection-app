import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class StrokeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze everything initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Fine-tune deeper layers
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace classification head
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            3
        )

    def forward(self, x):

        return self.model(x)
