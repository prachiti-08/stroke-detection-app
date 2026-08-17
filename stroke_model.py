import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class StrokeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer2.parameters():
            param.requires_grad = True

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(
                self.model.fc.in_features,
                3
            )
        )

    def forward(self, x):
        return self.model(x)