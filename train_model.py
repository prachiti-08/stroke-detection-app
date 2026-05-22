import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import os

# -----------------------------
# CBAM Attention Block
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size,
                               padding=kernel_size//2,
                               bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):

        x = x * self.ca(x)
        x = x * self.sa(x)

        return x


# -----------------------------
# Improved StrokeNet
# -----------------------------
class StrokeNet(nn.Module):

    def __init__(self):
        super(StrokeNet, self).__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.cbam = CBAM(512)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

best_acc = 0
patience = 5
counter = 0 
# -----------------------------
# Training
# -----------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training ResNet18 model...")

    train_transform = transforms.Compose([

    transforms.Resize((384,384)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05,0.05),
        scale=(0.95,1.05)
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3,[0.5]*3)
])

    val_transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder("dataset/validate", transform=val_transform)
    test_dataset  = datasets.ImageFolder("dataset/test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8)
    test_loader  = DataLoader(test_dataset, batch_size=8)

    print("Classes:", train_dataset.classes)

    model = StrokeNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW([
    {"params": model.features[-2].parameters(), "lr": 1e-4},
    {"params": model.features[-1].parameters(), "lr": 1e-4},
    {"params": model.cbam.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

    epochs = 15
    best_acc = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_dataset)

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:

          best_acc = val_acc
          counter = 0

          os.makedirs("model", exist_ok=True)

          torch.save(
          model.state_dict(),
          "model/best_model.pth"
    )

        else:
            counter += 1

        if counter >= patience:
         print("Early stopping triggered")
        break

    print("Loading best model...")

    model.load_state_dict(torch.load("model/best_model.pth"))
    model.eval()

    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    test_acc = correct / len(test_dataset)

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print("✅ Training complete")