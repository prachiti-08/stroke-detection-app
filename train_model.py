import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import os

# -----------------------------
# Model Definition
# -----------------------------
class StrokeNet(nn.Module):

    def __init__(self):
        super(StrokeNet, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Fine-tune deeper layers
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

        # Ensure classifier is trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Training Pipeline
# -----------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Training ResNet18 model...")

    # -----------------------------
    # Transforms (FIXED: ImageNet normalization)
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # Dataset
    # -----------------------------
    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder("dataset/validate", transform=val_transform)
    test_dataset  = datasets.ImageFolder("dataset/test", transform=val_transform)

    print("Classes:", train_dataset.classes)

    # -----------------------------
    # DataLoaders (OPTIMIZED)
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = StrokeNet().to(device)

    # -----------------------------
    # Loss (optional class imbalance support can be added later)
    # -----------------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = optim.Adam([
        {"params": model.model.layer3.parameters(), "lr": 1e-4},
        {"params": model.model.layer4.parameters(), "lr": 1e-4},
        {"params": model.model.fc.parameters(), "lr": 1e-3}
    ])

    # -----------------------------
    # LR Scheduler (IMPORTANT STABILITY FIX)
    # -----------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    # -----------------------------
    # Training Loop
    # -----------------------------
    epochs = 7
    best_acc = 0.0

    os.makedirs("model", exist_ok=True)

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
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
        val_acc = correct / float(len(val_dataset))

        # Scheduler step
        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ---- SAVE BEST MODEL ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/best_model.pth")
            print("✅ Best model saved")

    # -----------------------------
    # Load Best Model Safely
    # -----------------------------
    print("\nLoading best model...")

    if os.path.exists("model/best_model.pth"):
        model.load_state_dict(torch.load("model/best_model.pth", map_location=device))

    model.eval()

    # -----------------------------
    # TEST EVALUATION
    # -----------------------------
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

    test_acc = correct / float(len(test_dataset))

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print("✅ Training complete")