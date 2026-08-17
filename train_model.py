import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_DIR = "dataset_new/train"
VAL_DIR = "dataset_new/validate"
TEST_DIR = "dataset_new/test"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 25

# Lower LR for pretrained backbone
BACKBONE_LR = 3e-5

# Higher LR for newly initialized classification head
FC_LR = 3e-4

WEIGHT_DECAY = 1e-4

# Label smoothing helps reduce overconfident predictions
LABEL_SMOOTHING = 0.05

# Early stopping
PATIENCE = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# DATASET PATH VERIFICATION
# ============================================================

print("\nChecking dataset paths...")

for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    print(
        f"{path} -> "
        f"{os.path.abspath(path)}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{os.path.abspath(path)}"
        )


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ------------------------------------------------------------
# Training augmentation
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ------------------------------------------------------------
# Validation / Test
# ------------------------------------------------------------

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_transform
)


print("\nClasses:", train_dataset.classes)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

print("\nClass mappings:")

print(
    "Train:",
    train_dataset.class_to_idx
)

print(
    "Val:  ",
    val_dataset.class_to_idx
)

print(
    "Test: ",
    test_dataset.class_to_idx
)


if (
    train_dataset.class_to_idx
    != val_dataset.class_to_idx
    or
    train_dataset.class_to_idx
    != test_dataset.class_to_idx
):

    raise ValueError(
        "\nERROR: Class mappings are different!"
    )


# ============================================================
# VERIFY CLASS COUNTS
# ============================================================

print("\nClass distribution:")

train_counts = np.bincount(
    train_dataset.targets,
    minlength=len(train_dataset.classes)
)

val_counts = np.bincount(
    val_dataset.targets,
    minlength=len(val_dataset.classes)
)

test_counts = np.bincount(
    test_dataset.targets,
    minlength=len(test_dataset.classes)
)


for i, class_name in enumerate(
    train_dataset.classes
):

    print(
        f"{class_name:12s} | "
        f"Train: {train_counts[i]:4d} | "
        f"Val: {val_counts[i]:4d} | "
        f"Test: {test_counts[i]:4d}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

NUM_WORKERS = 0

pin_memory = torch.cuda.is_available()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# ============================================================
# MODEL
# ============================================================

class StrokeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze everything first
        for param in self.model.parameters():

            param.requires_grad = False


        # Fine-tune deeper feature layers
        for param in self.model.layer2.parameters():

            param.requires_grad = True


        for param in self.model.layer3.parameters():

            param.requires_grad = True


        for param in self.model.layer4.parameters():

            param.requires_grad = True


        # Replace classification head
        self.model.fc = nn.Sequential(

            nn.Dropout(
                p=0.25
            ),

            nn.Linear(
                self.model.fc.in_features,
                3
            )
        )


    def forward(self, x):

        return self.model(x)


model = StrokeNet().to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Training classes are already close to balanced.
# The weights are therefore only a mild correction.

class_weights = (
    len(train_dataset)
    /
    (
        len(train_dataset.classes)
        *
        train_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


print("\nClass weights:")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):

    print(
        f"{class_name}: {weight.item():.4f}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    [
        {
            "params":
                model.model.layer2.parameters(),
            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.model.layer3.parameters(),
            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.model.layer4.parameters(),
            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.model.fc.parameters(),
            "lr":
                FC_LR
        }
    ],

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=1e-7
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_f1 = 0.0

epochs_without_improvement = 0

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_train_loss = 0.0

    train_predictions = []

    train_labels = []


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_train_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


        train_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    train_loss = (
        running_train_loss
        /
        len(train_dataset)
    )


    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )


    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    val_predictions = []

    val_labels = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            running_val_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


            val_labels.extend(
                labels
                .cpu()
                .numpy()
            )


    val_loss = (
        running_val_loss
        /
        len(val_dataset)
    )


    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )


    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro"
    )


    val_balanced_accuracy = (
        balanced_accuracy_score(
            val_labels,
            val_predictions
        )
    )


    # ========================================================
    # CURRENT LEARNING RATE
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1: "
        f"{train_f1:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1: "
        f"{val_f1:.4f}"
    )

    print(
        f"Val Balanced Accuracy: "
        f"{val_balanced_accuracy:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # VALIDATION CLASSIFICATION REPORT
    # ========================================================

    print("\nValidation Classification Report:")

    print(
        classification_report(
            val_labels,
            val_predictions,
            target_names=train_dataset.classes,
            digits=4,
            zero_division=0
        )
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_accuracy,

                "classes":
                    train_dataset.classes
            },

            MODEL_PATH
        )


        print(
            "\n✅ Best model saved!"
        )


    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_f1
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\n🛑 Early stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)

print(
    "LOADING BEST MODEL"
)

print("=" * 60)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    f"Best validation Macro F1: "
    f"{checkpoint['val_f1']:.4f}"
)


print(
    f"Best validation Accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)


print(
    f"Best epoch: "
    f"{checkpoint['epoch'] + 1}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_predictions = []

test_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )


        outputs = model(images)


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )


        test_labels.extend(
            labels
            .numpy()
        )


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)


test_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro"
)


test_balanced_accuracy = (
    balanced_accuracy_score(
        test_labels,
        test_predictions
    )
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "FINAL TEST RESULTS"
)

print("=" * 60)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)


print(
    f"Test Balanced Accuracy: "
    f"{test_balanced_accuracy:.4f}"
)


# ============================================================
# TEST CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)


print(
    classification_report(

        test_labels,

        test_predictions,

        target_names=
            train_dataset.classes,

        digits=4,

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# PER-CLASS TEST ACCURACY
# ============================================================

print(
    "\nPer-class accuracy:"
)


for i, class_name in enumerate(
    train_dataset.classes
):

    class_mask = (
        np.array(test_labels) == i
    )

    class_accuracy = (
        np.mean(
            np.array(test_predictions)[
                class_mask
            ]
            == i
        )
        if class_mask.sum() > 0
        else 0
    )


    print(
        f"{class_name:12s}: "
        f"{class_accuracy:.4f}"
    )


print("\n" + "=" * 60)

print(
    "✅ TRAINING COMPLETE"
)

print("=" * 60)