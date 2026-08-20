
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1 - CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("data/fashion_mnist", exist_ok=True)
os.makedirs("data/cached_features", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# STEP 2 - DEVICE CONFIGURATION
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}")

# ============================================================
# STEP 3 - CLASS LABELS
# ============================================================

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# STEP 4 - IMAGE PREPROCESSING
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

print("\nImage preprocessing:")
print("- Resize: 224 x 224")
print("- Channels: 1 → 3")
print("- Normalization: ImageNet mean/std")

# ============================================================
# STEP 5 - DOWNLOAD FASHION-MNIST
# ============================================================

print("\nDownloading Fashion-MNIST...")

raw_train = datasets.FashionMNIST(
    root="data/fashion_mnist",
    train=True,
    download=True
)

raw_test = datasets.FashionMNIST(
    root="data/fashion_mnist",
    train=False,
    download=True
)

print(f"Training images : {len(raw_train)}")
print(f"Test images     : {len(raw_test)}")

# ============================================================
# STEP 6 - CREATE TRANSFORMED DATASETS
# ============================================================

full_train_dataset = datasets.FashionMNIST(
    root="data/fashion_mnist",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data/fashion_mnist",
    train=False,
    download=False,
    transform=transform
)

# ============================================================
# STEP 7 - STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

targets = np.array(raw_train.targets)
indices = np.arange(len(targets))

train_indices, val_indices = train_test_split(
    indices,
    test_size=6000,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

print("\nDataset split:")
print(f"Training   : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")

# ============================================================
# STEP 8 - VERIFY STRATIFICATION
# ============================================================

def print_distribution(name, idx):
    labels = targets[idx]
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    print(f"\n{name} class distribution:")

    for i, count in enumerate(counts):
        pct = count / len(labels) * 100
        print(f"{CLASS_NAMES[i]:15s}: {count:5d} ({pct:5.2f}%)")

print_distribution("Training", train_indices)
print_distribution("Validation", val_indices)

# ============================================================
# STEP 9 - DATALOADERS
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# STEP 10 - LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)

# Freeze backbone
for param in resnet.parameters():
    param.requires_grad = False

# Remove classifier to obtain 512-dimensional features
feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

print("Backbone frozen.")
print("Feature dimension: 512")

# ============================================================
# STEP 11 - FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            features = model(images)

            features = torch.flatten(
                features,
                start_dim=1
            )

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    return features, labels

# ============================================================
# STEP 12 - EXTRACT TRAINING FEATURES
# ============================================================

print("\nExtracting training features...")

train_features, train_labels = extract_features(
    feature_extractor,
    train_loader,
    device
)

print("Training feature shape:", train_features.shape)

# ============================================================
# STEP 13 - EXTRACT VALIDATION FEATURES
# ============================================================

print("\nExtracting validation features...")

val_features, val_labels = extract_features(
    feature_extractor,
    val_loader,
    device
)

print("Validation feature shape:", val_features.shape)

# ============================================================
# STEP 14 - CACHE FEATURES
# ============================================================

torch.save(
    {
        "features": train_features,
        "labels": train_labels
    },
    "data/cached_features/train_features.pt"
)

torch.save(
    {
        "features": val_features,
        "labels": val_labels
    },
    "data/cached_features/val_features.pt"
)

print("\nCached feature files created.")

# ============================================================
# STEP 15 - CLASSIFIER HEAD
# ============================================================

classifier = nn.Linear(
    512,
    NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)

print("\nClassifier head:")
print("Input features : 512")
print("Output classes : 10")
print("Optimizer      : Adam")
print("Learning rate  : 0.001")

# ============================================================
# STEP 16 - TRAIN CLASSIFIER HEAD
# ============================================================

EPOCHS = 10

print("\nTraining classifier head...\n")

for epoch in range(EPOCHS):

    classifier.train()

    permutation = torch.randperm(
        len(train_features)
    )

    running_loss = 0.0

    for i in range(0, len(train_features), BATCH_SIZE):

        idx = permutation[i:i+BATCH_SIZE]

        batch_features = train_features[idx].to(device)
        batch_labels = train_labels[idx].to(device)

        optimizer.zero_grad()

        outputs = classifier(batch_features)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (len(train_features) / BATCH_SIZE)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}"
    )

# ============================================================
# STEP 17 - SAVE CLASSIFIER HEAD
# ============================================================

classifier_path = (
    "models/classifier_head.pt"
)


torch.save(

    classifier.state_dict(),

    classifier_path

)


print(
    "\nClassifier head saved to:"
)

print(
    classifier_path
)

# ============================================================
# STEP 18 - VALIDATION ACCURACY
# ============================================================

def calculate_accuracy(model, features, labels):

    model.eval()

    with torch.no_grad():

        outputs = model(features.to(device))

        predictions = torch.argmax(outputs, dim=1)

        accuracy = (
            predictions == labels.to(device)
        ).float().mean().item()

    return accuracy

val_accuracy = calculate_accuracy(
    classifier,
    val_features,
    val_labels
)

print("\n" + "=" * 70)
print("FEATURE EXTRACTION RESULTS")
print("=" * 70)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# ============================================================
# STEP 19 - TRAINABLE PARAMETERS
# ============================================================

trainable = sum(
    p.numel()
    for p in classifier.parameters()
)

total = sum(
    p.numel()
    for p in resnet.parameters()
)

print("\nModel summary")
print("-" * 40)
print(f"Total ResNet parameters : {total:,}")
print(f"Trainable parameters    : {trainable:,}")
