import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# STEP 1 - CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("data/fashion_mnist", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# STEP 2 - SELECT COMPUTATION DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("FLIPKART PRODUCT IMAGE CATEGORISER")
print("=" * 70)

print(f"\nUsing device: {device}")


# ============================================================
# STEP 3 - DEFINE CLASS NAMES
# ============================================================
#
# Fashion-MNIST contains exactly 10 classes.
#
# The numeric labels are:
#
# 0 -> T-shirt/top
# 1 -> Trouser
# 2 -> Pullover
# 3 -> Dress
# 4 -> Coat
# 5 -> Sandal
# 6 -> Shirt
# 7 -> Sneaker
# 8 -> Bag
# 9 -> Ankle boot
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

NUM_CLASSES = 10


# ============================================================
# STEP 4 - IMAGE PREPROCESSING
# ============================================================
#
# ResNet-18 was originally trained on ImageNet.
#
# ImageNet expects:
#
#   3 channels
#   224 x 224 images
#   ImageNet normalization
#
# Fashion-MNIST images are:
#
#   1 channel
#   28 x 28
#
# Therefore:
#
#   28x28 grayscale
#          |
#          v
#   224x224
#          |
#          v
#   grayscale -> 3 channels
#          |
#          v
#   ImageNet normalization
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


transform = transforms.Compose([

    # Resize Fashion-MNIST from 28x28 to 224x224
    transforms.Resize((224, 224)),

    # Convert grayscale image to 3 channels
    transforms.Grayscale(num_output_channels=3),

    # Convert PIL image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize using ImageNet statistics
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])


print("\nImage preprocessing:")
print("  Original image : 28 x 28 grayscale")
print("  Final image    : 224 x 224")
print("  Channels       : 3")
print("  Normalization  : ImageNet mean/std")


# ============================================================
# STEP 5 - LOAD THE FASHION-MNIST TEST DATASET
# ============================================================


print("\nLoading Fashion-MNIST test dataset...")

test_dataset = datasets.FashionMNIST(

    root="data/fashion_mnist",

    train=False,

    download=True,

    transform=transform

)


print(f"Test images: {len(test_dataset)}")


# ============================================================
# STEP 6 - CREATE TEST DATALOADER
# ============================================================

BATCH_SIZE = 64


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# STEP 7 - RECREATE RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18 architecture...")


weights = models.ResNet18_Weights.DEFAULT


resnet = models.resnet18(
    weights=weights
)


# ============================================================
# STEP 8 - FREEZE RESNET BACKBONE
# ============================================================
#
# NOT fine-tuning because validation accuracy
# was already: 89.82%
# which is above the required 80%.
# Therefore:
# Feature extraction = sufficient
# The backbone remains frozen.
# ============================================================

for parameter in resnet.parameters():

    parameter.requires_grad = False


# ============================================================
# STEP 9 - REMOVE ORIGINAL RESNET CLASSIFIER
# ============================================================
#
# Original ResNet-18:
#
#     features -> 512 -> ImageNet classifier -> 1000 classes
#
# We need:
#
#     features -> 512 -> our classifier -> 10 classes
# ============================================================

feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)


feature_extractor = feature_extractor.to(device)


# ============================================================
# STEP 10 - CREATE OUR CLASSIFIER HEAD
# ============================================================

classifier = nn.Linear(

    512,

    NUM_CLASSES

)


classifier = classifier.to(device)


# ============================================================
# STEP 11 - LOAD THE TRAINED CLASSIFIER HEAD
# ============================================================


classifier_head_path = (
    "models/classifier_head.pt"
)


if os.path.exists(classifier_head_path):

    print(
        "\nLoading saved classifier head..."
    )

    classifier.load_state_dict(
        torch.load(
            classifier_head_path,
            map_location=device,
            weights_only=True
        )
    )

else:

    print(
        "\nNo saved classifier head found."
    )

    print(
        "Re-training the classifier head from cached features..."
    )


    # --------------------------------------------------------
    # Load cached training features
    # --------------------------------------------------------

    train_cache = torch.load(
        "data/cached_features/train_features.pt",
        map_location="cpu",
        weights_only=True
    )


    train_features = train_cache["features"]

    train_labels = train_cache["labels"]


    print(
        "Training feature shape:",
        train_features.shape
    )


    # --------------------------------------------------------
    # Define loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(

        classifier.parameters(),

        lr=0.001

    )


    # --------------------------------------------------------
    # Train classifier head
    # --------------------------------------------------------

    EPOCHS = 10


    print("\nTraining classifier head...")


    for epoch in range(EPOCHS):

        classifier.train()


        # Shuffle training examples
        permutation = torch.randperm(
            len(train_features)
        )


        running_loss = 0.0


        for start in range(
            0,
            len(train_features),
            BATCH_SIZE
        ):

            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]


            batch_features = train_features[
                batch_indices
            ].to(device)


            batch_labels = train_labels[
                batch_indices
            ].to(device)


            optimizer.zero_grad()


            outputs = classifier(
                batch_features
            )


            loss = criterion(
                outputs,
                batch_labels
            )


            loss.backward()


            optimizer.step()


            running_loss += loss.item()


        average_loss = (
            running_loss /
            (len(train_features) / BATCH_SIZE)
        )


        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )


    torch.save(

        classifier.state_dict(),

        classifier_head_path

    )


    print(
        f"\nClassifier head saved to:"
        f"\n{classifier_head_path}"
    )


# ============================================================
# STEP 12 - CREATE COMPLETE MODEL
# ============================================================
#
# Architecture:
#
#     Fashion-MNIST image
#             |
#             v
#     preprocessing
#             |
#             v
#     ResNet-18
#             |
#             v
#     512 features
#             |
#             v
#     Linear classifier
#             |
#             v
#     10 Fashion-MNIST classes
# ============================================================


class FashionMNISTResNet18(nn.Module):

    def __init__(self):

        super().__init__()


        # Pretrained ResNet feature extractor
        self.feature_extractor = feature_extractor


        # Our 10-class classifier
        self.classifier = classifier


    def forward(self, x):

        # Extract ResNet features
        features = self.feature_extractor(x)


        # Flatten:
        #
        # [batch, 512, 1, 1]
        #
        # becomes:
        #
        # [batch, 512]

        features = torch.flatten(
            features,
            start_dim=1
        )


        # Class prediction
        output = self.classifier(
            features
        )


        return output


# Create complete model

model = FashionMNISTResNet18()


model = model.to(device)


model.eval()


# ============================================================
# STEP 13 - FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)


all_predictions = []
all_true_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)


        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_true_labels.extend(
            labels.numpy()
        )


# Convert to NumPy arrays

y_true = np.array(
    all_true_labels
)

y_pred = np.array(
    all_predictions
)


# ============================================================
# STEP 14 - CALCULATE TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_true,
    y_pred
)


print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# STEP 15 - CONFUSION MATRIX
# ============================================================
#
# Rows    = actual classes
# Columns = predicted classes
#
# Example:
#
# confusion_matrix[6][0]
#
# means:
#
# Actual Shirt
# predicted as T-shirt/top
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(NUM_CLASSES)

)


print("\n" + "=" * 70)
print("10 x 10 CONFUSION MATRIX")
print("=" * 70)


# Print column header

print(
    f"{'Actual / Pred':18s}",
    end=""
)


for name in CLASS_NAMES:

    print(
        f"{name[:10]:>11s}",
        end=""
    )


print()


# Print matrix

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:18s}",
        end=""
    )


    for value in row:

        print(
            f"{value:11d}",
            end=""
        )


    print()


# ============================================================
# STEP 16 - SAVE CONFUSION MATRIX
# ============================================================

np.savetxt(

    "imageclassifier_confusion_matrix.csv",

    cm,

    delimiter=",",

    fmt="%d"

)


print(
    "\nConfusion matrix saved to:"
    "\nimageclassifier_confusion_matrix.csv"
)


# ============================================================
# STEP 17 - CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    labels=np.arange(NUM_CLASSES),

    target_names=CLASS_NAMES,

    digits=4

)


print("\n" + "=" * 70)
print("PER-CLASS PRECISION AND RECALL")
print("=" * 70)

print(report)


# ============================================================
# STEP 18 - SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "imageclassifier_classification_report.txt",
    "w"
) as file:

    file.write(
        "Fashion-MNIST ResNet-18 Classification Report\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(report)


print(
    "Classification report saved to:"
    "\nimageclassifier_classification_report.txt"
)


# ============================================================
# STEP 19 - FIND THE TWO BIGGEST CONFUSION PAIRS
# ============================================================
#
# We only examine OFF-DIAGONAL cells.
#
# Diagonal:
#
#     correct prediction
#
# Off-diagonal:
#
#     incorrect prediction
#
# We find the largest incorrect counts.
# ============================================================

confusion_pairs = []


for actual in range(NUM_CLASSES):

    for predicted in range(NUM_CLASSES):

        # Ignore correct predictions
        if actual == predicted:
            continue


        count = cm[
            actual,
            predicted
        ]


        confusion_pairs.append(
            (
                count,
                actual,
                predicted
            )
        )


# Sort from largest to smallest

confusion_pairs.sort(
    reverse=True
)


print("\n" + "=" * 70)
print("TOP CONFUSION PAIRS")
print("=" * 70)


shown_pairs = 0


for count, actual, predicted in confusion_pairs:

    print(
        f"{CLASS_NAMES[actual]} "
        f"-> "
        f"{CLASS_NAMES[predicted]} "
        f": "
        f"{count} images"
    )


    shown_pairs += 1


    if shown_pairs == 10:
        break


# ============================================================
# STEP 20 - SAVE CONFUSION PAIR INFORMATION
# ============================================================

with open(
    "imageclassifier_confusion_pairs.txt",
    "w"
) as file:

    file.write(
        "Top Fashion-MNIST Confusion Pairs\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )


    for count, actual, predicted in confusion_pairs[:10]:

        file.write(
            f"{CLASS_NAMES[actual]} -> "
            f"{CLASS_NAMES[predicted]}: "
            f"{count} images\n"
        )


print(
    "\nConfusion-pair results saved to:"
    "\nimageclassifier_confusion_pairs.txt"
)


# ============================================================
# STEP 21 - SAVE THE COMPLETE MODEL
# ============================================================


MODEL_PATH = (
    "models/product_classifier.pt"
)


torch.save(

    model.state_dict(),

    MODEL_PATH

)


print(
    "\n" + "=" * 70
)

print(
    "MODEL ARTIFACT SAVED"
)

print(
    "=" * 70
)

print(
    f"\nSaved model:"
    f"\n{MODEL_PATH}"
)


# ============================================================
# STEP 22 - VERIFY THE SAVED MODEL
# ============================================================

print(
    "\nVerifying saved model..."
)


verification_model = FashionMNISTResNet18()


verification_model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=True

    )

)


verification_model = verification_model.to(
    device
)


verification_model.eval()


print(
    "Saved model loaded successfully!"
)


# ============================================================
# STEP 23 - DEFINE SINGLE IMAGE PREDICTION FUNCTION
# ============================================================


def predict_single_image(
    image_path,
    model,
    device
):
    """
    Predict the Fashion-MNIST category of one image.

    Parameters
    ----------
    image_path : str
        Path to the PNG/JPG image.

    model : torch.nn.Module
        Trained Fashion-MNIST ResNet-18 model.

    device : torch.device
        CPU, MPS, or CUDA device.

    Returns
    -------
    dict
        Dictionary containing:
        - category
        - confidence
        - class_index
    """

    from PIL import Image


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")


    # --------------------------------------------------------
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # Add batch dimension
    #
    # [3,224,224]
    #
    # becomes:
    #
    # [1,3,224,224]

    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        # Convert logits to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Highest probability

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )


    predicted_index = (
        predicted_index.item()
    )


    confidence = (
        confidence.item()
    )


    predicted_category = (
        CLASS_NAMES[predicted_index]
    )


    return {

        "category": predicted_category,

        "confidence": confidence,

        "class_index": predicted_index

    }


# ============================================================
# STEP 24 - EXPORT FIVE REAL TEST IMAGES
# ============================================================

from PIL import Image


print(
    "\n" + "=" * 70
)

print(
    "EXPORTING REAL TEST IMAGES"
)

print(
    "=" * 70
)


# Keep track of classes already selected

selected_classes = set()


sample_count = 0


for index in range(
    len(test_dataset)
):

    # Get raw Fashion-MNIST image
    #
    # test_dataset.data contains the original
    # 28x28 grayscale image.

    image_array = (
        test_dataset.data[index]
        .numpy()
    )


    # True class label

    true_label = int(
        test_dataset.targets[index]
    )


    # Prefer a different class each time

    if true_label in selected_classes:

        continue


    selected_classes.add(
        true_label
    )


    # Convert numeric label to readable name

    class_name = CLASS_NAMES[
        true_label
    ]


    # Create safe filename
    #
    # Example:
    #
    # 01_sneaker.png

    safe_class_name = (
        class_name
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


    filename = (
        f"{sample_count + 1:02d}_"
        f"{safe_class_name}.png"
    )


    output_path = os.path.join(

        "data",

        "sample_images",

        filename

    )


    # Convert NumPy array to PIL image

    image = Image.fromarray(
        image_array
    )


    # Save actual PNG file

    image.save(
        output_path
    )


    print(
        f"Saved: {output_path}"
    )


    sample_count += 1


    if sample_count == 5:

        break


# ============================================================
# STEP 25 - TEST THE PREDICTION FUNCTION
# ============================================================
#
# We will run the saved model against the exported images.
#
# This proves that:
#
#     actual PNG
#          ↓
#     preprocessing
#          ↓
#     saved model
#          ↓
#     prediction
#
# works correctly.
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TESTING SINGLE-IMAGE PREDICTION"
)

print(
    "=" * 70
)


sample_files = sorted(

    [
        file
        for file in os.listdir(
            "data/sample_images"
        )
        if file.lower().endswith(".png")
    ]

)


for filename in sample_files:

    image_path = os.path.join(

        "data",

        "sample_images",

        filename

    )


    result = predict_single_image(

        image_path,

        verification_model,

        device

    )


    print(
        f"\nImage: {filename}"
    )

    print(
        f"Predicted category: "
        f"{result['category']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )


# ============================================================
# STEP 26 - FINAL SUMMARY
# ============================================================


print(
    f"""
Final Test Accuracy:
    {test_accuracy * 100:.2f}%

Validation Accuracy:
    89.82%

Fine-tuning:
    NOT REQUIRED

Reason:
    Feature extraction already achieved 89.82%
    validation accuracy, which exceeds the 80%
    requirement.

Test Set:
    10,000 images

Classes:
    10

Model:
    Pretrained ResNet-18 + custom 10-class classifier

Model Artifact:
    {MODEL_PATH}

Confusion Matrix:
    imageclassifier_confusion_matrix.csv

Classification Report:
    imageclassifier_classification_report.txt

Confusion Pairs:
    imageclassifier_confusion_pairs.txt

Sample Images:
    data/sample_images/
"""
)

