"""
Part 3 - Real tools for Part 1 and Part 2.

IMPORTANT:
These functions load and call the actual saved model artifacts.

Part 1:
    models/return_risk_model.pkl

Part 2:
    models/product_classifier.pt

No prediction is hardcoded.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import v2


# ============================================================
# PROJECT PATHS
# ============================================================

# __file__ points to this file:
#
#     src/tools.py
#
# .parent gives:
#
#     src/
#
# .parent.parent gives:
#
#     flipkart-support-assistant/
#
ROOT = Path(__file__).resolve().parent.parent


# Part 1 saved model
RETURN_MODEL_PATH = (
    ROOT / "models" / "return_risk_model.pkl"
)

# Part 1 Random Forest F1-maximizing threshold
RETURN_THRESHOLD_PATH = (
    ROOT / "models" / "return_risk_threshold.txt"
)

# Part 2 saved model
IMAGE_MODEL_PATH = (
    ROOT / "models" / "product_classifier.pt"
)


# ============================================================
# PART 1 - RETURN RISK TOOL
# ============================================================

def load_return_risk_model():
    """
    Load the real Part 1 Random Forest pipeline.

    The model is loaded from:

        models/return_risk_model.pkl
    """

    if not RETURN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Return-risk model not found: "
            f"{RETURN_MODEL_PATH}"
        )

    model = joblib.load(
        RETURN_MODEL_PATH
    )

    return model


def load_rf_threshold():
    """
    Load t*_rf.

    This is the F1-maximizing threshold calculated from
    the Random Forest's own predict_proba() output.
    """

    if not RETURN_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Return-risk threshold not found: "
            f"{RETURN_THRESHOLD_PATH}"
        )

    with open(
        RETURN_THRESHOLD_PATH,
        "r"
    ) as f:

        threshold = float(
            f.read().strip()
        )

    return threshold


def check_return_risk(
    order_features: dict
) -> dict:
    """
    Check return risk for one order.

    This function calls the actual saved Part 1 model.

    Parameters
    ----------
    order_features : dict
        Features representing one order.

    Returns
    -------
    dict
        Contains:

        return_probability
        risk_bucket
        t_rf
        low_cutoff
        high_cutoff
    """

    # --------------------------------------------------------
    # Load the actual trained Part 1 model
    # --------------------------------------------------------

    model = load_return_risk_model()

    # --------------------------------------------------------
    # Load the Random Forest threshold
    # --------------------------------------------------------

    t_rf = load_rf_threshold()

    # --------------------------------------------------------
    # Convert dictionary to one-row DataFrame
    # --------------------------------------------------------

    order_df = pd.DataFrame(
        [order_features]
    )

    # --------------------------------------------------------
    # Call the actual saved model
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        order_df
    )

    # Probability of returned = 1
    return_probability = float(
        probabilities[0][1]
    )

    # --------------------------------------------------------
    # Risk buckets
    #
    # Low:
    #     probability < t*_rf
    #
    # Medium:
    #     t*_rf <= probability < t*_rf + 0.15
    #
    # High:
    #     probability >= t*_rf + 0.15
    # --------------------------------------------------------

    low_cutoff = t_rf

    high_cutoff = t_rf + 0.15

    if return_probability < low_cutoff:

        risk_bucket = "Low"

    elif return_probability >= high_cutoff:

        risk_bucket = "High"

    else:

        risk_bucket = "Medium"

    return {
        "return_probability": round(
            return_probability,
            6
        ),

        "risk_bucket": risk_bucket,

        "t_rf": round(
            t_rf,
            6
        ),

        "low_cutoff": round(
            low_cutoff,
            6
        ),

        "high_cutoff": round(
            high_cutoff,
            6
        )
    }


# =========================================================
# PART 2 - PRODUCT IMAGE CLASSIFIER
# =========================================================

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
    "Ankle boot"
]


# =========================================================
# RECREATE THE EXACT MODEL ARCHITECTURE USED IN PART 2
# =========================================================

def build_product_classifier():
    """
    Recreates the same ResNet-18 architecture used during
    Part 2 training.

    The saved model contains:
        feature_extractor.*
        classifier.*

    Therefore we must recreate those exact layer names.
    """

    # Create ResNet-18 without downloading pretrained weights.
    backbone = models.resnet18(weights=None)

    # -----------------------------------------------------
    # Remove the original ResNet classification layer.
    #
    # ResNet-18 normally produces:
    #
    #     512 features
    #
    # before its original fc layer.
    # -----------------------------------------------------

    feature_extractor = torch.nn.Sequential(
        *list(backbone.children())[:-1]
    )

    # -----------------------------------------------------
    # The feature extractor outputs:
    #
    # [batch_size, 512, 1, 1]
    #
    # We flatten it to:
    #
    # [batch_size, 512]
    # -----------------------------------------------------

    class ProductClassifier(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.feature_extractor = feature_extractor

            # 512 ResNet features -> 10 Fashion-MNIST classes
            self.classifier = torch.nn.Linear(
                512,
                10
            )

        def forward(self, x):

            x = self.feature_extractor(x)

            x = torch.flatten(
                x,
                start_dim=1
            )

            x = self.classifier(x)

            return x

    return ProductClassifier()


# =========================================================
# LOAD SAVED PRODUCT CLASSIFIER
# =========================================================

def load_product_classifier():
    """
    Load the actual Part 2 trained model.

    This function loads:
        models/product_classifier.pt

    It does NOT create predictions manually.
    It uses the real saved model weights.
    """

    if not IMAGE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Product classifier not found: "
            f"{IMAGE_MODEL_PATH}"
        )

    # Recreate EXACT architecture used during Part 2.
    model = build_product_classifier()

    # Load saved weights.
    checkpoint = torch.load(
        IMAGE_MODEL_PATH,
        map_location="cpu"
    )

    # -----------------------------------------------------
    # Our Part 2 artifact should be a state_dict.
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        # Some training scripts save:
        #
        # torch.save(model.state_dict(), path)
        #
        # In that case the keys directly contain:
        #
        # feature_extractor.*
        # classifier.*
        #
        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        raise ValueError(
            "product_classifier.pt does not contain "
            "a supported state_dict."
        )

    # -----------------------------------------------------
    # Load the actual trained weights.
    # -----------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to("cpu")

    # Evaluation mode.
    model.eval()

    return model




# ============================================================
# IMAGE PREPROCESSING
# ============================================================

"""
Part 2 preprocessing:

1. Convert grayscale -> 3 channels
2. Resize -> 224 x 224
3. Convert to float tensor
4. Normalize using ImageNet mean/std

This matches the ResNet-18 transfer-learning pipeline.
"""

IMAGE_TRANSFORM = v2.Compose(
    [
        v2.Grayscale(
            num_output_channels=3
        ),

        v2.Resize(
            (224, 224)
        ),

        v2.ToImage(),

        v2.ToDtype(
            torch.float32,
            scale=True
        ),

        v2.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# CLASSIFY PRODUCT IMAGE
# ============================================================

def classify_product_image(
    image_path: str
) -> dict:
    """
    Classify a real PNG image using the saved Part 2 model.

    Parameters
    ----------
    image_path : str
        Path to a PNG image.

    Example:
        data/sample_images/01_ankle_boot.png

    Returns
    -------
    dict
        image_path
        predicted_category
        confidence
    """

    # --------------------------------------------------------
    # Convert path string to Path object
    # --------------------------------------------------------

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Verify image exists
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Load actual Part 2 model
    # --------------------------------------------------------

    model = load_product_classifier()

    # --------------------------------------------------------
    # Open image
    #
    # Fashion-MNIST images are grayscale.
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")

    # --------------------------------------------------------
    # Apply same preprocessing used during Part 2
    # --------------------------------------------------------

    image_tensor = IMAGE_TRANSFORM(
        image
    )

    # Add batch dimension
    #
    # [3, 224, 224]
    #
    # becomes:
    #
    # [1, 3, 224, 224]

    image_tensor = image_tensor.unsqueeze(
        0).to("cpu")
    

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    # Convert tensor -> Python values

    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    # Convert class index to category name

    predicted_category = CLASS_NAMES[
        predicted_index
    ]

    return {
        "image_path": str(
            image_path
        ),

        "predicted_category": predicted_category,

        "confidence": round(
            confidence,
            6
        )
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TESTING PART 1 - RETURN RISK TOOL")
    print("=" * 60)

    # Example order.
    #
    # These are the same feature names used by Part 1.

    sample_order = {

        "product_category": "Apparel",

        "price_inr": 1200,

        "discount_pct": 25.0,

        "payment_method": "COD",

        "customer_tenure_days": 200,

        "num_previous_orders": 5,

        "num_previous_returns": 1,

        "delivery_distance_km": 200.0,

        "delivery_days": 5,

        "is_weekend_order": 1,

        "rating_given": 4.0
    }

    try:

        risk_result = check_return_risk(
            sample_order
        )

        print(
            json.dumps(
                risk_result,
                indent=2
            )
        )

    except Exception as e:

        print(
            f"Part 1 tool error: {e}"
        )

    print()
    print("=" * 60)
    print("TESTING PART 2 - IMAGE CLASSIFIER")
    print("=" * 60)

    # Find real PNG files exported from Fashion-MNIST

    sample_images_dir = (
        ROOT
        / "data"
        / "sample_images"
    )

    sample_images = list(
        sample_images_dir.glob(
            "*.png"
        )
    )

    if sample_images:

        # Test the first real sample image

        image_result = classify_product_image(
            str(sample_images[0])
        )

        print(
            json.dumps(
                image_result,
                indent=2
            )
        )

    else:

        print(
            "No sample PNG files found in:"
        )

        print(
            sample_images_dir
        )