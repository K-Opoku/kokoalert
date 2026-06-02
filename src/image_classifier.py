"""
KokoAlert — Image Classifier Module (Eye)
File: src/image_classifier.py

Classifies poultry droppings photos into 3 classes using MobileNetV2:
  0 = healthy        (normal brown droppings)
  1 = coccidiosis    (bloody or dark chocolate droppings)
  2 = newcastle      (bright green droppings)

TensorFlow is imported lazily inside training functions only.
Inference (preprocess + predict) uses numpy — no TF required on Render.
"""

import io
import os

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_SIZE = (224, 224)
IMAGE_CLASSES = ["healthy", "coccidiosis", "newcastle"]
IMAGE_CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 5.5}
IMAGE_CONFIDENCE_THRESHOLD = 0.75
IMAGE_MODEL_PATH = "models/droppings_classifier.h5"

IMAGE_TO_DROPPINGS_MAP = {
    "healthy": "normal",
    "coccidiosis": "bloody_chocolate",
    "newcastle": "bright_green",
}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL BUILDING (local training only — requires full TensorFlow)
# ═══════════════════════════════════════════════════════════════════════════

def build_image_classifier():
    """
    Build the MobileNetV2-based droppings classifier.
    Requires full TensorFlow — run locally only.

    Architecture:
      MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(3, softmax)

    Training strategy:
      Phase 1: Train only the head (base frozen), 10–15 epochs, lr=1e-3
      Phase 2: Unfreeze top 30 layers of MobileNetV2, train with lr=1e-5
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model_input = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(model_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = models.Model(model_input, outputs, name="KokoAlert_Eye")
    return model


def compile_image_classifier(model):
    """
    Compile the image classifier with standard settings.
    Requires full TensorFlow — run locally only.
    Use class_weight=IMAGE_CLASS_WEIGHTS during fit() for NCD imbalance.
    """
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING (numpy only — no TensorFlow needed)
# ═══════════════════════════════════════════════════════════════════════════

def _mobilenet_preprocess(img_array: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 preprocessing: scale pixel values from [0, 255] to [-1, 1].
    Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input()
    but implemented in pure numpy — no TensorFlow import required.
    """
    return (img_array.astype(np.float32) / 127.5) - 1.0


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load image from file path, resize to 224×224, apply MobileNetV2
    preprocessing (scale to [-1, 1]), and add batch dimension.

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    preprocessed = _mobilenet_preprocess(img_array)
    return np.expand_dims(preprocessed, axis=0)


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Same as preprocess_image but accepts raw bytes (from WhatsApp download).

    Args:
        image_bytes: Raw image bytes (PNG/JPEG/etc.)

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    preprocessed = _mobilenet_preprocess(img_array)
    return np.expand_dims(preprocessed, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_droppings(model, image: np.ndarray) -> dict:
    """
    Run a preprocessed image through the classifier.

    Args:
        model: Trained Keras model (from load_image_classifier)
        image: Preprocessed array of shape (1, 224, 224, 3)

    Returns:
        {
            "class": str,               # "healthy" | "coccidiosis" | "newcastle"
            "class_index": int,         # 0 | 1 | 2
            "confidence": float,        # probability of predicted class, 0.0–1.0
            "all_probabilities": dict,  # {"healthy": 0.05, "coccidiosis": 0.87, ...}
            "image_provided": True,
            "reliable": bool,           # True if confidence >= 0.75
        }
    """
    if model is None:
        return {"image_provided": False}

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    probs = model.predict(image, verbose=0)[0]
    class_index = int(np.argmax(probs))
    confidence = float(probs[class_index])
    class_name = IMAGE_CLASSES[class_index]

    all_probabilities = {
        IMAGE_CLASSES[i]: float(probs[i]) for i in range(len(IMAGE_CLASSES))
    }

    return {
        "class": class_name,
        "class_index": class_index,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "image_provided": True,
        "reliable": confidence >= IMAGE_CONFIDENCE_THRESHOLD,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def load_image_classifier(model_path: str = IMAGE_MODEL_PATH):
    """
    Load a saved droppings classifier from disk.
    Called once at API startup.

    Returns None if model file doesn't exist — image classification is
    optional in KokoAlert. The system works without it.
    Returns the loaded Keras model if the file exists.
    """
    if not os.path.exists(model_path):
        print(
            f"Image classifier not found at {model_path} — "
            f"image classification disabled. Train it using "
            f"notebooks/05_image_classifier_training.ipynb"
        )
        return None

    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def save_image_classifier(model, model_path: str = IMAGE_MODEL_PATH) -> None:
    """
    Save a trained droppings classifier to disk.
    Requires full TensorFlow — run locally only.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)