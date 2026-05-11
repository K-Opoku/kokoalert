"""
KokoAlert — Image Classifier Module (Eye)
File: src/image_classifier.py

Classifies poultry droppings photos into 3 classes using MobileNetV2:
  0 = healthy        (normal brown droppings)
  1 = coccidiosis    (bloody or dark chocolate droppings)
  2 = newcastle      (bright green droppings)

This module is self-contained. If you already have these constants in
config.py, move them there and import instead.
"""

import io
import os
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS — move to config.py if you already define them there
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_SIZE = (224, 224)       # MobileNetV2 input size
IMAGE_CLASSES = ["healthy", "coccidiosis", "newcastle"]
IMAGE_CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 5.5}  # 2=newcastle — heavily upweighted
IMAGE_CONFIDENCE_THRESHOLD = 0.75  # Below this, image result is advisory only
IMAGE_MODEL_PATH = "models/droppings_classifier.h5"

# Map image class → droppings string for diagnosis engine
IMAGE_TO_DROPPINGS_MAP = {
    "healthy": "normal",
    "coccidiosis": "bloody_chocolate",
    "newcastle": "bright_green",
}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_image_classifier() -> tf.keras.Model:
    """
    Build the MobileNetV2-based droppings classifier.

    Architecture:
      base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
      base.trainable = False  # freeze during initial training
      x = GlobalAveragePooling2D()(base.output)
      x = Dense(128, activation='relu')(x)
      x = Dropout(0.3)(x)
      output = Dense(3, activation='softmax')(x)

    Training strategy:
      Phase 1: Train only the head (base frozen), 10–15 epochs, lr=1e-3
      Phase 2: Unfreeze top 30 layers of MobileNetV2, train with lr=1e-5

    NOTE: We wrap MobileNetV2 as a sub-model via base(model_input) so that
    model.layers[1] IS the MobileNetV2 object. This lets the training notebook
    access base_model.layers[:-30] for Phase 2 fine-tuning.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Phase 1: freeze backbone

    # Wrap as sub-model so notebook can access base_model.layers
    model_input = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(model_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = models.Model(model_input, outputs, name="KokoAlert_Eye")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

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

    # MobileNetV2 expects [-1, 1] scaling — NOT simple /255
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Add batch dimension
    return np.expand_dims(preprocessed, axis=0)


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Same as preprocess_image but accepts raw bytes (e.g. from WhatsApp download).

    Args:
        image_bytes: Raw image bytes (PNG/JPEG/etc.)

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)

    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(preprocessed, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_droppings(model: tf.keras.Model, image: np.ndarray) -> dict:
    """
    Run a preprocessed image through the classifier.

    Args:
        model: Trained Keras model (from build_image_classifier or load_image_classifier)
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

def load_image_classifier(model_path: str = IMAGE_MODEL_PATH) -> tf.keras.Model:
    """
    Load a saved droppings classifier from disk.
    Called once at API startup.

    Returns:
        Compiled Keras model ready for inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Image classifier not found at {model_path}. "
            f"Train it first using notebooks/05_image_classifier_training.ipynb"
        )
    return tf.keras.models.load_model(model_path)


def save_image_classifier(model: tf.keras.Model, model_path: str = IMAGE_MODEL_PATH) -> None:
    """
    Save a trained droppings classifier to disk.
    Creates the models/ directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL: Training helper (if you want to train from a script instead of
# the notebook). The notebook is the primary training path.
# ═══════════════════════════════════════════════════════════════════════════

def compile_image_classifier(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compile the image classifier with standard settings.
    Use class_weight=IMAGE_CLASS_WEIGHTS during fit() for NCD imbalance.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model