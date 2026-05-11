import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from src.config import INPUT_SHAPE, MODEL_DIR


# ── ARCHITECTURE ──────────────────────────────────────────────────────────────

def build_classifier() -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)  # (128, 157, 1)

    # Block 1 — low-level features
    # padding='valid' — no zero-padding at edges
    # Forces model to learn from real data only, not edge artifacts
    x = layers.Conv2D(32, (3, 3), padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    # Shape: (63, 78, 32)

    # Block 2 — mid-level features
    x = layers.Conv2D(64, (3, 3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    # Shape: (31, 38, 64)

    # Block 3 — high-level features
    x = layers.Conv2D(128, (3, 3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)
    # Shape: (15, 18, 128)

    # GlobalAveragePooling2D — handles any spatial size
    # This is why valid padding works: GAP collapses (15, 18, 128) → (128,)
    # regardless of exact spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head — unchanged
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='koko_classifier')
    return model

# ── TRAINING ──────────────────────────────────────────────────────────────────

def compile_classifier(model: keras.Model) -> keras.Model:
    """
    Binary cross-entropy loss for binary classification.

    Why binary cross-entropy and not MSE:
    MSE penalises distance from the target value equally.
    Binary cross-entropy penalises confident wrong predictions
    exponentially — if the model says 0.99 healthy for a sick bird,
    the loss is enormous. This forces the model to be calibrated,
    not just directionally correct.

    AUC as a metric — more informative than accuracy for this task.
    AUC measures the probability that the model ranks a random sick
    window higher than a random healthy window. 0.5 = random, 1.0 = perfect.
    It's threshold-independent so it tells you about the underlying
    separation quality, not just performance at 0.5 cutoff.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def get_training_callbacks() -> list:
    """
    Same callback strategy as before.
    Now monitoring val_auc instead of val_loss —
    AUC is a better measure of classifier quality than raw loss.
    mode='max' because higher AUC is better (opposite of loss).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'classifier_best.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def is_anomalous(
    model: keras.Model,
    spectrogram: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Run a single spectrogram window through the classifier.

    Returns probability of being sick. The threshold parameter
    is kept for API compatibility with pipeline.py but defaults
    to 0.5 — the natural decision boundary for a sigmoid output.

    The margin field tells you how far from the boundary you are:
    - margin = +0.4 means P(sick) = 0.9 — high confidence sick
    - margin = -0.4 means P(sick) = 0.1 — high confidence healthy
    - margin near 0 means the model is uncertain
    """
    spec_batch = np.expand_dims(spectrogram, axis=0)  # (1, 128, 157, 1)
    probability = float(model.predict(spec_batch, verbose=0)[0][0])

    return {
        'is_anomalous': probability > threshold,
        'probability': probability,
        'reconstruction_error': probability,  # kept for pipeline.py compatibility
        'threshold': threshold,
        'margin': probability - threshold
    }


def compute_window_probabilities(
    model: keras.Model,
    spectrograms: np.ndarray
) -> np.ndarray:
    """
    Run a batch of spectrograms through the classifier.
    Returns P(sick) for each window.
    Used during evaluation and threshold analysis.
    """
    probabilities = model.predict(spectrograms, verbose=0).flatten()
    return probabilities


# ── MODEL PERSISTENCE ─────────────────────────────────────────────────────────

def save_classifier(model: keras.Model):
    """Save classifier. No threshold file needed — default is 0.5."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, 'autoencoder.h5'))

    # Save threshold.json for pipeline.py compatibility
    with open(os.path.join(MODEL_DIR, 'threshold.json'), 'w') as f:
        json.dump({'threshold': 0.5, 'percentile': None}, f, indent=2)

    print(f"Classifier saved to {MODEL_DIR}/autoencoder.h5")


def load_autoencoder() -> tuple[keras.Model, float]:
    """
    Load classifier at API startup.
    Named load_autoencoder for pipeline.py compatibility —
    pipeline.py calls this function by name on startup.
    """
    model_path = os.path.join(MODEL_DIR, 'autoencoder.h5')
    threshold_path = os.path.join(MODEL_DIR, 'threshold.json')

    model = keras.models.load_model(model_path)

    with open(threshold_path) as f:
        data = json.load(f)
    threshold = data['threshold']

    print(f"Classifier loaded — threshold: {threshold}")
    return model, threshold