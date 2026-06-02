import numpy as np
import json
import os
from pathlib import Path

from src.config import INPUT_SHAPE, MODEL_DIR


# ── TFLITE RUNTIME IMPORT ─────────────────────────────────────────────────────
# On Render: tflite-runtime is installed (small, ~3MB, fits in 512MB RAM)
# Locally:   falls back to tensorflow.lite (full TF install)

try:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter


# ── ARCHITECTURE + TRAINING (local use only — requires full TensorFlow) ───────
# These functions are called from training notebooks on your local machine.
# They are NOT called on Render. Render only calls load_autoencoder()
# and is_anomalous().

def build_classifier():
    """
    Build CNN binary classifier.
    Requires full TensorFlow — run locally only.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=INPUT_SHAPE)  # (128, 157, 1)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='koko_classifier')
    return model


def compile_classifier(model):
    """Compile classifier. Requires full TensorFlow — run locally only."""
    import tensorflow as tf
    from tensorflow import keras

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
    """Training callbacks. Requires full TensorFlow — run locally only."""
    import tensorflow as tf
    from tensorflow import keras

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


def save_classifier(model):
    """Save classifier weights. Requires full TensorFlow — run locally only."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, 'autoencoder.h5'))

    with open(os.path.join(MODEL_DIR, 'threshold.json'), 'w') as f:
        json.dump({'threshold': 0.5, 'percentile': None}, f, indent=2)

    print(f"Classifier saved to {MODEL_DIR}/autoencoder.h5")


# ── INFERENCE (TFLite — runs on Render free tier) ─────────────────────────────

# Normalization params loaded once at startup
_NORM_MIN = -80.0
_NORM_MAX = 1.9073486328125e-06


def load_autoencoder() -> tuple:
    """
    Load TFLite classifier at API startup.
    Named load_autoencoder for pipeline.py compatibility.
    Returns: (interpreter, threshold)

    Memory usage: ~15MB vs ~400MB for full TensorFlow.
    """
    global _NORM_MIN, _NORM_MAX

    tflite_path = os.path.join(MODEL_DIR, 'classifier.tflite')
    threshold_path = os.path.join(MODEL_DIR, 'threshold.json')
    norm_path = os.path.join(MODEL_DIR, 'normalization_params.json')

    if not os.path.exists(tflite_path):
        raise FileNotFoundError(
            f"TFLite model not found at {tflite_path}. "
            f"Run the conversion script locally to generate classifier.tflite."
        )

    interpreter = _Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    with open(threshold_path) as f:
        data = json.load(f)
    threshold = data['threshold']

    # Load normalization params if available
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        _NORM_MIN = norm['min']
        _NORM_MAX = norm['max']
        print(f"Normalization loaded — min: {_NORM_MIN}, max: {_NORM_MAX}")

    print(f"Classifier loaded — threshold: {threshold}")
    return interpreter, threshold


def is_anomalous(
    interpreter,
    spectrogram: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Run a single spectrogram window through the TFLite classifier.

    Args:
        interpreter: TFLite Interpreter returned by load_autoencoder()
        spectrogram: np.ndarray of shape (128, 157, 1)
        threshold: decision boundary — P(sick) > threshold → anomalous

    Returns:
        {
            'is_anomalous': bool,
            'probability': float,        # P(sick) 0.0–1.0
            'reconstruction_error': float,  # alias for pipeline compatibility
            'threshold': float,
            'margin': float,             # probability - threshold
        }
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Min-max normalise to [0, 1] — must match training preprocessing.
    # Model was trained on spectrograms normalised with normalization_params.json:
    #   min = -80.0 dB, max = 0.0 dB (standard log-mel range)
    # Without this, the model sees out-of-distribution inputs and outputs ~0.
    SPEC_MIN = -80.0
    SPEC_MAX = 0.0
    spectrogram = (spectrogram - SPEC_MIN) / (SPEC_MAX - SPEC_MIN + 1e-8)
    spectrogram = np.clip(spectrogram, 0.0, 1.0)

    # Add batch dimension and ensure float32
    spec_batch = np.expand_dims(spectrogram, axis=0).astype(np.float32)  # (1, 128, 157, 1)

    interpreter.set_tensor(input_details[0]['index'], spec_batch)
    interpreter.invoke()

    probability = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    return {
        'is_anomalous': probability > threshold,
        'probability': probability,
        'reconstruction_error': probability,  # kept for pipeline.py compatibility
        'threshold': threshold,
        'margin': probability - threshold,
    }


def compute_window_probabilities(
    interpreter,
    spectrograms: np.ndarray
) -> np.ndarray:
    """
    Run a batch of spectrograms through the TFLite classifier.
    Returns P(sick) for each window as a 1D array.
    Used during evaluation and threshold analysis.

    Args:
        interpreter: TFLite Interpreter returned by load_autoencoder()
        spectrograms: np.ndarray of shape (N, 128, 157, 1)
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    SPEC_MIN = -80.0
    SPEC_MAX = 0.0
    probabilities = []
    for spec in spectrograms:
        spec = (spec - SPEC_MIN) / (SPEC_MAX - SPEC_MIN + 1e-8)
        spec = np.clip(spec, 0.0, 1.0)
        spec_batch = np.expand_dims(spec, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], spec_batch)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        probabilities.append(prob)

    return np.array(probabilities)