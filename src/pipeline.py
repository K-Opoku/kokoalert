"""
src/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Full inference pipeline: audio file → classifier → diagnosis engine → alert.

Architecture:
  Layer 1 — CNN Classifier (disease_classifier.py)
             Audio → mel spectrogram → binary classification
             Output: P(sick) per 5-second window, majority vote across windows
             Trained on: Bowen Nigeria healthy + sick clips
             Fine-tuned on: Ghanaian farm recordings

  Layer 2 — Diagnosis Engine (diagnosis_engine.py)
             Audio result + symptoms + farm profile → confirmed diagnosis
             Combines: audio signal, droppings colour, behaviour signs,
             flock age, vaccination status, season, farm conditions
             Output: disease name, confidence, reasons, actions, agrovet guide

Why the classifier replaced the autoencoder:
  The autoencoder was producing false negatives on the Bowen dataset —
  marking sick birds as healthy because reconstruction error thresholds
  were hard to calibrate on limited data. The CNN binary classifier
  (healthy=0, sick=1) trained directly on labelled data gives clearer
  separation and better generalisation.

  The classifier is the "something is wrong" signal.
  The diagnosis engine is the "here is what it is and why" intelligence.
  Both are needed — neither alone is sufficient.
"""

import numpy as np
import json
import os
import tempfile
from pathlib import Path
from tensorflow import keras
from src.image_classifier import load_image_classifier, preprocess_image_from_bytes, predict_droppings

from src.config import MODEL_DIR, CLASSIFIER_THRESHOLD
from src.preprocess import file_to_spectrograms, check_recording_quality
from src.anomaly_detector import is_anomalous, load_autoencoder
from src.diagnosis_engine import run_diagnosis


class KokoAlertPipeline:
    """
    Full inference pipeline: audio file → confirmed diagnosis.

    Usage:
        pipeline = KokoAlertPipeline()
        pipeline.load_models()

        result = pipeline.analyse_audio(
            audio_file_path="recording.ogg",
            farm_profile=farm_profile_dict,
            symptoms=symptoms_dict,
        )

        # result["whatsapp_message"] is ready to send
        # result["disease"] is the confirmed disease key
        # result["urgency"] tells you how fast to act
    """

    def __init__(self):
        self.classifier = None
        self.threshold = CLASSIFIER_THRESHOLD
        self._loaded = False
        self.image_classifier = None   # ← add this

    def load_models(self):
        """Load CNN classifier. Call once at API startup."""
        print("Loading KokoAlert classifier...")
        self.classifier, self.threshold = load_autoencoder()
        self._loaded = True
        self.image_classifier = load_image_classifier()   # ← add this
        print(f"Classifier loaded. Threshold: {self.threshold}")

    def analyse_audio(
        self,
        audio_file_path: str,
        farm_profile: dict = None,
        symptoms: dict = None,
    ) -> dict:
        """
        Full analysis of an audio file.

        Args:
            audio_file_path: Path to audio file (WAV, MP3, OGG)
            farm_profile: Farmer's profile from database.
                Required for full diagnosis:
                  - flock_age_weeks or doc_arrival_date
                  - gumboro_vaccinated, newcastle_vaccinated
                  - ventilation, region
                If None, returns audio-only result (no confirmed diagnosis).
            symptoms: From daily/weekly WhatsApp check.
                Optional but improves diagnosis significantly:
                  - droppings: "normal", "bloody_chocolate",
                                "bright_green", "white_watery"
                  - behavior: list of strings
                  - cocci_medicine_given: bool
                If None, diagnosis engine requests droppings info.

        Returns:
            Full result dict. Key fields:
              - status: "diagnosed" | "healthy" | "needs_symptoms" | "inconclusive"
              - disease: disease key or None
              - urgency: "emergency" | "urgent" | "monitor" | "none"
              - whatsapp_message: formatted message ready to send
              - confidence: "High" | "Medium" | "Low"
              - reasons: list of reason strings
        """
        if not self._loaded:
            self.load_models()

        if farm_profile is None:
            farm_profile = {}
        if symptoms is None:
            symptoms = {}

        # ── STEP 1: Recording quality check ──────────────────────────────────
        quality = check_recording_quality(audio_file_path)

        if not quality["usable"]:
            return self._inconclusive_result(quality)

        # ── STEP 2: Preprocessing — audio → spectrograms ──────────────────────
        specs = file_to_spectrograms(audio_file_path)

        if len(specs) == 0:
            return self._inconclusive_result({
                "usable": False,
                "reason": "no_windows",
            })

        # ── STEP 3: CNN classifier — window-level predictions ─────────────────
        window_results = [
            is_anomalous(self.classifier, spec, self.threshold)
            for spec in specs
        ]

        # Average probability across all windows
        avg_probability = float(np.mean(
            [r["probability"] for r in window_results]
        ))

        # Majority vote — more than half anomalous = flock is anomalous
        anomaly_votes = sum(1 for r in window_results if r["is_anomalous"])
        flock_is_anomalous = anomaly_votes > (len(window_results) / 2)

        audio_result = {
            "is_anomalous": flock_is_anomalous,
            "probability": avg_probability,
            "anomaly_votes": anomaly_votes,
            "total_windows": len(window_results),
            "threshold": self.threshold,
        }

        # ── STEP 4: Diagnosis engine — combine audio + all signals ────────────
        diagnosis = run_diagnosis(
            farm_profile=farm_profile,
            audio_result=audio_result,
            symptoms=symptoms,
        )

        # Attach audio metadata for logging / dashboard
        diagnosis["audio"] = audio_result
        diagnosis["quality"] = quality

        return diagnosis

    # ── RESULT HELPERS ────────────────────────────────────────────────────────

    def _inconclusive_result(self, quality: dict) -> dict:
        """Return an inconclusive result with a helpful re-recording message."""
        reason = quality.get("reason", "unknown")

        if reason == "too_short":
            msg = (
                "❓ *Recording too short to analyse.*\n\n"
                "Please send a voice note of at least *10 seconds* from inside "
                "your poultry house.\n\n"
                "Tip: Place your phone on a surface inside the house — do not hold it."
            )
        elif reason == "too_quiet":
            msg = (
                "❓ *Recording too quiet to analyse.*\n\n"
                "The microphone was too far from the birds. "
                "Please re-record from *inside* the poultry house with the door closed.\n\n"
                "Place your phone on a surface among the birds."
            )
        elif reason == "clipping":
            msg = (
                "❓ *Recording distorted — too loud.*\n\n"
                "The microphone is too close to a noise source. "
                "Move slightly away and re-record."
            )
        else:
            msg = (
                "❓ *Could not process the recording.*\n\n"
                "Please try again. Record from inside the poultry house, "
                "phone placed on a flat surface, for at least 10 seconds."
            )

        return {
            "status": "inconclusive",
            "disease": None,
            "urgency": "none",
            "whatsapp_message": msg,
            "quality": quality,
        }
    def analyse_image(self, image_bytes: bytes) -> dict:
        """
        Classify a droppings photo from raw bytes (WhatsApp download).
        Returns image_result dict for diagnosis engine.
        Returns {"image_provided": False} on any failure.
        """
        try:
            if self.image_classifier is None:
                return {"image_provided": False}
            image = preprocess_image_from_bytes(image_bytes)
            return predict_droppings(self.image_classifier, image)
        except Exception as e:
            print(f"Image analysis error: {e}")
            return {"image_provided": False}

# ── SINGLETON ─────────────────────────────────────────────────────────────────
# Loaded once at API startup. Shared across all WhatsApp requests.
pipeline = KokoAlertPipeline()
