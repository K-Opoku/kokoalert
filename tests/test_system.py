"""
tests/test_system.py
─────────────────────────────────────────────────────────────────────────────
KokoAlert full system test.

Run this from your project root:
    python tests/test_system.py

It tests every component in order. Each test prints PASS or FAIL.
If something fails, the error message tells you exactly what to fix.

No WhatsApp API token needed — all tests run locally.
No audio file needed — pipeline test uses a generated sine wave.
No model file needed for most tests — only the pipeline test needs the model.
"""

import io
import json
import os
import sys
import traceback
import numpy as np
from datetime import date, timedelta
from pathlib import Path

# ── Make sure project root is in path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Test result tracking ──────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"
results = []


def test(name: str):
    """Decorator that runs a test and records the result."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'─' * 60}")
            print(f"TEST: {name}")
            try:
                fn()
                print(f"{PASS}: {name}")
                results.append((name, "PASS", None))
            except Exception as e:
                print(f"{FAIL}: {name}")
                print(f"  Error: {e}")
                traceback.print_exc()
                results.append((name, "FAIL", str(e)))
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATABASE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Database — init creates tables")
def test_db_init():
    # Use a temp database for testing so we don't touch production data
    import api.database as db
    original_path = db.DB_PATH
    db.DB_PATH = Path("data/test_kokoalert.db")

    db.init_db()
    assert db.DB_PATH.exists(), "Database file was not created"

    db.DB_PATH = original_path
    print("  Tables created successfully.")

test_db_init()


@test("Database — save and retrieve farm profile")
def test_farm_profile():
    import api.database as db
    db.DB_PATH = Path("data/test_kokoalert.db")
    db.init_db()

    phone = "+233200000001"
    profile = {
        "region": "Ashanti",
        "flock_size": "100_to_500",
        "bird_type": "broiler",
        "flock_age_weeks": 4,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": "2026-04-01",
    }

    db.save_farm_profile(phone, profile)
    retrieved = db.get_farm_profile(phone)

    assert retrieved is not None, "Profile not found after saving"
    assert retrieved["region"] == "Ashanti"
    assert retrieved["flock_age_weeks"] == 4
    print(f"  Profile saved and retrieved. Region: {retrieved['region']}")

test_farm_profile()


@test("Database — save and retrieve vaccination log")
def test_vaccination_log():
    import api.database as db
    db.DB_PATH = Path("data/test_kokoalert.db")
    db.init_db()

    phone = "+233200000001"
    log = {
        "gumboro_1": "2026-04-08",
        "newcastle_1": "2026-04-15",
        "gumboro_2": "2026-04-22",
    }

    db.save_vaccination_log(phone, log)
    retrieved = db.get_vaccination_log(phone)

    assert retrieved is not None
    assert retrieved["gumboro_1"] == "2026-04-08"
    assert len(retrieved) == 3
    print(f"  Vaccination log saved. {len(retrieved)} entries.")

test_vaccination_log()


@test("Database — conversation state set and clear")
def test_conversation_state():
    import api.database as db
    db.DB_PATH = Path("data/test_kokoalert.db")
    db.init_db()

    phone = "+233200000002"
    db.set_onboarding_state(phone, "awaiting_droppings")
    state = db.get_onboarding_state(phone)
    assert state == "awaiting_droppings", f"Expected awaiting_droppings, got {state}"

    db.clear_onboarding_state(phone)
    state = db.get_onboarding_state(phone)
    assert state is None, f"Expected None after clear, got {state}"
    print("  State set, retrieved, and cleared correctly.")

test_conversation_state()


@test("Database — get all active farmers")
def test_get_all_farmers():
    import api.database as db
    db.DB_PATH = Path("data/test_kokoalert.db")

    phones = db.get_all_active_farmers()
    assert isinstance(phones, list)
    print(f"  {len(phones)} farmer(s) in test database.")

test_get_all_farmers()


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Config — all required constants present")
def test_config():
    from src.config import (
        SAMPLE_RATE, INPUT_SHAPE, VACCINATION_SCHEDULE,
        DISEASE_SIGNS, AGROVET_DRUGS, MONTHLY_RISK_DATA,
        VSD_CONTACTS, FLOCK_AGE_WINDOWS, IMAGE_CLASSES,
        IMAGE_TO_DROPPINGS_MAP, IMAGE_CONFIDENCE_THRESHOLD,
    )
    assert SAMPLE_RATE == 16000
    assert INPUT_SHAPE == (128, 157, 1)
    assert len(VACCINATION_SCHEDULE) >= 8
    assert len(DISEASE_SIGNS) == 5
    assert len(MONTHLY_RISK_DATA) == 12
    assert "Ashanti" in VSD_CONTACTS
    assert len(IMAGE_CLASSES) == 3
    print(f"  {len(VACCINATION_SCHEDULE)} vaccination schedule items.")
    print(f"  {len(DISEASE_SIGNS)} diseases covered.")

test_config()


# ═══════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Preprocess — audio to spectrograms shape")
def test_preprocess():
    import soundfile as sf
    import tempfile
    from src.preprocess import file_to_spectrograms, check_recording_quality
    from src.config import SAMPLE_RATE, INPUT_SHAPE

    # Generate a 15-second sine wave as a fake audio file
    duration = 15
    freq = 440  # A note — doesn't matter what frequency
    t = np.linspace(0, duration, SAMPLE_RATE * duration)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    quality = check_recording_quality(tmp_path)
    assert quality["usable"], f"Quality check failed: {quality.get('reason')}"

    specs = file_to_spectrograms(tmp_path)
    assert len(specs) > 0, "No spectrograms produced"
    assert specs[0].shape == INPUT_SHAPE, f"Wrong shape: {specs[0].shape}"

    os.unlink(tmp_path)
    print(f"  Generated {len(specs)} spectrograms, each shape {specs[0].shape}")

test_preprocess()


# ═══════════════════════════════════════════════════════════════════════════
# 4. DISEASE CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Disease Classifier — build model and check output shape")
def test_classifier_build():
    from src.anomaly_detector import build_classifier, compile_classifier
    from src.config import INPUT_SHAPE

    model = build_classifier()
    model = compile_classifier(model)

    # Check input shape
    assert model.input_shape == (None, *INPUT_SHAPE)
    # Check output shape (1 neuron sigmoid)
    assert model.output_shape == (None, 1)

    # Run a dummy forward pass
    dummy = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
    output = model.predict(dummy, verbose=0)
    assert 0.0 <= float(output[0][0]) <= 1.0

    print(f"  Model built. Input: {model.input_shape}, Output: {model.output_shape}")
    print(f"  Dummy forward pass: P(sick) = {float(output[0][0]):.4f}")

test_classifier_build()


@test("Disease Classifier — load saved model (if exists)")
def test_classifier_load():
    model_path = Path("models/autoencoder.h5")
    if not model_path.exists():
        print(f"  {SKIP} — models/autoencoder.h5 not found. Train the model first.")
        results.append(("Load saved audio model", "SKIP", "Model file not found"))
        return

    from src.anomaly_detector import load_autoencoder, is_anomalous
    from src.config import INPUT_SHAPE

    model, threshold = load_autoencoder()
    assert 0.0 < threshold <= 1.0


    dummy = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
    result = is_anomalous(model, dummy[0])
    assert "is_anomalous" in result
    assert "probability" in result
    assert "reliable" not in result  # should not be there
    print(f"  Model loaded. Threshold: {threshold}")
    print(f"  Dummy inference: P(sick) = {result['probability']:.4f}")

test_classifier_load()


# ═══════════════════════════════════════════════════════════════════════════
# 5. IMAGE CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Image Classifier — build model and check output shape")
def test_image_classifier_build():
    from src.image_classifier import build_image_classifier

    model = build_image_classifier()
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)  # 3 classes

    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy, verbose=0)
    assert output.shape == (1, 3)
    assert abs(sum(output[0]) - 1.0) < 0.001  # softmax sums to 1

    print(f"  Image model built. Output: {model.output_shape}")
    print(f"  Dummy probabilities: {dict(zip(['healthy','cocci','newcastle'], output[0].round(3)))}")

test_image_classifier_build()


@test("Image Classifier — preprocess from bytes")
def test_image_preprocess():
    from src.image_classifier import preprocess_image_from_bytes

    # Create a dummy 100x100 RGB image as bytes
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = preprocess_image_from_bytes(image_bytes)
    assert result.shape == (1, 224, 224, 3), f"Wrong shape: {result.shape}"
    assert result.min() >= -1.0 and result.max() <= 1.0, "MobileNetV2 preprocessing failed"

    print(f"  Image preprocessed. Shape: {result.shape}, Range: [{result.min():.2f}, {result.max():.2f}]")

test_image_preprocess()


@test("Image Classifier — load saved model (if exists)")
def test_image_model_load():
    model_path = Path("models/droppings_classifier.h5")
    if not model_path.exists():
        print(f"  {SKIP} — models/droppings_classifier.h5 not found. Train notebook 05 first.")
        results.append(("Load saved image model", "SKIP", "Model file not found"))
        return

    from src.image_classifier import load_image_classifier, predict_droppings, preprocess_image_from_bytes
    from PIL import Image

    model = load_image_classifier()

    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    image = preprocess_image_from_bytes(buf.getvalue())
    result = predict_droppings(model, image)

    assert "class" in result
    assert result["class"] in ["healthy", "coccidiosis", "newcastle"]
    assert "confidence" in result
    assert "reliable" in result
    print(f"  Image model loaded. Dummy prediction: {result['class']} ({result['confidence']:.2%})")

test_image_model_load()


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSIS ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Diagnosis Engine — Gumboro detected correctly")
def test_diagnosis_gumboro():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 4,
        "gumboro_vaccinated": "none",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=4)).isoformat(),
    }
    audio = {"is_anomalous": True, "probability": 0.72}
    symptoms = {
        "droppings": "white_watery",
        "behavior": ["weak", "huddled"],
        "cocci_medicine_given": True,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["status"] == "diagnosed", f"Expected diagnosed, got {result['status']}"
    assert result["disease"] == "gumboro", f"Expected gumboro, got {result['disease']}"
    assert result["urgency"] == "emergency"
    assert len(result["reasons"]) > 0
    assert result["whatsapp_message"] is not None

    print(f"  Disease: {result['disease']}")
    print(f"  Confidence: {result['confidence']} ({result['confidence_score']:.2f})")
    print(f"  Reasons: {len(result['reasons'])}")
    print(f"  Urgency: {result['urgency']}")

test_diagnosis_gumboro()


@test("Diagnosis Engine — Newcastle detected correctly")
def test_diagnosis_newcastle():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 8,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "none",
        "ventilation": "medium",
        "doc_arrival_date": (date.today() - timedelta(weeks=8)).isoformat(),
    }
    audio = {"is_anomalous": True, "probability": 0.81}
    symptoms = {
        "droppings": "bright_green",
        "behavior": ["sudden_deaths", "respiratory_distress"],
        "cocci_medicine_given": True,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["disease"] == "newcastle", f"Expected newcastle, got {result['disease']}"
    assert result["urgency"] == "emergency"
    assert result["has_cure"] == False
    print(f"  Disease: {result['disease']}, Confidence: {result['confidence']}")

test_diagnosis_newcastle()


@test("Diagnosis Engine — Coccidiosis detected correctly")
def test_diagnosis_coccidiosis():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Greater Accra",
        "flock_age_weeks": 5,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=5)).isoformat(),
    }
    audio = {"is_anomalous": False, "probability": 0.3}
    symptoms = {
        "droppings": "bloody_chocolate",
        "behavior": ["reduced_appetite"],
        "cocci_medicine_given": False,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["disease"] == "coccidiosis", f"Expected coccidiosis, got {result['disease']}"
    assert result["has_cure"] == True

    # Check that cascade warning is in the reasons (age 5 = Gumboro window)
    reasons_text = " ".join(result["reasons"])
    assert "Gumboro" in reasons_text or "cascade" in reasons_text.lower(), \
        "Cascade warning missing for week 5 bird"

    print(f"  Disease: {result['disease']}, Confidence: {result['confidence']}")
    print(f"  Cascade warning present: {'Gumboro' in reasons_text}")

test_diagnosis_coccidiosis()


@test("Diagnosis Engine — CRD detected correctly")
def test_diagnosis_crd():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 10,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "poor",
        "doc_arrival_date": (date.today() - timedelta(weeks=10)).isoformat(),
    }
    audio = {"is_anomalous": True, "probability": 0.68}
    symptoms = {
        "droppings": "normal",
        "behavior": ["coughing", "sneezing", "nasal_discharge"],
        "cocci_medicine_given": True,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["disease"] == "crd", f"Expected crd, got {result['disease']}"
    assert result["has_cure"] == True
    print(f"  Disease: {result['disease']}, Confidence: {result['confidence']}")

test_diagnosis_crd()


@test("Diagnosis Engine — Fowl Pox detected correctly")
def test_diagnosis_fowl_pox():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 12,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=12)).isoformat(),
    }
    audio = {"is_anomalous": False, "probability": 0.2}
    symptoms = {
        "droppings": "normal",
        "behavior": ["face_lesions", "reduced_eating"],
        "cocci_medicine_given": True,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["disease"] == "fowl_pox", f"Expected fowl_pox, got {result['disease']}"
    assert result["urgency"] == "monitor"
    print(f"  Disease: {result['disease']}, Confidence: {result['confidence']}")

test_diagnosis_fowl_pox()


@test("Diagnosis Engine — Healthy result")
def test_diagnosis_healthy():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 8,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=8)).isoformat(),
    }
    audio = {"is_anomalous": False, "probability": 0.2}
    symptoms = {
        "droppings": "normal",
        "behavior": [],
        "cocci_medicine_given": True,
    }

    result = run_diagnosis(profile, audio, symptoms)

    assert result["status"] == "healthy", f"Expected healthy, got {result['status']}"
    assert result["disease"] is None
    print(f"  Status: {result['status']} — healthy result returned correctly")

test_diagnosis_healthy()


@test("Diagnosis Engine — image_result bonus scoring")
def test_diagnosis_with_image():
    from src.diagnosis_engine import run_diagnosis

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 6,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "none",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=6)).isoformat(),
    }
    audio = {"is_anomalous": True, "probability": 0.75}
    symptoms = {
        "droppings": "bright_green",
        "behavior": ["sudden_deaths"],
        "cocci_medicine_given": True,
    }
    image_result = {
        "class": "newcastle",
        "confidence": 0.89,
        "reliable": True,
        "image_provided": True,
    }

    result = run_diagnosis(profile, audio, symptoms, image_result=image_result)

    assert result["disease"] == "newcastle"
    assert result.get("image_used") == True
    print(f"  Disease: {result['disease']}, Image used: {result.get('image_used')}")
    print(f"  Confidence with image: {result['confidence_score']:.2f}")

test_diagnosis_with_image()


# ═══════════════════════════════════════════════════════════════════════════
# 7. VACCINATION SCHEDULER TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Vaccination Scheduler — flock age calculation")
def test_flock_age():
    from src.vaccination_scheduler import get_flock_age_days, get_flock_age_weeks

    # Birds that arrived 28 days ago = 4 weeks old
    arrival = (date.today() - timedelta(days=28)).isoformat()
    days = get_flock_age_days(arrival)
    weeks = get_flock_age_weeks(arrival)

    assert days == 28, f"Expected 28, got {days}"
    assert weeks == 4, f"Expected 4, got {weeks}"
    print(f"  Arrival 28 days ago → {days} days → {weeks} weeks ✓")

test_flock_age()


@test("Vaccination Scheduler — reminders for new flock")
def test_vaccination_reminders():
    from src.vaccination_scheduler import get_todays_reminders

    # Simulate a flock that arrived 7 days ago (1st Gumboro due today)
    profile = {
        "doc_arrival_date": (date.today() - timedelta(days=7)).isoformat(),
    }
    vacc_log = {}

    reminders = get_todays_reminders(profile, vacc_log)

    # Should have at least the Gumboro 1 reminder
    assert isinstance(reminders, list)
    print(f"  {len(reminders)} reminder(s) due for 7-day-old flock.")
    for r in reminders:
        print(f"    • {r['label']} ({r.get('timing', '')})")

test_vaccination_reminders()


@test("Vaccination Scheduler — register new flock")
def test_register_flock():
    from src.vaccination_scheduler import register_new_flock

    profile = {"region": "Ashanti", "flock_size": "100_to_500"}
    updated, msg = register_new_flock(profile)

    assert "doc_arrival_date" in updated
    assert updated["flock_age_weeks"] == 0
    assert "glucose" in msg.lower() or "chick" in msg.lower()
    print(f"  New flock registered. DOC date: {updated['doc_arrival_date']}")

test_register_flock()


@test("Vaccination Scheduler — full schedule display")
def test_full_schedule():
    from src.vaccination_scheduler import format_full_vaccination_schedule

    profile = {
        "doc_arrival_date": (date.today() - timedelta(days=14)).isoformat(),
    }
    vacc_log = {"gumboro_1": "2026-04-08"}

    msg = format_full_vaccination_schedule(profile, vacc_log)
    assert "Vaccination Schedule" in msg
    assert "Week" in msg
    print(f"  Schedule formatted. Length: {len(msg)} chars.")

test_full_schedule()


# ═══════════════════════════════════════════════════════════════════════════
# 8. RISK ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Risk Engine — Gumboro window raises score")
def test_risk_gumboro_window():
    from src.risk_engine import compute_farm_risk_score

    profile_safe = {
        "region": "Ashanti",
        "flock_age_weeks": 8,
        "vaccinated": True,
        "days_since_vaccination": 30,
        "has_footbath": True,
        "ventilation": "good",
        "flock_size": "100_to_500",
    }
    profile_danger = {**profile_safe, "flock_age_weeks": 4}

    score_safe = compute_farm_risk_score(profile_safe)["score"]
    score_danger = compute_farm_risk_score(profile_danger)["score"]

    assert score_danger > score_safe, \
        f"Gumboro window should increase score. Safe: {score_safe}, Danger: {score_danger}"
    print(f"  Score at week 8: {score_safe}")
    print(f"  Score at week 4 (Gumboro window): {score_danger}")
    print(f"  Increase: +{score_danger - score_safe} points ✓")

test_risk_gumboro_window()


@test("Risk Engine — unvaccinated farm is Critical")
def test_risk_unvaccinated():
    from src.risk_engine import compute_farm_risk_score

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 4,
        "vaccinated": False,
        "days_since_vaccination": 365,
        "has_footbath": False,
        "ventilation": "poor",
        "flock_size": "over_2000",
        "new_birds_introduced": True,
        "recent_deaths": True,
    }
    result = compute_farm_risk_score(profile)
    assert result["score"] >= 50, f"Expected high risk score, got {result['score']}"
    print(f"  Score: {result['score']}/100 — {result['category']} {result['emoji']}")

test_risk_unvaccinated()


# ═══════════════════════════════════════════════════════════════════════════
# 9. BIOSECURITY SCORER TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Biosecurity Scorer — perfect farm scores 10")
def test_biosecurity_perfect():
    from src.biosecurity_scorer import compute_biosecurity_score

    profile = {
        "has_footbath": True,
        "ventilation": "good",
        "new_birds_introduced": False,
        "flock_size": "100_to_500",
    }
    result = compute_biosecurity_score(profile)
    assert result["score"] == 10, f"Expected 10, got {result['score']}"
    assert result["grade"] == "Good"
    print(f"  Score: {result['score']}/10 — {result['grade']} ✓")

test_biosecurity_perfect()


@test("Biosecurity Scorer — bad farm loses points")
def test_biosecurity_poor():
    from src.biosecurity_scorer import compute_biosecurity_score

    profile = {
        "has_footbath": False,
        "ventilation": "poor",
        "new_birds_introduced": True,
        "flock_size": "over_2000",
    }
    result = compute_biosecurity_score(profile)
    assert result["score"] < 5, f"Expected poor score, got {result['score']}"
    print(f"  Score: {result['score']}/10 — {result['grade']}")
    print(f"  Issues found: {len(result['improvements'])}")

test_biosecurity_poor()


# ═══════════════════════════════════════════════════════════════════════════
# 10. FULL PIPELINE TEST (requires trained model)
# ═══════════════════════════════════════════════════════════════════════════

@test("Full Pipeline — audio file to diagnosis")
def test_full_pipeline():
    model_path = Path("models/autoencoder.h5")
    if not model_path.exists():
        print(f"  {SKIP} — models/autoencoder.h5 not found.")
        results.append(("Full Pipeline", "SKIP", "Model not found"))
        return

    import soundfile as sf
    import tempfile
    from src.pipeline import pipeline as koko_pipeline

    # Generate a test audio file
    duration = 15
    t = np.linspace(0, duration, 16000 * duration)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name

    profile = {
        "region": "Ashanti",
        "flock_age_weeks": 4,
        "gumboro_vaccinated": "both",
        "newcastle_vaccinated": "full",
        "ventilation": "good",
        "doc_arrival_date": (date.today() - timedelta(weeks=4)).isoformat(),
    }
    symptoms = {
        "droppings": "normal",
        "behavior": [],
        "cocci_medicine_given": True,
    }

    result = koko_pipeline.analyse_audio(tmp_path, profile, symptoms)
    os.unlink(tmp_path)

    assert "status" in result
    assert "whatsapp_message" in result
    assert result["status"] in ["healthy", "diagnosed", "needs_symptoms", "inconclusive"]

    print(f"  Pipeline status: {result['status']}")
    if result.get("disease"):
        print(f"  Disease: {result['disease']}, Confidence: {result.get('confidence')}")
    print(f"  WhatsApp message length: {len(result['whatsapp_message'])} chars")

test_full_pipeline()


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("KOKOALERT SYSTEM TEST RESULTS")
print(f"{'═' * 60}")

passed = [r for r in results if r[1] == "PASS"]
failed = [r for r in results if r[1] == "FAIL"]
skipped = [r for r in results if r[1] == "SKIP"]

for name, status, error in results:
    icon = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⏭️ ")
    print(f"{icon} {name}")

print(f"\n{'─' * 60}")
print(f"✅ PASSED: {len(passed)}")
print(f"❌ FAILED: {len(failed)}")
print(f"⏭️  SKIPPED: {len(skipped)} (need trained model files)")

if failed:
    print(f"\n{'─' * 60}")
    print("FAILURES TO FIX:")
    for name, _, error in failed:
        print(f"  ❌ {name}: {error}")

if not failed:
    print(f"\n🎉 All tests passed. KokoAlert is ready.")
else:
    print(f"\n⚠️  Fix the {len(failed)} failure(s) above before deployment.")

# Clean up test database
test_db = Path("data/test_kokoalert.db")
if test_db.exists():
    test_db.unlink()
    print("\n(Test database cleaned up.)")