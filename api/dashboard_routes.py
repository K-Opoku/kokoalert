"""
api/dashboard_routes.py
─────────────────────────────────────────────────────────────────────────────
Dashboard and analysis API routes.

HOW TO ADD THIS TO YOUR APP:
  In api/main.py, add these three lines:

    from api.dashboard_routes import router as dashboard_router
    app.include_router(dashboard_router)
    # (add this after app = FastAPI(...) and BEFORE the lifespan context)

  Also add CORS middleware in main.py after app is created:

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

NEW ENDPOINTS THIS FILE ADDS:
  POST /api/analyze/audio   — audio file → spectrogram + P(sick)
  POST /api/analyze/full    — audio + symptoms + optional image → full diagnosis
  GET  /api/dashboard/stats — aggregated farm stats for command center
"""

import base64
import io
import json
import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="/api")


# ── SPECTROGRAM HELPER ────────────────────────────────────────────────────────

def spectrogram_to_base64(spectrograms: list) -> str:
    """
    Convert the first spectrogram in the list to a base64-encoded PNG.
    Returns empty string if no spectrograms or matplotlib unavailable.
    """
    if not spectrograms:
        return ""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        spec = spectrograms[0][:, :, 0]  # (128, 157)

        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, 5, 0.5, 8],
            interpolation="nearest",
        )
        ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)
        ax.set_ylabel("Freq (kHz)", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#0d1117")
        buf.seek(0)
        plt.close(fig)

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return ""


# ── AUDIO ANALYSIS ────────────────────────────────────────────────────────────

@router.post("/analyze/audio")
async def analyze_audio(
    audio: UploadFile = File(...),
    flock_age_weeks: int = Form(default=0),
    region: str = Form(default="Ashanti"),
):
    """
    Step 1 of the live diagnosis flow.
    Accepts a WAV/MP3/OGG voice note.
    Returns spectrogram image (base64 PNG) + window-level predictions.

    Response:
        usable:             bool   — False if recording is too short/quiet
        is_anomalous:       bool   — True if ≥2 windows exceed threshold
        probability:        float  — average P(sick) across all windows
        anomalous_windows:  int
        total_windows:      int
        spectrogram_base64: str    — base64 PNG of first spectrogram window
        duration_seconds:   float
    """
    suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        from src.preprocess import file_to_spectrograms, check_recording_quality
        from src.disease_classifier import load_autoencoder, is_anomalous
        import numpy as np

        quality = check_recording_quality(tmp_path)
        if not quality["usable"]:
            return {"usable": False, "error": quality["reason"]}

        spectrograms = file_to_spectrograms(tmp_path)
        if not spectrograms:
            return {"usable": False, "error": "No valid audio windows extracted"}

        model, threshold = load_autoencoder()
        results = [is_anomalous(model, s, threshold) for s in spectrograms]
        probs = [r["probability"] for r in results]
        anomalous_count = sum(1 for r in results if r["is_anomalous"])

        return {
            "usable": True,
            "is_anomalous": anomalous_count >= 2,
            "probability": round(float(np.mean(probs)), 4),
            "anomalous_windows": anomalous_count,
            "total_windows": len(spectrograms),
            "spectrogram_base64": spectrogram_to_base64(spectrograms),
            "duration_seconds": round(quality.get("duration", 0), 1),
        }

    finally:
        os.unlink(tmp_path)


# ── FULL DIAGNOSIS ────────────────────────────────────────────────────────────

@router.post("/analyze/full")
async def analyze_full(
    audio: UploadFile = File(...),
    symptoms_json: str = Form(...),
    farm_profile_json: str = Form(default="{}"),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Full multi-modal diagnosis.
    Accepts audio + symptom form data + optional droppings image.
    Returns complete diagnosis dict (same shape as WhatsApp pipeline output)
    plus spectrogram_base64 for the dashboard.

    symptoms_json fields:
        droppings:            "normal" | "bloody_chocolate" | "bright_green" | "white_watery"
        behavior:             list[str]  e.g. ["coughing", "weak"]
        cocci_medicine_given: bool

    farm_profile_json fields (all optional, sensible defaults applied):
        region, flock_age_weeks, gumboro_vaccinated, newcastle_vaccinated,
        ventilation, bird_type, cocci_medicine_given
    """
    suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    image_bytes = await image.read() if image else None

    try:
        from src.pipeline import pipeline
        from src.preprocess import file_to_spectrograms

        symptoms = json.loads(symptoms_json)
        farm_profile = json.loads(farm_profile_json)

        # Apply defaults so the diagnosis engine always gets a complete profile
        defaults = {
            "region": "Ashanti",
            "flock_age_weeks": symptoms.get("flock_age_weeks", 4),
            "gumboro_vaccinated": "both",
            "newcastle_vaccinated": "full",
            "ventilation": "medium",
            "bird_type": "broiler",
            "cocci_medicine_given": False,
        }
        for k, v in defaults.items():
            farm_profile.setdefault(k, v)

        result = pipeline.analyse_audio(tmp_path, farm_profile, symptoms)

        if image_bytes:
            image_result = pipeline.analyse_image(image_bytes)
            result["image_result"] = image_result

        spectrograms = file_to_spectrograms(tmp_path)
        result["spectrogram_base64"] = spectrogram_to_base64(spectrograms)

        return result

    finally:
        os.unlink(tmp_path)


# ── DASHBOARD STATS ───────────────────────────────────────────────────────────

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """
    Aggregated statistics for the Command Center dashboard.

    Returns:
        total_farms:               int
        total_birds:               int   (estimated from flock_size bucket)
        gumboro_danger_count:      int   (farms where flock is weeks 3–6)
        gumboro_danger_farms:      list  [{name, age_weeks, region}]
        vaccination_compliance_pct: int  (% fully vaccinated)
        farms_by_region:           dict  {region: count}
        recent_alerts:             list  (last 10 disease detections)
        detections_this_month:     int
    """
    from api.database import get_all_active_farmers, get_farm_profile, get_recent_analysis

    FLOCK_SIZE_MAP = {
        "under_100": 50,
        "100_to_500": 300,
        "500_to_2000": 1000,
        "over_2000": 2500,
    }

    phones = get_all_active_farmers()
    total_birds = 0
    regions: dict = {}
    gumboro_danger = []
    recent_alerts = []
    vacc_compliant = 0
    this_month = datetime.now().strftime("%Y-%m")

    for phone in phones:
        profile = get_farm_profile(phone)
        if not profile:
            continue

        total_birds += FLOCK_SIZE_MAP.get(profile.get("flock_size", "under_100"), 50)

        region = profile.get("region", "Unknown")
        regions[region] = regions.get(region, 0) + 1

        age = profile.get("flock_age_weeks", 0)
        if 3 <= age <= 6:
            gumboro_danger.append({
                "name": profile.get("farmer_name", f"Farm {phone[-4:]}"),
                "age_weeks": age,
                "region": region,
            })

        if (profile.get("gumboro_vaccinated") == "both" and
                profile.get("newcastle_vaccinated") == "full"):
            vacc_compliant += 1

        for a in get_recent_analysis(phone, limit=5):
            if a.get("disease") and a.get("urgency") in ("emergency", "urgent"):
                recent_alerts.append({
                    "disease":      a.get("disease"),
                    "disease_name": a.get("disease_name", ""),
                    "confidence":   a.get("confidence", "Medium"),
                    "urgency":      a.get("urgency", "urgent"),
                    "region":       region,
                    "farm_name":    profile.get("farmer_name", f"Farm {phone[-4:]}"),
                    "recorded_at":  a.get("recorded_at", ""),
                })

    recent_alerts.sort(key=lambda x: x.get("recorded_at", ""), reverse=True)

    return {
        "total_farms":                len(phones),
        "total_birds":                total_birds,
        "gumboro_danger_count":       len(gumboro_danger),
        "gumboro_danger_farms":       gumboro_danger,
        "vaccination_compliance_pct": round(vacc_compliant / len(phones) * 100) if phones else 0,
        "farms_by_region":            regions,
        "recent_alerts":              recent_alerts[:10],
        "detections_this_month":      sum(
            1 for a in recent_alerts if this_month in a.get("recorded_at", "")
        ),
    }