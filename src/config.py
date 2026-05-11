import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ── AUDIO SETTINGS ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
HOP_SECONDS = 2
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 500       # Focuses on respiratory frequency range (coughs, rales, gasping)
FMAX = 8000
SPEC_TIME_STEPS = 157

# ── MODEL SETTINGS ────────────────────────────────────────────────────────────
INPUT_SHAPE = (128, 157, 1)
CLASSIFIER_THRESHOLD = 0.5   # Sigmoid output — 0.5 is the natural boundary

# ── IMAGE ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
IMAGE_CLASSES = ["healthy", "coccidiosis", "newcastle"]
IMAGE_CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 5.5}
IMAGE_CONFIDENCE_THRESHOLD = 0.75
IMAGE_MODEL_PATH = "models/droppings_classifier.h5"

IMAGE_TO_DROPPINGS_MAP = {
    "healthy":     "normal",
    "coccidiosis": "bloody_chocolate",
    "newcastle":   "bright_green",
}
# ── FLOCK AGE RISK WINDOWS ────────────────────────────────────────────────────
# These define which diseases are most likely at each age range.
# Based on your father's knowledge + veterinary science.
# Key insight: disease risk is entirely age-dependent.
FLOCK_AGE_WINDOWS = {
    "chick":       {"min_weeks": 0,  "max_weeks": 2,  "label": "Chick stage (0–2 weeks)"},
    "gumboro":     {"min_weeks": 3,  "max_weeks": 6,  "label": "Gumboro danger window (3–6 weeks)"},
    "growing":     {"min_weeks": 7,  "max_weeks": 9,  "label": "Growing stage (7–9 weeks)"},
    "booster":     {"min_weeks": 10, "max_weeks": 15, "label": "Booster stage (10–15 weeks)"},
    "pre_laying":  {"min_weeks": 16, "max_weeks": 20, "label": "Pre-laying stage (16–20 weeks)"},
    "laying":      {"min_weeks": 21, "max_weeks": 999,"label": "Laying stage (21+ weeks)"},
}

def get_age_window(age_weeks: int) -> str:
    """Return the age window key for a given flock age in weeks."""
    for key, w in FLOCK_AGE_WINDOWS.items():
        if w["min_weeks"] <= age_weeks <= w["max_weeks"]:
            return key
    return "laying"


# ── GHANA VSD VACCINATION SCHEDULE ───────────────────────────────────────────
# Based on Ghana VSD/MoFA official schedule + your father's protocol.
# key = flock age in days (approximate trigger point)
# This is used by vaccination_scheduler.py to send WhatsApp reminders.
VACCINATION_SCHEDULE = [
    {
        "id": "doc_arrival",
        "trigger_day": 1,
        "label": "Day-old chick arrival care",
        "instruction": (
            "Your day-old chicks have arrived! 🐥\n\n"
            "Give them *glucose + antibiotic vitamins* in their drinking water today.\n"
            "This reduces transport stress and gives them a strong start.\n\n"
            "Do NOT give them plain water on day 1."
        ),
        "drug": "Glucose powder + poultry multivitamins",
        "is_vaccine": False,
        "urgency": "important",
    },
    {
        "id": "gumboro_1",
        "trigger_day": 7,
        "label": "1st Gumboro vaccine",
        "instruction": (
            "⏰ *1st Gumboro vaccine due today (Day 7)*\n\n"
            "Buy *Gumboro Intermediate* or *Gumboro Plus* vaccine from your agrovet.\n"
            "Mix in drinking water. Withdraw water 2 hours before giving vaccine.\n\n"
            "⚠️ Store vaccine in fridge until use. Use within 2 hours of mixing."
        ),
        "drug": "Gumboro Intermediate / Gumboro Plus vaccine",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "newcastle_1",
        "trigger_day": 14,
        "label": "1st Newcastle vaccine (Lasota)",
        "instruction": (
            "⏰ *1st Newcastle vaccine due today (Day 14)*\n\n"
            "Buy *Lasota* (Newcastle Live Vaccine) from your agrovet.\n"
            "Mix in drinking water or give by eye drop.\n"
            "Mix with a small amount of skimmed milk powder to protect the virus.\n\n"
            "⚠️ Do NOT give alongside antibiotics — antibiotics can neutralise the vaccine."
        ),
        "drug": "Lasota Newcastle vaccine",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "gumboro_2",
        "trigger_day": 21,
        "label": "2nd Gumboro vaccine",
        "instruction": (
            "⏰ *2nd Gumboro vaccine due today (Day 21)*\n\n"
            "Same as before — Gumboro Intermediate or Gumboro Plus in drinking water.\n"
            "This booster is important. Your birds are entering the peak Gumboro\n"
            "danger window (weeks 3–6). Do not skip this."
        ),
        "drug": "Gumboro Intermediate / Gumboro Plus vaccine",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "newcastle_2",
        "trigger_day": 28,
        "label": "2nd Newcastle vaccine",
        "instruction": (
            "⏰ *2nd Newcastle vaccine due today (Day 28)*\n\n"
            "Give Lasota again in drinking water.\n\n"
            "After today: give birds *plain water* for a few days. "
            "No more medication in the water unless there is a problem."
        ),
        "drug": "Lasota Newcastle vaccine",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "cocci_stop_warning",
        "trigger_day": 98,  # Week 14 — warning before week 15 stop
        "label": "Coccidiosis medicine — stop soon",
        "instruction": (
            "⚠️ *Important: Stop coccidiosis medicine at Week 15*\n\n"
            "Your birds are approaching Week 15. Stop all anti-coccidial medicine\n"
            "at the end of this week.\n\n"
            "Continuing beyond Week 15 will interfere with egg production in layers.\n"
            "For broilers going to market before Week 15, observe withdrawal periods."
        ),
        "drug": None,
        "is_vaccine": False,
        "urgency": "important",
    },
    {
        "id": "newcastle_booster",
        "trigger_day": 70,  # Week 10
        "label": "Newcastle booster vaccine",
        "instruction": (
            "⏰ *Newcastle booster due this week (Week 10)*\n\n"
            "Give Lasota in drinking water again.\n"
            "This maintains immunity as maternal antibody protection fades completely."
        ),
        "drug": "Lasota Newcastle vaccine",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "newcastle_final",
        "trigger_day": 112,  # Week 16
        "label": "Final Newcastle injection (inactivated oil vaccine)",
        "instruction": (
            "⏰ *Final Newcastle vaccination due (Week 16)*\n\n"
            "This is the injectable oil-based inactivated Newcastle vaccine.\n"
            "It gives long-term immunity for laying birds.\n\n"
            "Contact your agrovet or VSD officer for the injection.\n"
            "Each bird must be injected individually (subcutaneous).\n\n"
            "After this, your laying birds are protected. Revaccinate every 3–4 months."
        ),
        "drug": "Newcastle inactivated oil-adjuvanted vaccine (injectable)",
        "is_vaccine": True,
        "urgency": "critical",
    },
    {
        "id": "fowl_pox",
        "trigger_day": 84,  # Week 12 — typical timing in Ghana
        "label": "Fowl Pox vaccine",
        "instruction": (
            "⏰ *Fowl Pox vaccine due this week (Week 12)*\n\n"
            "Apply by wing-web puncture (not drinking water).\n"
            "Ask your agrovet for the wing-web applicator needle.\n\n"
            "Check vaccinated birds 7–10 days later — a small scab at the\n"
            "injection site means the vaccine has taken. No scab = re-vaccinate."
        ),
        "drug": "Fowl Pox vaccine (wing-web application)",
        "is_vaccine": True,
        "urgency": "important",
    },
]

# ── COCCIDIOSIS DAILY MEDICINE SCHEDULE ──────────────────────────────────────
# Give 3 days per week, weeks 1–15.
# Protects intestinal lining from sharp grinded maize in feed.
# STOP at week 15 — affects egg production in layers.
COCCI_MEDICINE_WEEKS = {"start": 1, "stop": 15}
COCCI_DRUG = "Amprolium (Amprocox) or Sulphachlopyrazine (ESB3)"


# ── DISEASE SIGNS — WHAT EACH DISEASE LOOKS LIKE ────────────────────────────
# Used by the diagnosis engine.
# Based on your father's knowledge + veterinary science.
DISEASE_SIGNS = {
    "gumboro": {
        "name": "Gumboro Disease (Infectious Bursal Disease)",
        "local_name": "Gumboro",
        "age_window": (3, 6),          # Weeks — peak danger window
        "audio_signal": "weak",        # Birds go quiet, not noisy
        "droppings": "white_watery",
        "behavior": ["weak", "quiet", "huddled", "ruffled_feathers"],
        "has_cure": False,
        "urgency": "emergency",
        "kills_in": "3–7 days if untreated",
        "max_mortality": "up to 70%",
        "emoji": "🔴",
    },
    "newcastle": {
        "name": "Newcastle Disease",
        "local_name": "Newcastle",
        "age_window": (0, 999),        # Any age
        "audio_signal": "strong",      # Coughing, gasping, gurgling
        "droppings": "bright_green",
        "behavior": ["respiratory_distress", "twisted_neck", "sudden_deaths"],
        "has_cure": False,
        "urgency": "emergency",
        "kills_in": "2–5 days",
        "max_mortality": "up to 100%",
        "emoji": "🔴",
    },
    "coccidiosis": {
        "name": "Coccidiosis",
        "local_name": "Cocci",
        "age_window": (1, 15),         # Weeks — most dangerous under 15 weeks
        "audio_signal": "none",        # No audio signal
        "droppings": "bloody_chocolate",
        "behavior": ["slow_growth", "pale_comb", "reduced_appetite"],
        "has_cure": True,
        "urgency": "urgent",
        "kills_in": "opens door to Gumboro and other diseases if untreated",
        "max_mortality": "low directly, but enables cascade",
        "emoji": "🟠",
    },
    "crd": {
        "name": "Chronic Respiratory Disease (CRD)",
        "local_name": "Amaman (Twi) / CRD",
        "age_window": (0, 999),        # Any age
        "audio_signal": "strong",      # Coughing, sneezing, gurgling
        "droppings": "normal",         # No droppings change in early CRD
        "behavior": ["coughing", "sneezing", "nasal_discharge", "open_mouth"],
        "has_cure": True,
        "urgency": "urgent",
        "kills_in": "opens door to Newcastle if not treated",
        "max_mortality": "low directly — but becomes Newcastle without treatment",
        "emoji": "🟠",
    },
    "fowl_pox": {
        "name": "Fowl Pox",
        "local_name": "Fowl Pox",
        "age_window": (0, 999),        # Any age
        "audio_signal": "none",
        "droppings": "normal",
        "behavior": ["face_lesions", "comb_lesions", "eye_lesions", "reduced_eating"],
        "has_cure": False,             # No cure, but manageable
        "urgency": "monitor",
        "kills_in": "rarely fatal — reduces production and spreads slowly",
        "max_mortality": "low",
        "emoji": "🟡",
    },
}

# ── AGROVET DRUG GUIDE ────────────────────────────────────────────────────────
# What to tell the farmer to buy and how to use it.
# Based on what is actually available in Ghanaian agrovet shops.
AGROVET_DRUGS = {
    "coccidiosis": {
        "drug_names": ["Amprolium (Amprocox)", "Sulphachlopyrazine (ESB3)", "Toltrazuril (Baycox)"],
        "instruction": (
            "Mix in drinking water for 3–5 days.\n"
            "Follow the package dosage instructions carefully.\n"
            "Also add vitamin K to the water (helps stop intestinal bleeding).\n"
            "Remove medicated water at night — give plain water."
        ),
        "agrovet_message": "Tell the agrovet: 'I need medicine for Coccidiosis — bloody droppings in my birds.'"
    },
    "crd": {
        "drug_names": ["Tylosin 50 (Tylan Soluble)", "Doxycycline", "Oxytetracycline"],
        "instruction": (
            "Mix in drinking water for 5 days.\n"
            "Tylosin 50 at 0.5g per litre of water.\n"
            "Also add multivitamins to support recovery.\n"
            "Fix the problem that caused CRD — clean drinkers, improve ventilation,\n"
            "remove wet litter immediately."
        ),
        "agrovet_message": "Tell the agrovet: 'I need Tylosin or Doxycycline — my birds are coughing and sneezing.'"
    },
    "gumboro_supportive": {
        "drug_names": ["Poultry electrolytes + vitamins", "Vitamin C", "Liver tonic"],
        "instruction": (
            "There is NO cure for Gumboro. Treatment is supportive only.\n"
            "Give electrolytes + vitamins in water immediately.\n"
            "Keep the house warm and dry.\n"
            "Remove dead birds immediately — they spread the virus.\n"
            "Do NOT give antibiotics — they do nothing against Gumboro virus."
        ),
        "agrovet_message": "Tell the agrovet: 'My 3–6 week birds are very weak — I need poultry electrolytes and vitamins urgently.'"
    },
    "fowl_pox_supportive": {
        "drug_names": ["Iodine solution (topical)", "Vitamin A supplement", "Antibiotics (for secondary infection only)"],
        "instruction": (
            "No cure exists — birds usually recover in 3–4 weeks.\n"
            "Apply iodine solution to lesions on face/comb.\n"
            "Separate badly affected birds from the flock.\n"
            "Give Vitamin A in water — supports skin healing.\n"
            "If you have unaffected birds: vaccinate them NOW to prevent spread."
        ),
        "agrovet_message": "Tell the agrovet: 'My birds have scabs and sores on their face — I need Fowl Pox vaccine and iodine solution.'"
    },
}

# ── SEASONAL RISK DATA ────────────────────────────────────────────────────────
# From Mensah et al. (2023) PAMJ-One Health — KNUST VS Lab 2018–2021
MONTHLY_RISK_DATA = {
    1:  {"level": "Medium",  "pct": 27.0,  "label": "January"},
    2:  {"level": "Medium",  "pct": 26.0,  "label": "February"},
    3:  {"level": "Lowest",  "pct": 22.97, "label": "March"},
    4:  {"level": "Low",     "pct": 25.0,  "label": "April"},
    5:  {"level": "Low",     "pct": 26.0,  "label": "May"},
    6:  {"level": "Medium",  "pct": 27.0,  "label": "June"},
    7:  {"level": "Medium",  "pct": 28.0,  "label": "July"},
    8:  {"level": "Medium",  "pct": 29.0,  "label": "August"},
    9:  {"level": "High",    "pct": 31.86, "label": "September"},
    10: {"level": "High",    "pct": 33.0,  "label": "October"},
    11: {"level": "Highest", "pct": 34.27, "label": "November"},
    12: {"level": "High",    "pct": 32.0,  "label": "December"},
}

# ── WHATSAPP ──────────────────────────────────────────────────────────────────
WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_API_URL = (
    f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
)

# ── VSD CONTACTS ──────────────────────────────────────────────────────────────
VSD_CONTACTS = {
    "Ashanti":       {"office": "VSD Ashanti Regional Office, Kumasi",       "phone": "0322-024567"},
    "Greater Accra": {"office": "VSD Greater Accra Regional Office",          "phone": "0302-665567"},
    "Northern":      {"office": "VSD Northern Regional Office, Tamale",       "phone": "0372-022345"},
    "Western":       {"office": "VSD Western Regional Office, Takoradi",      "phone": "0312-023456"},
    "Eastern":       {"office": "VSD Eastern Regional Office, Koforidua",     "phone": "0342-022234"},
    "Central":       {"office": "VSD Central Regional Office, Cape Coast",    "phone": "0332-133456"},
    "Volta":         {"office": "VSD Volta Regional Office, Ho",              "phone": "0362-027890"},
    "Brong-Ahafo":   {"office": "VSD Brong-Ahafo Regional Office, Sunyani",  "phone": "0352-027123"},
    "Upper East":    {"office": "VSD Upper East Regional Office, Bolgatanga", "phone": "0382-022345"},
    "Upper West":    {"office": "VSD Upper West Regional Office, Wa",         "phone": "0392-022678"},
}

# ── ONBOARDING QUESTIONS ──────────────────────────────────────────────────────
# These build the farmer's profile in a single conversation.
# Added: flock_age_weeks — the most important field for disease risk.
ONBOARDING_QUESTIONS = [
    {
        "key": "region",
        "question": (
            "Welcome to KokoAlert 🐔\n\n"
            "I will ask you 7 quick questions to set up your farm profile. "
            "This only happens once.\n\n"
            "*Question 1 of 7 — Which region are you in?*\n\n"
            "1 — Ashanti\n"
            "2 — Greater Accra\n"
            "3 — Northern\n"
            "4 — Western\n"
            "5 — Eastern\n"
            "6 — Other"
        ),
        "options": {
            "1": "Ashanti", "2": "Greater Accra", "3": "Northern",
            "4": "Western", "5": "Eastern", "6": "Other"
        }
    },
    {
        "key": "flock_size",
        "question": (
            "*Question 2 of 7 — How many birds do you currently have?*\n\n"
            "1 — Under 100\n"
            "2 — 100 to 500\n"
            "3 — 500 to 2000\n"
            "4 — Over 2000"
        ),
        "options": {
            "1": "under_100", "2": "100_to_500",
            "3": "500_to_2000", "4": "over_2000"
        }
    },
    {
        "key": "bird_type",
        "question": (
            "*Question 3 of 7 — What type of birds do you keep?*\n\n"
            "1 — Broilers (for meat)\n"
            "2 — Layers (for eggs)\n"
            "3 — Both broilers and layers"
        ),
        "options": {"1": "broiler", "2": "layer", "3": "both"}
    },
    {
        "key": "doc_arrival_date",
        "question": (
            "*Question 4 of 7 — When did your current flock of day-old chicks arrive?*\n\n"
            "Reply with a number for how old your birds are now:\n"
            "1 — This week (0–7 days old)\n"
            "2 — 1–2 weeks old\n"
            "3 — 3–4 weeks old\n"
            "4 — 5–8 weeks old\n"
            "5 — 9–15 weeks old\n"
            "6 — Over 16 weeks old / laying birds"
        ),
        "options": {
            "1": 0, "2": 7, "3": 21, "4": 35, "5": 70, "6": 112
        }  # Approximate age in days — used to calculate flock_age_weeks
    },
    {
        "key": "gumboro_vaccinated",
        "question": (
            "*Question 5 of 7 — Has your flock received the Gumboro vaccine?*\n\n"
            "1 — Yes — both 1st and 2nd dose given\n"
            "2 — Yes — only 1st dose given\n"
            "3 — No / not sure"
        ),
        "options": {"1": "both", "2": "first_only", "3": "none"}
    },
    {
        "key": "newcastle_vaccinated",
        "question": (
            "*Question 6 of 7 — Has your flock received the Newcastle (Lasota) vaccine?*\n\n"
            "1 — Yes — fully vaccinated\n"
            "2 — Partially vaccinated\n"
            "3 — No / not sure"
        ),
        "options": {"1": "full", "2": "partial", "3": "none"}
    },
    {
        "key": "ventilation",
        "question": (
            "*Question 7 of 7 — How is the ventilation inside your poultry house?*\n\n"
            "1 — Good — windows open, good airflow\n"
            "2 — Medium — some airflow\n"
            "3 — Poor — hot and stuffy inside"
        ),
        "options": {"1": "good", "2": "medium", "3": "poor"}
    },
]
