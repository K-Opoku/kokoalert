"""
src/diagnosis_engine.py
─────────────────────────────────────────────────────────────────────────────
The brain of KokoAlert.

This module takes ALL available signals and produces a confirmed diagnosis
with detailed reasons — not just "something is wrong."

Signal sources it combines:
  1. Audio classifier result  — CNN detected respiratory anomaly (yes/no + confidence)
  2. Droppings colour         — bloody, green, white/watery, or normal
  3. Behaviour signs          — coughing, weak/quiet, face lesions, deaths
  4. Flock age (weeks)        — the single most important variable
  5. Vaccination status       — which vaccines given, which missed
  6. Season                   — month-based risk from KNUST data
  7. Farm conditions          — ventilation, wet areas (CRD risk)

Disease priority order (when multiple diseases are possible):
  1. Gumboro   — fastest killer in the danger window (3–6 weeks)
  2. Newcastle  — no cure, must contain immediately
  3. Coccidiosis — treatable but opens door to Gumboro
  4. CRD        — treatable but opens door to Newcastle
  5. Fowl Pox   — manageable, rarely fatal

The confirmed diagnosis message includes:
  - Disease name (English + local name where applicable)
  - Confidence level + explanation of confidence
  - Bullet-point reasons WHY we believe this is the disease
  - Urgency level
  - Exact actions for the next 2 hours
  - Agrovet instruction (exact drug name to ask for)
  - Whether VSD must be contacted
"""
from src.config import IMAGE_TO_DROPPINGS_MAP, IMAGE_CONFIDENCE_THRESHOLD
from datetime import datetime
from src.config import (
    MONTHLY_RISK_DATA, DISEASE_SIGNS, AGROVET_DRUGS,
    VSD_CONTACTS, get_age_window, FLOCK_AGE_WINDOWS
)


# ── CONFIDENCE SCORING ────────────────────────────────────────────────────────

def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    else:
        return "Low"


# ── INDIVIDUAL DISEASE CHECKS ─────────────────────────────────────────────────

def _check_gumboro(
    flock_age_weeks: int,
    droppings: str,
    behavior: list,
    gumboro_vaccinated: str,
    audio_anomalous: bool,
) -> dict | None:
    """
    Gumboro check.
    Peak danger: weeks 3–6. White/watery droppings. Weak, quiet birds.
    No treatment — only prevention. Fastest killer.
    """
    reasons = []
    score = 0.0

    # Age window — the strongest signal for Gumboro
    if 3 <= flock_age_weeks <= 6:
        reasons.append(
            f"Your birds are *{flock_age_weeks} weeks old* — this is the peak Gumboro "
            f"danger window (weeks 3–6). Gumboro strikes hardest at this exact age."
        )
        score += 0.45
    elif flock_age_weeks < 3:
        reasons.append(
            f"Your birds are *{flock_age_weeks} weeks old* — Gumboro can still cause "
            f"subclinical damage at this age (no visible signs but immune system is harmed)."
        )
        score += 0.15
    else:
        return None  # Gumboro clinical disease is unlikely after week 6

    # Droppings — white/watery is the classic Gumboro sign
    if droppings == "white_watery":
        reasons.append(
            "White, watery droppings reported — this is the *classic sign of Gumboro*. "
            "The virus attacks the digestive and immune organs, causing this type of diarrhoea."
        )
        score += 0.30

    # Behaviour — birds go very weak and quiet
    if "weak" in behavior or "quiet" in behavior or "huddled" in behavior:
        reasons.append(
            "Birds appear weak, quiet, or huddled together — Gumboro causes sudden severe "
            "depression. Affected birds often sit with their heads down and stop eating."
        )
        score += 0.15

    # Vaccination status — unvaccinated birds are wide open
    if gumboro_vaccinated == "none":
        reasons.append(
            "Your flock has NOT received the Gumboro vaccine — *unprotected birds in the "
            "3–6 week window face up to 70% mortality from virulent strains.*"
        )
        score += 0.20
    elif gumboro_vaccinated == "first_only":
        reasons.append(
            "Only the 1st Gumboro dose has been given — *the 2nd dose is required for full "
            "protection.* Partial vaccination still leaves birds vulnerable."
        )
        score += 0.10

    # Audio signal — Gumboro birds go quiet (weak signal)
    if audio_anomalous:
        reasons.append(
            "KokoAlert detected unusual sounds in the flock audio — "
            "Gumboro birds may produce abnormal vocalisations from distress."
        )
        score += 0.05

    if score < 0.45:
        return None  # Not enough evidence

    return {
        "disease": "gumboro",
        "confidence_score": min(score, 0.97),
        "reasons": reasons,
    }


def _check_newcastle(
    flock_age_weeks: int,
    droppings: str,
    behavior: list,
    newcastle_vaccinated: str,
    audio_anomalous: bool,
    audio_probability: float,
    month: int,
    image_result: dict = None,
) -> dict | None:
    """
    Newcastle check.
    Bright green droppings is the definitive sign.
    Strong audio signal. Any age. No cure.
    """
    reasons = []
    score = 0.0

    # Bright green droppings — the strongest Newcastle signal
    if droppings == "bright_green":
        reasons.append(
            "Bright green or yellow-green droppings reported — *this is the most "
            "recognisable sign of Newcastle Disease.* The virus attacks the gut and "
            "bile system, causing this characteristic green diarrhoea."
        )
        score += 0.55
    if image_result and image_result.get("reliable") and image_result.get("class") == "newcastle":
        reasons.append("Photo of droppings confirmed Newcastle — image analysis agrees with your observation.")
        score += 0.15
 
    
    # Audio anomaly — Newcastle causes coughing, gasping, gurgling
    if audio_anomalous:
        reasons.append(
            f"KokoAlert detected abnormal respiratory sounds (confidence: "
            f"{audio_probability:.0%}) — Newcastle causes severe respiratory distress "
            f"including coughing, gasping, and gurgling sounds."
        )
        score += 0.25

    # Behaviour
    if "respiratory_distress" in behavior or "twisted_neck" in behavior:
        reasons.append(
            "Respiratory distress or twisted neck (torticollis) reported — "
            "these are known signs of virulent Newcastle Disease affecting the nervous system."
        )
        score += 0.20

    if "sudden_deaths" in behavior:
        reasons.append(
            "Sudden bird deaths reported — Newcastle can cause 50–100% mortality "
            "in unvaccinated flocks within days of outbreak."
        )
        score += 0.15

    # Vaccination status
    if newcastle_vaccinated == "none":
        reasons.append(
            "Your flock has NOT been vaccinated against Newcastle — "
            "*unvaccinated birds face up to 100% mortality in an outbreak.*"
        )
        score += 0.15
    elif newcastle_vaccinated == "partial":
        reasons.append(
            "Partial Newcastle vaccination recorded — incomplete vaccination "
            "leaves significant portions of the flock unprotected."
        )
        score += 0.08

    # Seasonal risk
    monthly = MONTHLY_RISK_DATA[month]
    if monthly["level"] in ["High", "Highest"]:
        reasons.append(
            f"*{monthly['label']} is a {monthly['level'].lower()} risk month* — "
            f"{monthly['pct']}% of Ashanti's annual respiratory disease cases occur this month "
            f"(KNUST VS Lab, 2018–2021). Newcastle peaks November–December in Ghana."
        )
        score += 0.10

    # CRD cascade — Newcastle is more likely if CRD is present
    if "coughing" in behavior and droppings != "bright_green":
        # Already covered by CRD check — don't double-count here
        pass

    if score < 0.30:
        return None

    return {
        "disease": "newcastle",
        "confidence_score": min(score, 0.97),
        "reasons": reasons,
    }


def _check_coccidiosis(
    flock_age_weeks: int,
    droppings: str,
    behavior: list,
    cocci_medicine_given: bool,
    image_result: dict = None,
) -> dict | None:
    """
    Coccidiosis check.
    Bloody or dark chocolate droppings. Age under 15 weeks.
    Treatable — but if missed, opens door to Gumboro and other diseases.
    """
    reasons = []
    score = 0.0

    # Droppings — definitive sign
    if droppings == "bloody_chocolate":
        reasons.append(
            "Bloody or dark chocolate-coloured droppings reported — *this is the "
            "classic sign of Coccidiosis.* The parasite (Eimeria) destroys the lining "
            "of the intestines, causing bleeding into the droppings."
        )
        score += 0.60
    if image_result and image_result.get("reliable") and image_result.get("class") == "coccidiosis":
        reasons.append("Photo of droppings confirmed Coccidiosis — image analysis agrees with your observation.")
        score += 0.15    

    # Age — coccidiosis is most dangerous and common under 15 weeks
    if flock_age_weeks <= 8:
        reasons.append(
            f"Your birds are *{flock_age_weeks} weeks old* — Coccidiosis is most severe "
            f"in young birds. The sharp grinded maize in starter feed can damage the "
            f"intestinal lining, making infection worse."
        )
        score += 0.20
    elif flock_age_weeks <= 15:
        reasons.append(
            f"Your birds are {flock_age_weeks} weeks old — still within the high-risk "
            f"Coccidiosis period (weeks 1–15)."
        )
        score += 0.10
    elif droppings != "bloody_chocolate":
        return None  # Coccidiosis very unlikely after week 15 without blood in droppings

    # No preventive medicine given
    if not cocci_medicine_given:
        reasons.append(
            "No Coccidiosis prevention medicine has been recorded — birds should receive "
            "anti-coccidial medicine 3 days per week from week 1 to week 15."
        )
        score += 0.15

    # Gumboro cascade warning
    if 3 <= flock_age_weeks <= 6:
        reasons.append(
            "⚠️ *Cascade risk:* Your birds are in the Gumboro danger window (3–6 weeks). "
            "Coccidiosis weakens the immune system — if not treated, Gumboro can follow."
        )

    # Behaviour
    if "reduced_appetite" in behavior or "slow_growth" in behavior:
        reasons.append(
            "Reduced appetite or slow growth reported — Coccidiosis impairs nutrient "
            "absorption as it destroys the intestinal lining."
        )
        score += 0.10

    if score < 0.30:
        return None

    return {
        "disease": "coccidiosis",
        "confidence_score": min(score, 0.97),
        "reasons": reasons,
    }


def _check_crd(
    audio_anomalous: bool,
    audio_probability: float,
    behavior: list,
    ventilation: str,
    droppings: str,
) -> dict | None:
    """
    CRD check.
    Coughing + sneezing + poor ventilation + no green droppings.
    Treatable with antibiotics. If untreated, opens door to Newcastle.
    The local Twi name is 'Amaman' — visible infection when you open bird's mouth.
    """
    reasons = []
    score = 0.0

    # Audio is the primary signal for CRD
    if audio_anomalous:
        reasons.append(
            f"KokoAlert detected abnormal sounds in the flock audio "
            f"(confidence: {audio_probability:.0%}) — "
            f"CRD (Chronic Respiratory Disease) causes coughing, sneezing, and "
            f"gurgling sounds (called *'Amaman'* in Twi)."
        )
        score += 0.35

    # Coughing/sneezing behaviour
    if "coughing" in behavior:
        reasons.append(
            "Coughing reported in the flock — this is the primary symptom of CRD "
            "(Mycoplasma gallisepticum infection)."
        )
        score += 0.25

    if "sneezing" in behavior:
        reasons.append(
            "Sneezing reported — nasal irritation and discharge are characteristic "
            "of CRD. If you open an affected bird's mouth, you may see yellowish "
            "infection inside the throat (Amaman)."
        )
        score += 0.15

    if "nasal_discharge" in behavior:
        reasons.append(
            "Nasal discharge observed — CRD causes watery or foamy discharge from "
            "the nostrils and around the eyes."
        )
        score += 0.10

    # Environmental risk factors — CRD is triggered by wet conditions
    if ventilation == "poor":
        reasons.append(
            "Poor ventilation recorded in your farm profile — *CRD thrives in hot, "
            "stuffy, poorly ventilated houses.* Ammonia build-up from litter damages "
            "the respiratory mucosa, letting Mycoplasma take hold."
        )
        score += 0.20
    elif ventilation == "medium":
        reasons.append(
            "Medium ventilation recorded — improving airflow reduces CRD risk significantly."
        )
        score += 0.08

    # Green droppings would suggest Newcastle, not CRD — reduce score
    if droppings == "bright_green":
        score -= 0.30  # CRD alone doesn't cause green droppings

    # CRD → Newcastle cascade warning
    if score > 0.40:
        reasons.append(
            "⚠️ *Cascade risk:* CRD weakens the respiratory tract and immune response. "
            "If not treated, *Newcastle Disease can follow.* Treat CRD now."
        )

    if score < 0.30:
        return None

    return {
        "disease": "crd",
        "confidence_score": min(score, 0.93),
        "reasons": reasons,
    }


def _check_fowl_pox(behavior: list) -> dict | None:
    """
    Fowl Pox check.
    Visible lesions on face, comb, or around eyes.
    No audio signal, no droppings change.
    Manageable — birds usually recover. Vaccine available.
    """
    reasons = []
    score = 0.0

    if "face_lesions" in behavior or "comb_lesions" in behavior or "eye_lesions" in behavior:
        reasons.append(
            "Scabs, warts, or sores reported on birds' face, comb, or around the eyes — "
            "*this is the defining sign of Fowl Pox.* The poxvirus causes these crusty "
            "lesions on unfeathered skin."
        )
        score += 0.75

    if "reduced_eating" in behavior and ("face_lesions" in behavior or "comb_lesions" in behavior):
        reasons.append(
            "Reduced eating alongside facial lesions — in the wet form of Fowl Pox, "
            "lesions can develop inside the mouth and throat, making eating painful."
        )
        score += 0.15

    if score < 0.50:
        return None

    return {
        "disease": "fowl_pox",
        "confidence_score": min(score, 0.95),
        "reasons": reasons,
    }


# ── MAIN DIAGNOSIS FUNCTION ───────────────────────────────────────────────────

def run_diagnosis(
    farm_profile: dict,
    audio_result: dict,
    symptoms: dict,
    image_result: dict = None,    # ← add this
) -> dict:
    """
    Run the full diagnosis combining all available signals.

    Args:
        farm_profile: Farmer's profile from database
            Required keys: region, flock_age_weeks (or doc_arrival_days),
                           gumboro_vaccinated, newcastle_vaccinated, ventilation
        audio_result: From pipeline.py
            Keys: is_anomalous (bool), probability (float)
        symptoms: From WhatsApp daily/weekly check
            Keys: droppings (str), behavior (list of str), cocci_medicine_given (bool)

    Returns:
        Full diagnosis dict with disease, confidence, reasons, actions, agrovet instruction
    """
    # ── Extract inputs ────────────────────────────────────────────────────────
    month = datetime.now().month
    monthly = MONTHLY_RISK_DATA[month]

    # Flock age — most important variable
    flock_age_weeks = farm_profile.get("flock_age_weeks", 0)
    if flock_age_weeks == 0 and "doc_arrival_days" in farm_profile:
        flock_age_weeks = farm_profile["doc_arrival_days"] // 7

    region = farm_profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    audio_anomalous = audio_result.get("is_anomalous", False)
    audio_probability = audio_result.get("probability", 0.0)

    droppings = symptoms.get("droppings", "normal")
    behavior = symptoms.get("behavior", [])
    cocci_medicine_given = symptoms.get("cocci_medicine_given", False)

    gumboro_vaccinated = farm_profile.get("gumboro_vaccinated", "none")
    newcastle_vaccinated = farm_profile.get("newcastle_vaccinated", "none")
    ventilation = farm_profile.get("ventilation", "medium")

    # ── Resolve final droppings using image if provided ───────────────────
    image_used = False
    image_confirmed = False

    if image_result and image_result.get("image_provided"):
        image_class = image_result["class"]
        image_droppings = IMAGE_TO_DROPPINGS_MAP[image_class]
        reliable = image_result.get("reliable", False)

        if reliable and image_droppings == droppings:
            image_confirmed = True
            image_used = True
        elif reliable and image_droppings != droppings:
            droppings = image_droppings
            image_used = True

    # ── Run all disease checks ────────────────────────────────────────────
    # ── Run all disease checks ────────────────────────────────────────────────
    candidates = []

    gumboro = _check_gumboro(
        flock_age_weeks, droppings, behavior, gumboro_vaccinated, audio_anomalous
    )
    if gumboro:
        candidates.append(gumboro)

    newcastle = _check_newcastle(
        flock_age_weeks, droppings, behavior, newcastle_vaccinated,
        audio_anomalous, audio_probability, month, image_result=image_result,
    )
    if newcastle:
        candidates.append(newcastle)

    coccidiosis = _check_coccidiosis(
        flock_age_weeks, droppings, behavior, cocci_medicine_given, image_result=image_result,
    )
    if coccidiosis:
        candidates.append(coccidiosis)

    crd = _check_crd(
        audio_anomalous, audio_probability, behavior, ventilation, droppings
    )
    if crd:
        candidates.append(crd)

    fowl_pox = _check_fowl_pox(behavior)
    if fowl_pox:
        candidates.append(fowl_pox)

    # ── No disease detected ───────────────────────────────────────────────────
    if not candidates and not audio_anomalous:
        return _healthy_result(flock_age_weeks, monthly)

    # If audio anomaly but no other signals → flag but don't diagnose
    if not candidates and audio_anomalous:
        return _audio_only_result(audio_probability, flock_age_weeks, vsd)

    # ── Select primary diagnosis (highest confidence score) ───────────────────
    # Gumboro gets a priority boost in the danger window — it's the fastest killer
    for c in candidates:
        if c["disease"] == "gumboro" and 3 <= flock_age_weeks <= 6:
            c["confidence_score"] = min(c["confidence_score"] + 0.05, 0.97)

    primary = max(candidates, key=lambda x: x["confidence_score"])
    secondary = [c for c in candidates if c["disease"] != primary["disease"]]

    # ── Build the full diagnosis response ─────────────────────────────────────
    return _build_diagnosis_response(
        primary=primary,
        secondary=secondary,
        flock_age_weeks=flock_age_weeks,
        monthly=monthly,
        vsd=vsd,
        farm_profile=farm_profile,
        image_used=image_used,
    )


# ── RESPONSE BUILDERS ─────────────────────────────────────────────────────────

def _build_diagnosis_response(
    primary: dict,
    secondary: list,
    flock_age_weeks: int,
    monthly: dict,
    vsd: dict,
    farm_profile: dict,
    image_used: bool = False,
) -> dict:
    """Build the full diagnosis response dict."""
    disease_key = primary["disease"]
    disease_info = DISEASE_SIGNS[disease_key]
    confidence = _confidence_label(primary["confidence_score"])
    reasons = primary["reasons"]

    # Build the WhatsApp message
    msg_parts = []

    # Header
    msg_parts.append(
        f"{disease_info['emoji']} *DIAGNOSIS: {disease_info['name']}*\n"
        f"Confidence: *{confidence}*\n"
    )

    # Why we believe this
    msg_parts.append("*Why KokoAlert believes this:*")
    for i, reason in enumerate(reasons, 1):
        msg_parts.append(f"{i}. {reason}")

    msg_parts.append("")

    # Urgency + what to do
    urgency = disease_info["urgency"]

    if urgency == "emergency":
        msg_parts.append("🚨 *THIS IS AN EMERGENCY*")
        msg_parts.append(f"This disease can kill {disease_info['max_mortality']} of your flock in {disease_info['kills_in']}.")
        msg_parts.append("")

    # Disease-specific actions
    actions = _get_actions(disease_key, farm_profile, vsd)
    msg_parts.append("*What to do RIGHT NOW:*")
    msg_parts.append(actions["immediate"])

    # Agrovet instruction if applicable
    if disease_key in AGROVET_DRUGS:
        drug_info = AGROVET_DRUGS[disease_key] if disease_key in AGROVET_DRUGS else None
        if not drug_info and disease_key == "gumboro":
            drug_info = AGROVET_DRUGS["gumboro_supportive"]
        if not drug_info and disease_key == "fowl_pox":
            drug_info = AGROVET_DRUGS["fowl_pox_supportive"]
        if drug_info:
            msg_parts.append("")
            msg_parts.append(f"🏪 *Go to your agrovet:*")
            msg_parts.append(drug_info["agrovet_message"])

    # VSD contact if needed
    if actions.get("contact_vsd"):
        msg_parts.append("")
        msg_parts.append(
            f"📞 *Contact VSD NOW:*\n"
            f"{vsd['office']}\n"
            f"{vsd['phone']}"
        )

    # Secondary findings
    if secondary:
        sec_names = [DISEASE_SIGNS[s["disease"]]["name"] for s in secondary]
        msg_parts.append("")
        msg_parts.append(
            f"⚠️ *Also possible:* {', '.join(sec_names)}\n"
            f"Treating the primary diagnosis first will address these risks too."
        )

    full_message = "\n".join(msg_parts)
    full_message += "\n\n_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"

    return {
        "status": "diagnosed",
        "disease": disease_key,
        "disease_name": disease_info["name"],
        "confidence": confidence,
        "confidence_score": primary["confidence_score"],
        "urgency": urgency,
        "has_cure": disease_info["has_cure"],
        "reasons": reasons,
        "secondary_diseases": [s["disease"] for s in secondary],
        "contact_vsd": actions.get("contact_vsd", False),
        "whatsapp_message": full_message,
        "flock_age_weeks": flock_age_weeks,
        "monthly_data": monthly,
        "image_used": image_used,

    }


def _get_actions(disease_key: str, farm_profile: dict, vsd: dict) -> dict:
    """Return disease-specific immediate action instructions."""

    if disease_key == "gumboro":
        return {
            "immediate": (
                "1. Go to your agrovet NOW — ask for *poultry electrolytes and vitamins*\n"
                "2. Mix in drinking water immediately\n"
                "3. Keep the house *warm and dry*\n"
                "4. Remove dead birds immediately — they spread the virus\n"
                "5. Do NOT sell or move any birds\n"
                "6. Do NOT give antibiotics — they do nothing against Gumboro virus\n"
                "7. Count dead birds every 3 hours and reply with the number"
            ),
            "contact_vsd": False,
            "note": "There is no cure. Your only job is to support the surviving birds."
        }

    elif disease_key == "newcastle":
        return {
            "immediate": (
                "1. Do NOT sell or move any birds — Newcastle spreads through movement\n"
                "2. Keep all visitors and vehicles out of the farm\n"
                "3. Give *electrolytes + Vitamin C* in drinking water now\n"
                "4. Contact VSD immediately — Newcastle is a notifiable disease\n"
                "5. Remove and safely bury or burn dead birds\n"
                "6. There is no cure — focus on containment"
            ),
            "contact_vsd": True,
        }

    elif disease_key == "coccidiosis":
        drug = AGROVET_DRUGS["coccidiosis"]
        return {
            "immediate": (
                f"1. Go to your agrovet NOW. Ask for: *{drug['drug_names'][0]}*\n"
                f"2. {drug['instruction']}\n"
                "3. Also add *Vitamin K* to the water — reduces intestinal bleeding\n"
                "4. Check your litter — remove any wet or dirty areas\n"
                "5. Clean all drinkers — dirty water spreads Coccidiosis\n"
                "6. If your birds are in the Gumboro window (3–6 weeks): watch closely "
                "for birds going very weak — Coccidiosis can open the door to Gumboro"
            ),
            "contact_vsd": False,
        }

    elif disease_key == "crd":
        drug = AGROVET_DRUGS["crd"]
        return {
            "immediate": (
                f"1. Go to your agrovet NOW. Ask for: *{drug['drug_names'][0]}*\n"
                f"2. {drug['instruction']}\n"
                "3. Check your house for *wet areas, leaking drinkers, poor airflow* — "
                "fix these immediately. CRD is triggered by bad conditions.\n"
                "4. Open windows and improve ventilation\n"
                "5. Do not vaccinate while birds are sick — wait until recovered\n"
                "6. Treat now — if CRD is ignored, Newcastle can follow"
            ),
            "contact_vsd": False,
        }

    elif disease_key == "fowl_pox":
        drug = AGROVET_DRUGS["fowl_pox_supportive"]
        return {
            "immediate": (
                f"1. Separate badly affected birds from the rest of the flock\n"
                f"2. Go to your agrovet. Ask for: *iodine solution and Vitamin A supplement*\n"
                f"3. Apply iodine to the lesions on affected birds\n"
                f"4. Vaccinate all healthy, unaffected birds NOW — wing-web puncture\n"
                f"5. Most birds recover in 3–4 weeks with supportive care\n"
                f"6. Fowl Pox is spread by mosquitoes and direct contact — "
                f"reduce standing water near the farm to reduce mosquito breeding"
            ),
            "contact_vsd": False,
        }

    return {"immediate": "Monitor closely and contact your agrovet.", "contact_vsd": False}


def _healthy_result(flock_age_weeks: int, monthly: dict) -> dict:
    """Return a healthy result with age-appropriate advice."""
    age_window = get_age_window(flock_age_weeks)
    window_info = FLOCK_AGE_WINDOWS[age_window]

    age_advice = ""
    if age_window == "gumboro":
        age_advice = (
            "\n\n⚠️ *Watch closely this week:* Your birds are in the Gumboro danger window "
            f"({flock_age_weeks} weeks old). Even with healthy sounds, check droppings daily "
            "for any white/watery diarrhoea."
        )
    elif age_window == "chick":
        age_advice = (
            f"\n\n💧 Remember: Give coccidiosis medicine 3 days this week. "
            f"Birds are {flock_age_weeks} weeks old — intestinal protection is important."
        )

    msg = (
        f"✅ *Your flock sounds healthy.*\n\n"
        f"No respiratory anomaly detected in the audio.\n"
        f"Flock age: *{flock_age_weeks} weeks* ({window_info['label']})\n"
        f"Season risk: *{monthly['level']}* ({monthly['label']})"
        f"{age_advice}\n\n"
        f"Send a voice note anytime for another check.\n"
        f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
    )

    return {
        "status": "healthy",
        "disease": None,
        "urgency": "none",
        "whatsapp_message": msg,
        "flock_age_weeks": flock_age_weeks,
        "monthly_data": monthly,
    }


def _audio_only_result(audio_probability: float, flock_age_weeks: int, vsd: dict) -> dict:
    """
    Audio anomaly detected but no supporting symptoms reported.
    Give a cautious alert and ask the farmer to check droppings.
    """
    msg = (
        f"⚠️ *Abnormal sounds detected in your flock audio*\n"
        f"Confidence: {audio_probability:.0%}\n\n"
        f"No additional symptoms have been reported. Before I can give you a "
        f"confirmed diagnosis, I need you to check one thing:\n\n"
        f"*What do the droppings look like today?*\n\n"
        f"1 — Normal (brown/dark)\n"
        f"2 — Bloody or dark chocolate coloured\n"
        f"3 — Bright green or yellow-green\n"
        f"4 — White and very watery\n\n"
        f"Reply with 1, 2, 3, or 4."
    )

    return {
        "status": "needs_symptoms",
        "disease": None,
        "urgency": "monitor",
        "whatsapp_message": msg,
        "flock_age_weeks": flock_age_weeks,
        "awaiting": "droppings_check",
    }
