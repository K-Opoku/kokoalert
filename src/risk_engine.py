from datetime import datetime
from src.config import MONTHLY_RISK_DATA, VSD_CONTACTS


# ── SCORING WEIGHTS ───────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "not_vaccinated":         35,
    "vaccination_overdue":    20,
    "peak_season":            15,
    "gumboro_window":         20,   # NEW: age 3–6 weeks is the single highest-risk period
    "newcastle_booster_due":  15,   # NEW: past week 10, Newcastle booster missed
    "recent_deaths":          10,
    "new_birds_introduced":    8,
    "no_footbath":             5,
    "poor_ventilation":        4,
    "large_flock":             3,
}

RISK_CATEGORIES = [
    (75, 100, "Critical", "🔴"),
    (50,  74, "High",     "🟠"),
    (25,  49, "Medium",   "🟡"),
    (0,   24, "Low",      "🟢"),
]


def compute_farm_risk_score(farm_profile: dict) -> dict:
    score = 0
    risk_factors = []
    month = farm_profile.get("current_month", datetime.now().month)
    monthly = MONTHLY_RISK_DATA[month]

    # ── FLOCK AGE — the most important dynamic risk factor ────────────────
    flock_age_weeks = farm_profile.get("flock_age_weeks", 0)

    if 3 <= flock_age_weeks <= 6:
        # Gumboro danger window — the most deadly age period
        score += RISK_WEIGHTS["gumboro_window"]
        risk_factors.append({
            "factor": (
                f"Flock is {flock_age_weeks} weeks old — inside the Gumboro "
                f"danger window (weeks 3–6)"
            ),
            "points": RISK_WEIGHTS["gumboro_window"],
            "action": (
                "Monitor every morning for white/watery droppings and weak birds. "
                "Confirm 2nd Gumboro vaccine was given at week 3. "
                "Gumboro can kill 70% of the flock in 3 days if unvaccinated."
            ),
            "source": "Father's protocol + OIE IBD guidelines"
        })

    elif flock_age_weeks >= 10:
        # Past week 10 — Newcastle booster window
        newcastle_vaccinated = farm_profile.get("newcastle_vaccinated", "none")
        if newcastle_vaccinated in ["none", "partial"]:
            score += RISK_WEIGHTS["newcastle_booster_due"]
            risk_factors.append({
                "factor": (
                    f"Flock is {flock_age_weeks} weeks old — "
                    f"Newcastle booster overdue (due at week 10)"
                ),
                "points": RISK_WEIGHTS["newcastle_booster_due"],
                "action": (
                    "Give Newcastle booster (Lasota) in drinking water this week. "
                    "Immunity from earlier doses wanes significantly by week 10."
                ),
                "source": "Ghana VSD vaccination schedule"
            })

    elif flock_age_weeks > 0 and flock_age_weeks < 3:
        # Chick stage — coccidiosis risk period, lower severity
        risk_factors.append({
            "factor": f"Chick stage ({flock_age_weeks} weeks) — intestinal vulnerability",
            "points": 5,
            "action": (
                "Confirm coccidiosis prevention medicine is being given "
                "3 days per week. Sharp grinded maize in starter feed "
                "can damage intestinal lining."
            ),
            "source": "Father's protocol"
        })
        score += 5

    # ── VACCINATION STATUS ────────────────────────────────────────────────
    vaccinated = farm_profile.get("vaccinated", True)
    days_since_vacc = farm_profile.get("days_since_vaccination", 30)

    if not vaccinated:
        score += RISK_WEIGHTS["not_vaccinated"]
        risk_factors.append({
            "factor": "Flock not vaccinated against Newcastle Disease",
            "points": RISK_WEIGHTS["not_vaccinated"],
            "action": (
                "Vaccinate immediately with Lasota (Newcastle). "
                "Contact your VSD office for vaccine availability. "
                "Unvaccinated flocks face up to 100% mortality in an outbreak."
            ),
            "source": "Ouma et al. (2023)"
        })
    elif days_since_vacc > 90:
        score += RISK_WEIGHTS["vaccination_overdue"]
        risk_factors.append({
            "factor": f"Vaccination overdue — {days_since_vacc} days since last dose",
            "points": RISK_WEIGHTS["vaccination_overdue"],
            "action": (
                "Schedule a booster vaccination this week. "
                "Newcastle immunity wanes significantly after 90 days."
            ),
            "source": "Ghana VSD vaccination schedule"
        })

    # ── SEASONAL RISK ─────────────────────────────────────────────────────
    min_pct = 22.97
    max_pct = 34.27
    seasonal_contribution = int(
        (monthly["pct"] - min_pct) / (max_pct - min_pct)
        * RISK_WEIGHTS["peak_season"]
    )
    score += seasonal_contribution

    if monthly["level"] in ["High", "Highest"]:
        risk_factors.append({
            "factor": (
                f"{monthly['label']} is a {monthly['level'].lower()} risk month "
                f"— {monthly['pct']}% of annual cases"
            ),
            "points": seasonal_contribution,
            "action": (
                "Increase monitoring to twice daily during "
                f"{monthly['label']}. Check for coughing, nasal discharge, "
                "reduced feed intake."
            ),
            "source": "Mensah et al. (2023) — KNUST VS Lab"
        })

    # ── RECENT DEATHS ─────────────────────────────────────────────────────
    if farm_profile.get("recent_deaths", False):
        score += RISK_WEIGHTS["recent_deaths"]
        death_count = farm_profile.get("death_count_this_week", 0)
        risk_factors.append({
            "factor": f"Recent bird deaths reported"
                      + (f" ({death_count} this week)" if death_count else ""),
            "points": RISK_WEIGHTS["recent_deaths"],
            "action": (
                "Isolate dead birds immediately — do not move them off the farm. "
                "Send a voice note to KokoAlert. Contact VSD for diagnostic support."
            ),
            "source": "Ghana VSD biosecurity guidelines"
        })

    # ── NEW BIRDS INTRODUCED ──────────────────────────────────────────────
    if farm_profile.get("new_birds_introduced", False):
        score += RISK_WEIGHTS["new_birds_introduced"]
        risk_factors.append({
            "factor": "New birds introduced without quarantine",
            "points": RISK_WEIGHTS["new_birds_introduced"],
            "action": (
                "Keep new birds in a separate house for 14 days minimum "
                "before mixing with existing flock."
            ),
            "source": "Ayim-Akonor et al. (2020)"
        })

    # ── BIOSECURITY FACTORS ───────────────────────────────────────────────
    if not farm_profile.get("has_footbath", True):
        score += RISK_WEIGHTS["no_footbath"]
        risk_factors.append({
            "factor": "No footbath at poultry house entrance",
            "points": RISK_WEIGHTS["no_footbath"],
            "action": (
                "Install a footbath with 2% Virkon S or formalin solution. "
                "71.1% of Ashanti farms lack this basic biosecurity measure."
            ),
            "source": "Ayim-Akonor et al. (2020)"
        })

    ventilation = farm_profile.get("ventilation", "good")
    if ventilation == "poor":
        score += RISK_WEIGHTS["poor_ventilation"]
        risk_factors.append({
            "factor": "Poor ventilation in poultry house",
            "points": RISK_WEIGHTS["poor_ventilation"],
            "action": (
                "Improve airflow by opening additional vents or windows. "
                "Poor ventilation concentrates airborne pathogens and "
                "increases CRD risk — which opens the door to Newcastle."
            ),
            "source": "Ghana VSD biosecurity guidelines"
        })
    elif ventilation == "medium":
        score += int(RISK_WEIGHTS["poor_ventilation"] / 2)

    # ── FLOCK SIZE ────────────────────────────────────────────────────────
    flock_size = farm_profile.get("flock_size", "100_to_500")
    if flock_size == "over_2000":
        score += RISK_WEIGHTS["large_flock"]
        risk_factors.append({
            "factor": "Large flock (over 2,000 birds) — faster disease spread",
            "points": RISK_WEIGHTS["large_flock"],
            "action": (
                "For large flocks, divide into separate houses where possible "
                "to limit transmission if disease enters."
            ),
            "source": "FAO poultry biosecurity guidelines"
        })

    # ── WEEKLY CHECK FACTORS ──────────────────────────────────────────────
    if farm_profile.get("weekly_appetite") in ["some_reduced", "most_reduced"]:
        score = min(score + 5, 100)
        risk_factors.append({
            "factor": "Reduced appetite reported in weekly check",
            "points": 5,
            "action": (
                "Reduced appetite combined with any respiratory sounds "
                "is a serious warning sign. Send a voice note now."
            ),
            "source": "Clinical observation"
        })

    score = min(score, 100)

    # ── CATEGORY ──────────────────────────────────────────────────────────
    category = "Low"
    emoji = "🟢"
    for min_score, max_score, cat, emo in RISK_CATEGORIES:
        if min_score <= score <= max_score:
            category = cat
            emoji = emo
            break

    risk_factors.sort(key=lambda x: x["points"], reverse=True)
    top_action = (
        risk_factors[0]["action"] if risk_factors
        else "Continue regular monitoring and maintain vaccination schedule."
    )

    region = farm_profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    return {
        "score": score,
        "category": category,
        "emoji": emoji,
        "risk_factors": risk_factors,
        "top_action": top_action,
        "monthly_data": monthly,
        "vsd_contact": vsd,
        "month": month,
        "flock_age_weeks": flock_age_weeks,
    }


def format_risk_for_whatsapp(risk_result: dict) -> str:
    score = risk_result["score"]
    category = risk_result["category"]
    emoji = risk_result["emoji"]
    monthly = risk_result["monthly_data"]
    top_action = risk_result["top_action"]
    vsd = risk_result["vsd_contact"]
    flock_age = risk_result.get("flock_age_weeks", 0)

    age_note = ""
    if flock_age > 0:
        age_note = f"Flock age: *{flock_age} weeks*\n"

    msg = (
        f"{emoji} *Farm Risk Score: {score}/100 — {category}*\n\n"
        f"{age_note}"
        f"📅 *{monthly['label']}* risk level: *{monthly['level']}*\n"
        f"{monthly['pct']}% of annual Ashanti cases occur this month.\n"
        f"_(Mensah et al., 2023 — KNUST VS Lab)_\n\n"
    )

    if risk_result["risk_factors"]:
        msg += "*Your risk factors:*\n"
        for rf in risk_result["risk_factors"][:3]:
            msg += f"• {rf['factor']}\n"
        msg += "\n"

    msg += f"*Top action:* {top_action}\n\n"
    msg += f"📞 *{vsd['office']}*: {vsd['phone']}\n\n"
    msg += "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"

    return msg