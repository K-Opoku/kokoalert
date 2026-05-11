from src.config import VSD_CONTACTS


BIOSECURITY_DEDUCTIONS = {
    "no_footbath": {
        "points": 2,
        "factor": "No footbath at poultry house entrance",
        "recommendation": (
            "Install a footbath with 2% Virkon S or formalin solution "
            "at your poultry house entrance. Change solution every 3 days. "
            "71.1% of Ashanti farms lack this — it is the easiest disease "
            "entry point to close."
        ),
        "source": "Ayim-Akonor et al. (2020)"
    },
    "poor_ventilation": {
        "points": 2,
        "factor": "Poor ventilation",
        "recommendation": (
            "Open additional vents or windows to improve airflow. "
            "Poor ventilation concentrates airborne pathogens and "
            "raises temperature stress, weakening bird immunity."
        ),
        "source": "Ghana VSD biosecurity guidelines"
    },
    "medium_ventilation": {
        "points": 1,
        "factor": "Ventilation could be improved",
        "recommendation": (
            "Ensure all existing vents are fully open during the day. "
            "Consider adding roof ventilation if birds are above "
            "optimal stocking density."
        ),
        "source": "Ghana VSD biosecurity guidelines"
    },
    "no_quarantine": {
        "points": 3,
        "factor": "New birds introduced without quarantine period",
        "recommendation": (
            "Always keep new birds in a completely separate house "
            "for a minimum of 14 days before mixing with your existing flock. "
            "New birds are the single most common route of disease introduction."
        ),
        "source": "Ayim-Akonor et al. (2020)"
    },
    "large_flock_single_house": {
        "points": 1,
        "factor": "Large flock in single house increases transmission risk",
        "recommendation": (
            "For flocks over 2,000 birds, divide into separate houses "
            "where possible. If one house becomes infected, the others "
            "remain protected."
        ),
        "source": "FAO poultry biosecurity guidelines"
    },
    "on_farm_waste": {
        "points": 2,
        "factor": "Waste disposal risk",
        "recommendation": (
            "Remove litter and dead birds from the farm premises promptly. "
            "55.3% of Ashanti farms dispose of waste on-farm, creating "
            "persistent disease reservoirs."
        ),
        "source": "Ayim-Akonor et al. (2020)"
    },
}

BIOSECURITY_GRADES = [
    (8, 10, "Good",  "🟢"),
    (5,  7, "Fair",  "🟡"),
    (0,  4, "Poor",  "🔴"),
]


def compute_biosecurity_score(farm_profile: dict) -> dict:
    deductions = []
    total_deducted = 0

    if not farm_profile.get("has_footbath", True):
        d = BIOSECURITY_DEDUCTIONS["no_footbath"].copy()
        deductions.append(d)
        total_deducted += d["points"]

    ventilation = farm_profile.get("ventilation", "good")
    if ventilation == "poor":
        d = BIOSECURITY_DEDUCTIONS["poor_ventilation"].copy()
        deductions.append(d)
        total_deducted += d["points"]
    elif ventilation == "medium":
        d = BIOSECURITY_DEDUCTIONS["medium_ventilation"].copy()
        deductions.append(d)
        total_deducted += d["points"]

    if farm_profile.get("new_birds_introduced", False):
        d = BIOSECURITY_DEDUCTIONS["no_quarantine"].copy()
        deductions.append(d)
        total_deducted += d["points"]

    flock_size = farm_profile.get("flock_size", "100_to_500")
    if flock_size == "over_2000":
        d = BIOSECURITY_DEDUCTIONS["large_flock_single_house"].copy()
        deductions.append(d)
        total_deducted += d["points"]

    score = max(0, 10 - total_deducted)

    grade = "Poor"
    emoji = "🔴"
    for min_s, max_s, g, e in BIOSECURITY_GRADES:
        if min_s <= score <= max_s:
            grade = g
            emoji = e
            break

    deductions.sort(key=lambda x: x["points"], reverse=True)

    improvements = [
        {
            "factor": d["factor"],
            "points_lost": d["points"],
            "recommendation": d["recommendation"],
            "source": d["source"],
        }
        for d in deductions
    ]

    return {
        "score": score,
        "grade": grade,
        "emoji": emoji,
        "improvements": improvements,
        "total_deducted": total_deducted,
    }


def format_biosecurity_for_whatsapp(result: dict) -> str:
    score = result["score"]
    grade = result["grade"]
    emoji = result["emoji"]

    msg = f"{emoji} *Biosecurity Score: {score}/10 — {grade}*\n\n"

    if not result["improvements"]:
        msg += (
            "✅ No major biosecurity gaps identified. "
            "Keep maintaining your current practices.\n\n"
        )
    else:
        msg += "*Improvements needed:*\n"
        for i, imp in enumerate(result["improvements"][:3], 1):
            msg += f"{i}. {imp['recommendation']}\n\n"

    msg += "_Reply RISK to see your disease risk score._"
    return msg