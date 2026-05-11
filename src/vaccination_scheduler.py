"""
src/vaccination_scheduler.py
─────────────────────────────────────────────────────────────────────────────
Tracks flock age from DOC arrival date and manages the full vaccination
schedule. This is one of KokoAlert's highest-value features — most Ghanaian
farmers miss vaccines simply because no one reminded them at the right time.

How it works:
  1. When a farmer registers a new flock (DOC arrival), KokoAlert records
     the arrival date in the farm profile.
  2. Every morning, the Farm Monitor Agent calls get_todays_reminders().
  3. If any vaccine or medicine is due ±2 days, a WhatsApp reminder is sent.
  4. When the farmer confirms they vaccinated, the log is updated.

The schedule is based on:
  - Ghana VSD/MoFA official vaccination protocol
  - Your father's protocol (glucose day 1, cocci weeks 1–15, stop at 15)
  - Standard West African broiler/layer management practice
"""

from datetime import date, datetime, timedelta
from src.config import VACCINATION_SCHEDULE, COCCI_MEDICINE_WEEKS, COCCI_DRUG


# ── FLOCK AGE CALCULATION ─────────────────────────────────────────────────────

def get_flock_age_days(doc_arrival_date: str) -> int:
    """
    Calculate how many days old the flock is.

    Args:
        doc_arrival_date: ISO format string "YYYY-MM-DD"

    Returns:
        Age in days (0 if arrived today)
    """
    try:
        arrival = date.fromisoformat(doc_arrival_date)
        today = date.today()
        return max(0, (today - arrival).days)
    except (ValueError, TypeError):
        return 0


def get_flock_age_weeks(doc_arrival_date: str) -> int:
    """Return flock age in complete weeks."""
    return get_flock_age_days(doc_arrival_date) // 7


# ── VACCINATION STATUS CHECK ─────────────────────────────────────────────────

def get_vaccination_status(
    farm_profile: dict,
    vaccination_log: dict
) -> list:
    """
    Return the full vaccination status for a flock.
    Shows what has been done, what is due, and what is overdue.

    Args:
        farm_profile: Must contain doc_arrival_date
        vaccination_log: Dict of {vaccine_id: date_given_str}

    Returns:
        List of status dicts, one per scheduled item
    """
    doc_arrival_date = farm_profile.get("doc_arrival_date")
    if not doc_arrival_date:
        return []

    flock_age_days = get_flock_age_days(doc_arrival_date)
    statuses = []

    for item in VACCINATION_SCHEDULE:
        trigger_day = item["trigger_day"]
        given_date = vaccination_log.get(item["id"])

        if given_date:
            status = "done"
            urgency = "done"
        elif flock_age_days > trigger_day + 5:
            status = "overdue"
            urgency = "urgent"
        elif flock_age_days >= trigger_day - 2:
            status = "due_now"
            urgency = "critical" if item["urgency"] == "critical" else "important"
        else:
            days_until = trigger_day - flock_age_days
            status = "upcoming"
            urgency = "upcoming"

        statuses.append({
            "id": item["id"],
            "label": item["label"],
            "trigger_day": trigger_day,
            "trigger_week": trigger_day // 7,
            "status": status,
            "urgency": urgency,
            "given_date": given_date,
            "drug": item.get("drug"),
            "is_vaccine": item["is_vaccine"],
            "instruction": item["instruction"],
            "days_until": max(0, trigger_day - flock_age_days) if status == "upcoming" else 0,
        })

    return statuses


# ── TODAY'S REMINDERS ─────────────────────────────────────────────────────────

def get_todays_reminders(
    farm_profile: dict,
    vaccination_log: dict
) -> list:
    """
    Return any vaccinations or medicine reminders due today (±2 day window).
    Called every morning by the Farm Monitor Agent.

    Returns empty list if nothing is due today.
    """
    doc_arrival_date = farm_profile.get("doc_arrival_date")
    if not doc_arrival_date:
        return []

    flock_age_days = get_flock_age_days(doc_arrival_date)
    due_today = []

    for item in VACCINATION_SCHEDULE:
        given = vaccination_log.get(item["id"])
        if given:
            continue  # Already done

        trigger = item["trigger_day"]

        # Send reminder if within ±2 days of trigger day
        # This catches it before the exact day (preparation) and after (missed)
        if trigger - 2 <= flock_age_days <= trigger + 2:
            days_diff = flock_age_days - trigger
            if days_diff < 0:
                timing = f"due in {abs(days_diff)} day(s)"
            elif days_diff == 0:
                timing = "due TODAY"
            else:
                timing = f"*{days_diff} day(s) overdue*"

            due_today.append({
                **item,
                "timing": timing,
                "is_overdue": days_diff > 0,
                "flock_age_days": flock_age_days,
            })

    # Check coccidiosis weekly medicine reminder
    flock_age_weeks = flock_age_days // 7
    cocci_reminder = get_cocci_reminder(flock_age_weeks, farm_profile)
    if cocci_reminder:
        due_today.append(cocci_reminder)

    return due_today


def get_cocci_reminder(flock_age_weeks: int, farm_profile: dict) -> dict | None:
    """
    Return a coccidiosis medicine reminder if due this week.
    Medicine should be given 3 days per week, weeks 1–15.
    STOP after week 15.
    """
    start = COCCI_MEDICINE_WEEKS["start"]
    stop = COCCI_MEDICINE_WEEKS["stop"]

    if flock_age_weeks == stop:
        # Stop warning — this is important
        return {
            "id": "cocci_stop",
            "label": "Stop Coccidiosis medicine",
            "instruction": (
                "⚠️ *STOP Coccidiosis medicine this week (Week 15)*\n\n"
                "Do NOT give anti-coccidial medicine after this week.\n"
                "Continuing beyond week 15 interferes with egg production in layers.\n\n"
                "If your birds are broilers going to market soon, observe the "
                "drug withdrawal period before slaughter (usually 5–7 days)."
            ),
            "urgency": "important",
            "is_vaccine": False,
            "trigger_day": stop * 7,
            "timing": "this week",
            "is_overdue": False,
        }

    if start <= flock_age_weeks < stop:
        # Monday reminder (check if it's start of week)
        today_weekday = datetime.now().weekday()
        if today_weekday == 0:  # Monday
            return {
                "id": "cocci_weekly",
                "label": f"Coccidiosis medicine (Week {flock_age_weeks})",
                "instruction": (
                    f"💊 *Coccidiosis medicine reminder — Week {flock_age_weeks}*\n\n"
                    f"Give *{COCCI_DRUG}* in drinking water for 3 days this week.\n"
                    f"This protects your birds' intestines from the sharp grinded maize "
                    f"in their feed.\n\n"
                    f"Continue every week until Week 15, then stop."
                ),
                "urgency": "important",
                "is_vaccine": False,
                "trigger_day": flock_age_weeks * 7,
                "timing": "this week",
                "is_overdue": False,
            }

    return None


# ── WHATSAPP MESSAGE FORMATTING ───────────────────────────────────────────────

def format_vaccination_reminder_message(reminders: list) -> str:
    """
    Format today's reminders into a WhatsApp message.
    Called by the Farm Monitor Agent every morning.
    """
    if not reminders:
        return ""

    lines = ["📋 *KokoAlert — Today's Farm Reminders*\n"]

    for r in reminders:
        urgency_icon = "🔴" if r.get("is_overdue") else "⏰"
        lines.append(f"{urgency_icon} *{r['label']}* ({r.get('timing', '')})")
        lines.append(r["instruction"])
        lines.append("")

    lines.append("Reply *VACC* anytime to see your full vaccination schedule.")
    lines.append("_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭")

    return "\n".join(lines)


def format_full_vaccination_schedule(
    farm_profile: dict,
    vaccination_log: dict
) -> str:
    """
    Format the complete vaccination schedule for the VACC command.
    Shows done, overdue, due now, and upcoming vaccines.
    """
    statuses = get_vaccination_status(farm_profile, vaccination_log)

    if not statuses:
        return (
            "I don't have a flock start date for your farm yet.\n\n"
            "Tell me when your day-old chicks arrived:\n"
            "Reply: *DOC [date]* (e.g. DOC 2025-05-01)"
        )

    flock_age_weeks = get_flock_age_weeks(farm_profile.get("doc_arrival_date", ""))

    lines = [
        f"💉 *Vaccination Schedule*\n"
        f"Flock age: *{flock_age_weeks} weeks*\n"
    ]

    # Group by status
    overdue = [s for s in statuses if s["status"] == "overdue"]
    due_now = [s for s in statuses if s["status"] == "due_now"]
    upcoming = [s for s in statuses if s["status"] == "upcoming"]
    done = [s for s in statuses if s["status"] == "done"]

    if overdue:
        lines.append("🔴 *OVERDUE — Act now:*")
        for s in overdue:
            lines.append(f"  • {s['label']} (was due Week {s['trigger_week']})")
        lines.append("")

    if due_now:
        lines.append("⏰ *Due this week:*")
        for s in due_now:
            lines.append(f"  • {s['label']}")
            if s.get("drug"):
                lines.append(f"    Drug: {s['drug']}")
        lines.append("")

    if upcoming:
        lines.append("📅 *Upcoming:*")
        for s in upcoming[:4]:  # Show next 4 only to keep message short
            lines.append(f"  • Week {s['trigger_week']}: {s['label']}")
        lines.append("")

    if done:
        lines.append(f"✅ *Completed:* {len(done)} items done")

    lines.append("\n_Reply RISK to see your disease risk score._")
    lines.append("_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭")

    return "\n".join(lines)


# ── RECORD A VACCINATION ──────────────────────────────────────────────────────

def record_vaccination(
    vaccine_id: str,
    vaccination_log: dict,
    given_date: str = None
) -> dict:
    """
    Record that a vaccination was given.
    Updates vaccination_log in place and returns the updated log.

    Args:
        vaccine_id: Must match an id in VACCINATION_SCHEDULE
        vaccination_log: Existing log dict from database
        given_date: ISO date string. Defaults to today.

    Returns:
        Updated vaccination_log
    """
    if given_date is None:
        given_date = date.today().isoformat()

    vaccination_log[vaccine_id] = given_date
    return vaccination_log


# ── NEW FLOCK REGISTRATION ────────────────────────────────────────────────────

def register_new_flock(
    farm_profile: dict,
    doc_arrival_date: str = None
) -> tuple[dict, str]:
    """
    Register a new flock of day-old chicks.
    Updates farm_profile with arrival date and resets vaccination log.

    Args:
        farm_profile: Existing farm profile
        doc_arrival_date: ISO date string. Defaults to today.

    Returns:
        (updated_farm_profile, confirmation_message)
    """
    if doc_arrival_date is None:
        doc_arrival_date = date.today().isoformat()

    farm_profile["doc_arrival_date"] = doc_arrival_date
    farm_profile["flock_age_weeks"] = 0
    farm_profile["vaccination_log"] = {}

    # Reset disease flags from previous flock
    farm_profile.pop("recent_deaths", None)
    farm_profile.pop("new_birds_introduced", None)
    farm_profile.pop("gumboro_vaccinated", None)
    farm_profile.pop("newcastle_vaccinated", None)

    msg = (
        f"✅ *New flock registered!*\n\n"
        f"Arrival date: *{doc_arrival_date}*\n\n"
        f"*Your first reminder:*\n"
        f"Give your chicks *glucose + antibiotic vitamins* in their water TODAY.\n"
        f"Do not give plain water on day 1.\n\n"
        f"KokoAlert will send you a reminder for each vaccine and medicine "
        f"at the right time. You don't need to remember the schedule.\n\n"
        f"*Next reminder:* Day 7 — 1st Gumboro vaccine\n\n"
        f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
    )

    return farm_profile, msg
