"""
whatsapp_bot/handlers.py
─────────────────────────────────────────────────────────────────────────────
WhatsApp message routing and response logic.
"""

import httpx
import tempfile
import os
import asyncio
from datetime import datetime, date

from src.pipeline import pipeline
from src.diagnosis_engine import run_diagnosis
from src.vaccination_scheduler import (
    get_todays_reminders, format_vaccination_reminder_message,
    format_full_vaccination_schedule, record_vaccination,
    register_new_flock, get_flock_age_weeks
)
from src.risk_engine import compute_farm_risk_score
from src.biosecurity_scorer import compute_biosecurity_score
from api.database import (
    get_farm_profile, save_farm_profile,
    save_analysis_result, log_agent_action,
    get_onboarding_state, set_onboarding_state,
    clear_onboarding_state, get_vaccination_log,
    save_vaccination_log,
)
from src.config import (
    WHATSAPP_API_URL, WHATSAPP_API_TOKEN,
    VSD_CONTACTS, ONBOARDING_QUESTIONS, MONTHLY_RISK_DATA
)


# ── SEND MESSAGE ──────────────────────────────────────────────────────────────

async def send_whatsapp_message(phone: str, message: str):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": message},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(WHATSAPP_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"[WHATSAPP ERROR] {response.status_code} → {response.text}")


# ── DROPPINGS QUESTION ────────────────────────────────────────────────────────

DROPPINGS_QUESTION = (
    "🔍 *KokoAlert detected something in your flock audio.*\n\n"
    "To give you a confirmed diagnosis, I need to know:\n\n"
    "*What do the droppings look like right now?*\n\n"
    "1 — Normal (brown or dark)\n"
    "2 — Bloody or dark chocolate coloured\n"
    "3 — Bright green or yellow-green\n"
    "4 — White and very watery\n"
    "5 — Haven't checked yet\n\n"
    "Go check and reply with a number."
)

DROPPINGS_MAP = {
    "1": "normal",
    "2": "bloody_chocolate",
    "3": "bright_green",
    "4": "white_watery",
    "5": "normal",
}

BEHAVIOR_QUESTION = (
    "*One more question — are the birds showing any of these signs?*\n\n"
    "Reply with all numbers that apply (e.g. *1 3*):\n\n"
    "1 — Coughing or sneezing\n"
    "2 — Very weak or quiet\n"
    "3 — Huddled together\n"
    "4 — Reduced eating or drinking\n"
    "5 — Sores or scabs on face or comb\n"
    "6 — None of the above"
)

BEHAVIOR_MAP = {
    "1": "coughing",
    "2": "weak",
    "3": "huddled",
    "4": "reduced_appetite",
    "5": "face_lesions",
}


# ── MAIN ROUTER ───────────────────────────────────────────────────────────────

async def handle_incoming_message(webhook_data: dict):
    """Route incoming WhatsApp messages to the correct handler."""
    try:
        entry = webhook_data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "no_message"}

        message = messages[0]
        phone = message.get("from")
        msg_type = message.get("type")
        state = get_onboarding_state(phone)

        if msg_type == "text":
            text = message.get("text", {}).get("body", "").strip()
            text_upper = text.upper()

            if state and state.startswith("onboarding_"):
                idx = int(state.replace("onboarding_", ""))
                await handle_onboarding_reply(phone, text, idx)

            elif state == "awaiting_droppings":
                await handle_droppings_reply(phone, text)

            elif state == "awaiting_behavior":
                await handle_behavior_reply(phone, text)

            elif state and state.startswith("weekly_"):
                idx = int(state.replace("weekly_", ""))
                await handle_weekly_check_reply(phone, text, idx)

            elif state == "awaiting_death_count":
                await handle_death_count_reply(phone, text)

            elif state == "awaiting_doc_date":
                await handle_doc_date_reply(phone, text)

            elif text_upper in ["HI", "HELLO", "START", "KOKOALERT"]:
                profile = get_farm_profile(phone)
                if not profile:
                    await start_onboarding(phone)
                else:
                    await send_main_menu(phone, profile)

            else:
                await handle_command(phone, text_upper, text)

        elif msg_type == "audio":
            audio_id = message.get("audio", {}).get("id")
            await handle_audio_message(phone, audio_id)

        return {"status": "processed"}

    except Exception as e:
        print(f"[ERROR] handle_incoming_message: {e}")
        return {"status": "error", "message": str(e)}


# ── ONBOARDING ────────────────────────────────────────────────────────────────

async def start_onboarding(phone: str):
    set_onboarding_state(phone, "onboarding_0")
    await send_whatsapp_message(phone, ONBOARDING_QUESTIONS[0]["question"])


async def handle_onboarding_reply(phone: str, reply: str, question_index: int):
    question = ONBOARDING_QUESTIONS[question_index]
    options = question["options"]

    if reply not in options:
        await send_whatsapp_message(phone,
            f"Please reply with one of the numbers shown.\n\n{question['question']}"
        )
        return

    profile = get_farm_profile(phone) or {}
    answer = options[reply]

    if question["key"] == "doc_arrival_date":
        arrival_days_ago = answer
        from datetime import timedelta
        arrival_date = (date.today() - timedelta(days=arrival_days_ago)).isoformat()
        profile["doc_arrival_date"] = arrival_date
        profile["flock_age_weeks"] = arrival_days_ago // 7
    else:
        profile[question["key"]] = answer

    save_farm_profile(phone, profile)

    next_index = question_index + 1

    if next_index < len(ONBOARDING_QUESTIONS):
        set_onboarding_state(phone, f"onboarding_{next_index}")
        await send_whatsapp_message(phone, ONBOARDING_QUESTIONS[next_index]["question"])
    else:
        clear_onboarding_state(phone)
        await send_onboarding_complete(phone, profile)


async def send_onboarding_complete(phone: str, profile: dict):
    flock_age_weeks = profile.get("flock_age_weeks", 0)
    vacc_log = get_vaccination_log(phone) or {}
    reminders = get_todays_reminders(profile, vacc_log)

    msg = (
        "✅ *Farm profile saved!*\n\n"
        "Here is how to use KokoAlert:\n\n"
        "🎙️ *Send a voice note* from inside your poultry house → "
        "I will tell you what disease your flock may have and what to do.\n\n"
        "📋 *Every morning* you will receive a daily check-in and any "
        "vaccination reminders.\n\n"
        "*Commands anytime:*\n"
        "RISK · VACC · BIOSEC · HELP · DOC\n\n"
    )

    if reminders:
        reminder_msg = format_vaccination_reminder_message(reminders)
        msg += f"*Your first reminder:*\n{reminder_msg}"
    elif flock_age_weeks <= 6:
        msg += (
            f"⚠️ Your birds are *{flock_age_weeks} weeks old*.\n"
            f"Check your vaccination schedule — reply *VACC* to see what is due."
        )

    msg += "\n\n_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
    await send_whatsapp_message(phone, msg)
    log_agent_action("onboarding", phone, "onboarding_complete")


# ── AUDIO MESSAGE HANDLER ─────────────────────────────────────────────────────

async def handle_audio_message(phone: str, audio_id: str):
    """
    Download voice note → run CNN classifier in background thread
    so the server stays responsive during analysis.
    """
    await send_whatsapp_message(phone,
        "🧠 *Analysing your flock...* About 10 seconds."
    )

    try:
        # Download audio from WhatsApp
        headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}"}
        async with httpx.AsyncClient() as client:
            meta_resp = await client.get(
                f"https://graph.facebook.com/v18.0/{audio_id}",
                headers=headers
            )
            audio_url = meta_resp.json().get("url")
            audio_resp = await client.get(audio_url, headers=headers)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_resp.content)
            tmp_path = tmp.name

        farm_profile = get_farm_profile(phone) or {}

        if not pipeline._loaded:
            pipeline.load_models()

        # ── FIX: Run blocking pipeline in thread executor ──────────────────
        # This keeps the server responsive while the CNN processes audio.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.analyse_audio(tmp_path, farm_profile, {})
        )

        os.unlink(tmp_path)
        save_analysis_result(phone, result)

        status = result.get("status")

        if status == "inconclusive":
            await send_whatsapp_message(phone, result["whatsapp_message"])

        elif status == "healthy":
            await send_whatsapp_message(phone, result["whatsapp_message"])

        elif status in ["diagnosed", "needs_symptoms"]:
            farm_profile["_pending_audio_result"] = result.get("audio", {})
            save_farm_profile(phone, farm_profile)

            if status == "needs_symptoms":
                set_onboarding_state(phone, "awaiting_droppings")
                await send_whatsapp_message(phone, DROPPINGS_QUESTION)
            else:
                await send_whatsapp_message(phone, result["whatsapp_message"])

    except Exception as e:
        print(f"[ERROR] handle_audio_message: {e}")
        await send_whatsapp_message(phone,
            "❌ Something went wrong analysing your recording.\n\n"
            "Please try again, or type *HELP* for emergency vet contacts."
        )


# ── DROPPINGS + BEHAVIOR FOLLOW-UP ───────────────────────────────────────────

async def handle_droppings_reply(phone: str, reply: str):
    if reply not in DROPPINGS_MAP:
        await send_whatsapp_message(phone,
            f"Please reply with a number 1 to 5.\n\n{DROPPINGS_QUESTION}"
        )
        return

    profile = get_farm_profile(phone) or {}
    profile["_pending_droppings"] = DROPPINGS_MAP[reply]
    save_farm_profile(phone, profile)

    set_onboarding_state(phone, "awaiting_behavior")
    await send_whatsapp_message(phone, BEHAVIOR_QUESTION)


async def handle_behavior_reply(phone: str, reply: str):
    """
    Farmer replied with behaviour signs.
    Run full diagnosis and send result.
    """
    profile = get_farm_profile(phone) or {}
    clear_onboarding_state(phone)

    # Parse multi-select behaviour reply (e.g. "1 3 5")
    behavior = []
    for num in reply.split():
        if num in BEHAVIOR_MAP:
            behavior.append(BEHAVIOR_MAP[num])

    droppings = profile.pop("_pending_droppings", "normal")
    audio_result = profile.pop("_pending_audio_result", {
        "is_anomalous": True,
        "probability": 0.6,
    })
    save_farm_profile(phone, profile)

    symptoms = {
        "droppings": droppings,
        "behavior": behavior,
        "cocci_medicine_given": profile.get("cocci_medicine_given", False),
    }

    # Run diagnosis
    diagnosis = run_diagnosis(
        farm_profile=profile,
        audio_result=audio_result,
        symptoms=symptoms,
    )

    save_analysis_result(phone, diagnosis)
    await send_whatsapp_message(phone, diagnosis["whatsapp_message"])

    log_agent_action(
        "diagnosis",
        phone,
        f"diagnosed_{diagnosis.get('disease', 'unknown')}_{diagnosis.get('confidence', '')}"
    )


# ── DAILY CHECK-IN ────────────────────────────────────────────────────────────

async def send_daily_checkin(phone: str, farmer_name: str, profile: dict):
    flock_age_weeks = get_flock_age_weeks(profile.get("doc_arrival_date", ""))
    vacc_log = get_vaccination_log(phone) or {}
    reminders = get_todays_reminders(profile, vacc_log)

    msg = (
        f"🐔 *Good morning {farmer_name}!*\n\n"
        f"How is your flock today?\n\n"
        f"1 — All good\n"
        f"2 — Some birds look sick or slow\n"
        f"3 — Birds have died\n"
        f"4 — I vaccinated today\n"
        f"5 — I brought in new birds\n"
        f"6 — New batch of day-old chicks arrived today\n\n"
        f"Or send a *voice note* for a full audio check."
    )

    if reminders:
        reminder_text = format_vaccination_reminder_message(reminders)
        msg += f"\n\n{reminder_text}"

    await send_whatsapp_message(phone, msg)


async def handle_daily_checkin_reply(phone: str, reply: str):
    profile = get_farm_profile(phone) or {}
    region = profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    if reply == "1":
        await send_whatsapp_message(phone,
            "✅ Good to hear. Keep up your daily checks.\n\n"
            "Send a voice note anytime for a full audio analysis."
        )

    elif reply == "2":
        await send_whatsapp_message(phone,
            "⚠️ *Birds looking sick or slow.*\n\n"
            "Send a *voice note* from inside the house now so I can analyse the flock.\n\n"
            "While recording, also look at the droppings — what colour are they?\n"
            "I will ask you this after the audio check."
        )

    elif reply == "3":
        set_onboarding_state(phone, "awaiting_death_count")
        await send_whatsapp_message(phone,
            "❗ *Bird deaths reported.*\n\nHow many birds died? Reply with a number (e.g. *5*)."
        )

    elif reply == "4":
        await send_whatsapp_message(phone,
            "✅ *Which vaccine did you give today?*\n\n"
            "1 — 1st Gumboro\n"
            "2 — 2nd Gumboro\n"
            "3 — 1st Newcastle (Lasota)\n"
            "4 — 2nd Newcastle (Lasota)\n"
            "5 — Newcastle booster\n"
            "6 — Newcastle injection (oil)\n"
            "7 — Fowl Pox\n"
            "8 — Other"
        )
        set_onboarding_state(phone, "awaiting_vaccine_confirm")

    elif reply == "5":
        profile["new_birds_introduced"] = True
        save_farm_profile(phone, profile)
        await send_whatsapp_message(phone,
            "⚠️ *New birds recorded.*\n\n"
            "Keep new birds in a *separate house for 14 days* before mixing with your flock.\n"
            "New birds are the most common way disease enters a farm.\n\n"
            "Your risk score has been updated."
        )
        log_agent_action("farm_monitor", phone, "new_birds_recorded")

    elif reply == "6":
        set_onboarding_state(phone, "awaiting_doc_date")
        await send_whatsapp_message(phone,
            "🐥 *New day-old chicks!*\n\n"
            "Did they arrive today?\n\n"
            "1 — Yes, arrived today\n"
            "2 — They arrived a few days ago"
        )

    else:
        await send_whatsapp_message(phone,
            "Please reply with a number 1 to 6, or send a voice note."
        )


async def handle_doc_date_reply(phone: str, reply: str):
    profile = get_farm_profile(phone) or {}
    clear_onboarding_state(phone)

    from datetime import timedelta
    if reply == "1":
        doc_date = date.today().isoformat()
    else:
        doc_date = (date.today() - timedelta(days=3)).isoformat()

    profile, msg = register_new_flock(profile, doc_date)
    save_farm_profile(phone, profile)
    save_vaccination_log(phone, {})
    await send_whatsapp_message(phone, msg)
    log_agent_action("farm_monitor", phone, "new_flock_registered")


async def handle_death_count_reply(phone: str, reply: str):
    profile = get_farm_profile(phone) or {}
    region = profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])
    clear_onboarding_state(phone)

    try:
        count = int(reply)
    except ValueError:
        await send_whatsapp_message(phone,
            "Please reply with a number — how many birds died? (e.g. *5*)"
        )
        set_onboarding_state(phone, "awaiting_death_count")
        return

    profile["recent_deaths"] = True
    profile["death_count_this_week"] = count
    save_farm_profile(phone, profile)

    flock_age_weeks = get_flock_age_weeks(profile.get("doc_arrival_date", ""))
    urgency_note = ""
    if 3 <= flock_age_weeks <= 6:
        urgency_note = (
            "\n\n🔴 *Your birds are in the Gumboro danger window (3–6 weeks).* "
            "Multiple deaths at this age is a serious warning sign. Send a voice note now."
        )

    await send_whatsapp_message(phone,
        f"❗ *{count} bird deaths recorded.*\n\n"
        f"Your risk score has been updated.\n\n"
        f"Send a *voice note* from inside the house so I can check for respiratory symptoms."
        f"{urgency_note}\n\n"
        f"If deaths are increasing rapidly:\n"
        f"📞 {vsd['office']}: {vsd['phone']}"
    )


# ── WEEKLY HEALTH CHECK ───────────────────────────────────────────────────────

WEEKLY_CHECK_QUESTIONS = [
    {
        "key": "weekly_pecking",
        "question": (
            "🗓️ *Weekly Flock Health Check — Question 1 of 5*\n\n"
            "Are birds pecking or attacking each other?\n\n"
            "1 — No\n2 — Occasionally\n3 — Serious problem"
        ),
        "options": {"1": "none", "2": "occasional", "3": "serious"}
    },
    {
        "key": "weekly_appetite",
        "question": (
            "*Question 2 of 5*\n\n"
            "Are birds eating and drinking normally?\n\n"
            "1 — Yes, normal\n2 — Some not eating\n3 — Most have reduced appetite"
        ),
        "options": {"1": "normal", "2": "some_reduced", "3": "most_reduced"}
    },
    {
        "key": "weekly_droppings",
        "question": (
            "*Question 3 of 5*\n\n"
            "What do the droppings look like this week?\n\n"
            "1 — Normal (brown/dark)\n"
            "2 — Bloody or dark chocolate\n"
            "3 — Bright green or yellow-green\n"
            "4 — White and watery\n"
            "5 — Mixed / not sure"
        ),
        "options": {
            "1": "normal", "2": "bloody_chocolate",
            "3": "bright_green", "4": "white_watery", "5": "mixed"
        }
    },
    {
        "key": "weekly_face_lesions",
        "question": (
            "*Question 4 of 5*\n\n"
            "Do any birds have scabs, warts, or sores on their face or comb?\n\n"
            "1 — No\n2 — Yes, a few birds\n3 — Yes, many birds"
        ),
        "options": {"1": "none", "2": "few", "3": "many"}
    },
    {
        "key": "weekly_deaths",
        "question": (
            "*Question 5 of 5*\n\n"
            "Any unexpected bird deaths this week?\n\n"
            "1 — No deaths\n2 — 1 to 3 deaths\n3 — More than 3 deaths"
        ),
        "options": {"1": "none", "2": "1_to_3", "3": "over_3"}
    },
]


async def send_weekly_health_check(phone: str, farmer_name: str):
    set_onboarding_state(phone, "weekly_0")
    await send_whatsapp_message(phone,
        f"👋 *Hi {farmer_name}!* Time for your weekly flock check.\n\n"
        f"5 quick questions — reply with a number.\n\n"
        + WEEKLY_CHECK_QUESTIONS[0]["question"]
    )


async def handle_weekly_check_reply(phone: str, reply: str, idx: int):
    question = WEEKLY_CHECK_QUESTIONS[idx]
    options = question["options"]

    if reply not in options:
        await send_whatsapp_message(phone,
            f"Please reply with a number.\n\n{question['question']}"
        )
        return

    profile = get_farm_profile(phone) or {}
    profile[question["key"]] = options[reply]

    if question["key"] == "weekly_droppings":
        profile["_last_droppings"] = options[reply]

    if question["key"] == "weekly_face_lesions" and options[reply] != "none":
        profile["_face_lesions_reported"] = True

    if question["key"] == "weekly_deaths" and options[reply] in ["1_to_3", "over_3"]:
        profile["recent_deaths"] = True

    save_farm_profile(phone, profile)
    next_idx = idx + 1

    if next_idx < len(WEEKLY_CHECK_QUESTIONS):
        set_onboarding_state(phone, f"weekly_{next_idx}")
        await send_whatsapp_message(phone, WEEKLY_CHECK_QUESTIONS[next_idx]["question"])
    else:
        clear_onboarding_state(phone)
        await send_weekly_feedback(phone, profile)


async def send_weekly_feedback(phone: str, profile: dict):
    droppings = profile.get("_last_droppings", "normal")
    face_lesions = profile.get("_face_lesions_reported", False)
    deaths = profile.get("weekly_deaths", "none")

    behavior = []
    if profile.get("weekly_appetite") in ["some_reduced", "most_reduced"]:
        behavior.append("reduced_appetite")
    if face_lesions:
        behavior.append("face_lesions")
    if deaths in ["1_to_3", "over_3"]:
        behavior.append("sudden_deaths")

    symptoms = {
        "droppings": droppings,
        "behavior": behavior,
        "cocci_medicine_given": profile.get("cocci_medicine_given", False),
    }

    audio_result = {"is_anomalous": False, "probability": 0.0}

    has_concerns = (
        droppings != "normal"
        or face_lesions
        or deaths != "none"
        or len(behavior) > 0
    )

    if has_concerns:
        diagnosis = run_diagnosis(
            farm_profile=profile,
            audio_result=audio_result,
            symptoms=symptoms,
        )
        if diagnosis.get("disease"):
            await send_whatsapp_message(phone,
                f"📋 *Weekly Check — Concern Found*\n\n"
                f"{diagnosis['whatsapp_message']}"
            )
            return

    flock_age_weeks = get_flock_age_weeks(profile.get("doc_arrival_date", ""))
    await send_whatsapp_message(phone,
        f"✅ *Weekly check complete — all good!*\n\n"
        f"No major issues reported this week. Keep up the daily checks.\n\n"
        f"Flock age: *{flock_age_weeks} weeks*\n\n"
        f"Reply *VACC* to check your vaccination schedule.\n"
        f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
    )


# ── COMMAND HANDLER ───────────────────────────────────────────────────────────

async def handle_command(phone: str, command: str, raw_text: str = ""):
    profile = get_farm_profile(phone) or {}
    state = get_onboarding_state(phone)
    region = profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    if state == "awaiting_death_count":
        await handle_death_count_reply(phone, command)
        return

    if state == "awaiting_vaccine_confirm":
        await handle_vaccine_confirmation(phone, command)
        return

    if state == "awaiting_doc_date":
        await handle_doc_date_reply(phone, command)
        return

    if command in ["1", "2", "3", "4", "5", "6"]:
        await handle_daily_checkin_reply(phone, command)
        return

    if command == "RESET":
        clear_onboarding_state(phone)
        profile.pop("_pending_audio_result", None)
        profile.pop("_pending_droppings", None)
        profile.pop("_pending_behavior", None)
        save_farm_profile(phone, profile)
        await send_whatsapp_message(phone,
            "✅ State reset. Send *HI* to continue."
        )

    elif command == "RISK":
        profile["current_month"] = datetime.now().month
        result = compute_farm_risk_score(profile)
        monthly = MONTHLY_RISK_DATA[datetime.now().month]

        await send_whatsapp_message(phone,
            f"{result['emoji']} *Farm Risk Score*\n\n"
            f"*Score:* {result['score']}/100 — *{result['category']}*\n\n"
            f"*{monthly['label']}* risk level: *{monthly['level']}*\n"
            f"{monthly['pct']}% of annual Ashanti cases occur this month.\n"
            f"_(KNUST VS Lab, 2018–2021)_\n\n"
            f"*Top action:* {result['top_action']}\n\n"
            f"Send a voice note to check your flock now."
        )

    elif command == "VACC":
        vacc_log = get_vaccination_log(phone) or {}
        msg = format_full_vaccination_schedule(profile, vacc_log)
        await send_whatsapp_message(phone, msg)

    elif command == "BIOSEC":
        result = compute_biosecurity_score(profile)
        msg = (
            f"🔒 *Biosecurity Score*\n\n"
            f"*Your score:* {result['score']}/10 — *{result['grade']}*\n\n"
        )
        if result.get("improvements"):
            top = result["improvements"][0]
            msg += f"*Top improvement:*\n{top['recommendation']}\n\n"
        msg += "_Reply RISK to see your disease risk score._"
        await send_whatsapp_message(phone, msg)

    elif command == "HELP":
        await send_whatsapp_message(phone,
            "🆘 *Emergency Contacts*\n\n"
            f"📍 *{vsd['office']}*\n"
            f"📞 {vsd['phone']}\n\n"
            "📍 *KNUST Vet Clinic*\n"
            "📞 0322-060137\n\n"
            "If birds are dying: *DO NOT move them.*\n"
            "Contain the flock and call now."
        )

    elif command.startswith("DOC"):
        parts = raw_text.split()
        if len(parts) == 2:
            doc_date = parts[1]
            try:
                date.fromisoformat(doc_date)
                profile, msg = register_new_flock(profile, doc_date)
                save_farm_profile(phone, profile)
                save_vaccination_log(phone, {})
                await send_whatsapp_message(phone, msg)
            except ValueError:
                await send_whatsapp_message(phone,
                    "Please use the format: *DOC YYYY-MM-DD*\n"
                    "Example: DOC 2025-05-01"
                )
        else:
            set_onboarding_state(phone, "awaiting_doc_date")
            await send_whatsapp_message(phone,
                "🐥 *Register new flock*\n\n"
                "Did your day-old chicks arrive today?\n\n"
                "1 — Yes, arrived today\n"
                "2 — They arrived a few days ago"
            )

    else:
        await send_whatsapp_message(phone,
            "Send a *voice note* from inside your poultry house to check your flock.\n\n"
            "Commands:\n"
            "*RISK* — farm risk score\n"
            "*VACC* — vaccination schedule\n"
            "*BIOSEC* — biosecurity score\n"
            "*HELP* — emergency vet contacts\n"
            "*DOC* — register new day-old chicks\n"
            "*RESET* — clear stuck state"
        )


async def handle_vaccine_confirmation(phone: str, reply: str):
    vacc_id_map = {
        "1": "gumboro_1", "2": "gumboro_2",
        "3": "newcastle_1", "4": "newcastle_2",
        "5": "newcastle_booster", "6": "newcastle_final",
        "7": "fowl_pox",
    }
    clear_onboarding_state(phone)

    if reply not in vacc_id_map:
        await send_whatsapp_message(phone,
            "✅ *Vaccination recorded for today.*\n\n"
            "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        return

    vacc_id = vacc_id_map[reply]
    vacc_log = get_vaccination_log(phone) or {}
    updated_log = record_vaccination(vacc_id, vacc_log)
    save_vaccination_log(phone, updated_log)

    profile = get_farm_profile(phone) or {}
    if "gumboro" in vacc_id:
        profile["gumboro_vaccinated"] = "both" if vacc_id == "gumboro_2" else "first_only"
    elif "newcastle" in vacc_id:
        profile["newcastle_vaccinated"] = "full"
    save_farm_profile(phone, profile)

    await send_whatsapp_message(phone,
        f"✅ *Vaccination recorded: {vacc_id.replace('_', ' ').title()}*\n\n"
        f"Your farm profile has been updated.\n\n"
        f"Reply *VACC* to see your full schedule."
    )
    log_agent_action("farm_monitor", phone, f"vaccination_recorded_{vacc_id}")


# ── MAIN MENU FOR RETURNING FARMER ───────────────────────────────────────────

async def send_main_menu(phone: str, profile: dict):
    profile["current_month"] = datetime.now().month
    risk = compute_farm_risk_score(profile)
    flock_age_weeks = get_flock_age_weeks(profile.get("doc_arrival_date", ""))
    vacc_log = get_vaccination_log(phone) or {}
    reminders = get_todays_reminders(profile, vacc_log)

    age_note = f"Flock age: *{flock_age_weeks} weeks*\n" if flock_age_weeks else ""
    reminder_note = f"\n⏰ *{len(reminders)} reminder(s) due today* — reply VACC to see them." if reminders else ""

    await send_whatsapp_message(phone,
        f"👋 *Welcome back to KokoAlert!*\n\n"
        f"{age_note}"
        f"Farm risk score: *{risk['score']}/100 — {risk['category']}* "
        f"{risk['emoji']}{reminder_note}\n\n"
        f"Send a *voice note* to check your flock, or use a command:\n"
        f"*RISK* · *VACC* · *BIOSEC* · *HELP* · *DOC*"
    )