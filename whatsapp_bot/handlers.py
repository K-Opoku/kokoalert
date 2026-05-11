import httpx
import tempfile
import os
from datetime import datetime

from src.pipeline import pipeline
from src.risk_engine import compute_farm_risk_score
from src.biosecurity_scorer import compute_biosecurity_score
from src.vaccination_tracker import get_all_vaccination_statuses
from api.database import (
    get_farm_profile, save_farm_profile,
    save_analysis_result, log_agent_action,
    get_onboarding_state, set_onboarding_state,
    clear_onboarding_state
)
from src.config import (
    WHATSAPP_API_URL, WHATSAPP_API_TOKEN,
    VSD_CONTACTS, ONBOARDING_QUESTIONS
)


# ── SEND MESSAGE ─────────────────────────────────────────────────────────────

async def send_whatsapp_message(phone: str, message: str):
    """Send a plain text WhatsApp message."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": message}
    }
    async with httpx.AsyncClient() as client:
        await client.post(WHATSAPP_API_URL, json=payload, headers=headers)


# ── MAIN ROUTER ──────────────────────────────────────────────────────────────

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

        # Check if farmer is mid-onboarding
        onboarding_state = get_onboarding_state(phone)

        if msg_type == "text":
            text = message.get("text", {}).get("body", "").strip()

            if onboarding_state is not None:
                # Farmer is answering onboarding questions
                await handle_onboarding_reply(phone, text, onboarding_state)

            elif text.upper() in ["HI", "HELLO", "START", "KOKOALERT"]:
                # New farmer — check if they have a profile
                profile = get_farm_profile(phone)
                if not profile:
                    await start_onboarding(phone)
                else:
                    await send_main_menu(phone, profile)

            else:
                await handle_command(phone, text.upper())

        elif msg_type == "audio":
            audio_id = message.get("audio", {}).get("id")
            await handle_audio_message(phone, audio_id)

        return {"status": "processed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── ONBOARDING ────────────────────────────────────────────────────────────────

async def start_onboarding(phone: str):
    """Send the first onboarding question."""
    # State = index of the current question (0-based)
    set_onboarding_state(phone, 0)
    await send_whatsapp_message(phone, ONBOARDING_QUESTIONS[0]["question"])


async def handle_onboarding_reply(phone: str, reply: str, question_index: int):
    """
    Process a reply to an onboarding question.
    Save the answer, move to the next question or finish setup.
    """
    question = ONBOARDING_QUESTIONS[question_index]
    options = question["options"]

    # Validate reply
    if reply not in options:
        await send_whatsapp_message(phone,
            f"Please reply with one of the numbers shown.\n\n"
            f"{question['question']}"
        )
        return

    # Save this answer to the partial profile
    profile = get_farm_profile(phone) or {}
    profile[question["key"]] = options[reply]
    save_farm_profile(phone, profile)

    next_index = question_index + 1

    if next_index < len(ONBOARDING_QUESTIONS):
        # Send next question
        set_onboarding_state(phone, next_index)
        await send_whatsapp_message(phone, ONBOARDING_QUESTIONS[next_index]["question"])
    else:
        # Onboarding complete
        clear_onboarding_state(phone)
        await send_whatsapp_message(phone,
            "✅ *Farm profile saved!*\n\n"
            "You are all set. Here is how to use KokoAlert:\n\n"
            "🎙️ *Send a voice note* from inside your poultry house → "
            "I will tell you if your flock sounds healthy or abnormal.\n\n"
            "Every morning I will also send you a daily check-in.\n\n"
            "Commands you can use anytime:\n"
            "*RISK* — your current farm risk score\n"
            "*VACC* — vaccination schedule\n"
            "*BIOSEC* — biosecurity score\n"
            "*HELP* — emergency vet contacts\n\n"
            "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        log_agent_action("onboarding", phone, "onboarding_complete")


# ── DAILY CHECK-IN (sent by Farm Monitor Agent, replied to here) ──────────────

async def send_daily_checkin(phone: str, farmer_name: str):
    """
    Sent by the Farm Monitor Agent every morning.
    Farmer replies with a number — structured, no typing required.
    """
    await send_whatsapp_message(phone,
        f"🐔 *Good morning {farmer_name}!*\n\n"
        f"How is your flock today?\n\n"
        f"1 — All good\n"
        f"2 — Some birds look sick or slow\n"
        f"3 — Birds have died\n"
        f"4 — I vaccinated today\n"
        f"5 — I brought in new birds\n\n"
        f"Or send a *voice note* from inside your poultry house "
        f"for a full audio check."
    )


async def handle_daily_checkin_reply(phone: str, reply: str):
    """Process the farmer's daily check-in reply."""
    profile = get_farm_profile(phone) or {}
    region = profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    if reply == "1":
        await send_whatsapp_message(phone,
            "✅ Good to hear. Keep up your regular checks.\n\n"
            "Send a voice note anytime for a full audio analysis."
        )

    elif reply == "2":
        await send_whatsapp_message(phone,
            "⚠️ *Birds looking sick or slow*\n\n"
            "Send a *voice note* from inside the house now so I can "
            "analyse the flock. Then do a visual check:\n\n"
            "• Are birds coughing or sneezing?\n"
            "• Any nasal discharge or swollen faces?\n"
            "• Are birds sitting away from the group?\n\n"
            "If you see any of those signs, contact your vet today.\n"
            f"📞 {vsd['office']}: {vsd['phone']}"
        )

    elif reply == "3":
        # Update profile — deaths reported
        profile["recent_deaths"] = True
        save_farm_profile(phone, profile)

        await send_whatsapp_message(phone,
            "❗ *Bird deaths reported*\n\n"
            "How many birds died? Reply with a number (e.g. *5*)."
        )
        # Set state to await death count
        set_onboarding_state(phone, "awaiting_death_count")

    elif reply == "4":
        # Farmer vaccinated today
        today = datetime.now().strftime("%Y-%m-%d")
        profile["vaccinated"] = True
        profile["days_since_vaccination"] = 0
        profile["last_vaccination_date"] = today
        save_farm_profile(phone, profile)

        await send_whatsapp_message(phone,
            "✅ *Vaccination recorded for today.*\n\n"
            "I have updated your farm profile. Your next booster reminder "
            "will come in 90 days.\n\n"
            "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        log_agent_action("farm_monitor", phone, "vaccination_recorded")

    elif reply == "5":
        # New birds introduced
        profile["new_birds_introduced"] = True
        save_farm_profile(phone, profile)

        await send_whatsapp_message(phone,
            "⚠️ *New birds recorded.*\n\n"
            "Remember: always keep new birds in a *separate house for 14 days* "
            "before mixing with your existing flock. New birds are the most "
            "common way disease enters a farm.\n\n"
            "Your risk score has been updated."
        )
        log_agent_action("farm_monitor", phone, "new_birds_recorded")

    else:
        await send_whatsapp_message(phone,
            "Please reply with a number 1 to 5, "
            "or send a voice note for a full audio check."
        )


# ── WEEKLY HEALTH CHECK ───────────────────────────────────────────────────────

WEEKLY_CHECK_QUESTIONS = [
    {
        "key": "weekly_pecking",
        "question": (
            "🗓️ *Weekly Flock Health Check — Question 1 of 4*\n\n"
            "Are birds pecking or attacking each other?\n\n"
            "1 — No\n"
            "2 — Yes, occasionally\n"
            "3 — Yes, it is a serious problem"
        ),
        "options": {"1": "none", "2": "occasional", "3": "serious"}
    },
    {
        "key": "weekly_appetite",
        "question": (
            "*Question 2 of 4*\n\n"
            "Are birds eating and drinking normally?\n\n"
            "1 — Yes, normal\n"
            "2 — Some birds not eating\n"
            "3 — Most birds have reduced appetite"
        ),
        "options": {"1": "normal", "2": "some_reduced", "3": "most_reduced"}
    },
    {
        "key": "weekly_deaths",
        "question": (
            "*Question 3 of 4*\n\n"
            "Any unexpected bird deaths this week?\n\n"
            "1 — No deaths\n"
            "2 — 1 to 3 deaths\n"
            "3 — More than 3 deaths"
        ),
        "options": {"1": "none", "2": "1_to_3", "3": "over_3"}
    },
    {
        "key": "weekly_new_birds",
        "question": (
            "*Question 4 of 4*\n\n"
            "Did you bring any new birds onto the farm this week?\n\n"
            "1 — No\n"
            "2 — Yes"
        ),
        "options": {"1": False, "2": True}
    },
]


async def send_weekly_health_check(phone: str, farmer_name: str):
    """
    Sent by Farm Monitor Agent every Sunday.
    Covers non-audio health issues: pecking, appetite, deaths, new birds.
    """
    set_onboarding_state(phone, "weekly_0")
    await send_whatsapp_message(phone,
        f"👋 *Hi {farmer_name}!* Time for your weekly flock check.\n\n"
        f"4 quick questions — reply with a number.\n\n"
        + WEEKLY_CHECK_QUESTIONS[0]["question"]
    )


async def handle_weekly_check_reply(
    phone: str, reply: str, state: str
):
    """Process a weekly health check reply and give feedback at the end."""
    idx = int(state.replace("weekly_", ""))
    question = WEEKLY_CHECK_QUESTIONS[idx]
    options = question["options"]

    if reply not in options:
        await send_whatsapp_message(phone,
            f"Please reply with a number.\n\n{question['question']}"
        )
        return

    # Save answer
    profile = get_farm_profile(phone) or {}
    answer = options[reply]
    profile[question["key"]] = answer

    # Update live risk fields from weekly check
    if question["key"] == "weekly_deaths" and answer in ["1_to_3", "over_3"]:
        profile["recent_deaths"] = True
    if question["key"] == "weekly_new_birds" and answer is True:
        profile["new_birds_introduced"] = True

    save_farm_profile(phone, profile)

    next_idx = idx + 1

    if next_idx < len(WEEKLY_CHECK_QUESTIONS):
        set_onboarding_state(phone, f"weekly_{next_idx}")
        await send_whatsapp_message(
            phone, WEEKLY_CHECK_QUESTIONS[next_idx]["question"]
        )
    else:
        # Weekly check complete — generate feedback
        clear_onboarding_state(phone)
        await send_weekly_check_feedback(phone, profile)


async def send_weekly_check_feedback(phone: str, profile: dict):
    """After weekly check, give the farmer a summary with any action needed."""
    issues = []

    pecking = profile.get("weekly_pecking", "none")
    appetite = profile.get("weekly_appetite", "normal")
    deaths = profile.get("weekly_deaths", "none")
    new_birds = profile.get("weekly_new_birds", False)

    if pecking == "serious":
        issues.append(
            "⚠️ *Serious pecking reported.*\n"
            "This is usually caused by overcrowding, too much light, "
            "or nutritional deficiency. Reduce light intensity and "
            "check stocking density."
        )
    elif pecking == "occasional":
        issues.append(
            "🟡 *Occasional pecking noted.* Monitor closely. "
            "If it increases, check crowding and lighting."
        )

    if appetite in ["some_reduced", "most_reduced"]:
        issues.append(
            "⚠️ *Reduced appetite reported.*\n"
            "Poor appetite combined with any respiratory sounds is a serious warning sign. "
            "Send a voice note today for an audio check."
        )

    if deaths == "over_3":
        issues.append(
            "❗ *Multiple bird deaths reported.*\n"
            "This needs immediate attention. Send a voice note and "
            "contact your vet today."
        )
    elif deaths == "1_to_3":
        issues.append(
            "🟡 *Some deaths reported.* Monitor closely. "
            "If deaths continue tomorrow, contact your vet."
        )

    if new_birds:
        issues.append(
            "⚠️ *New birds introduced.*\n"
            "Keep them in a separate house for 14 days. "
            "Your risk score has been updated."
        )

    if not issues:
        await send_whatsapp_message(phone,
            "✅ *Weekly check complete — all good!*\n\n"
            "No major issues reported this week. "
            "Keep up the routine daily checks.\n\n"
            "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
    else:
        msg = "📋 *Weekly Check Summary*\n\n" + "\n\n".join(issues)
        msg += "\n\n_Reply RISK to see your updated farm risk score._"
        await send_whatsapp_message(phone, msg)


# ── COMMAND HANDLER ───────────────────────────────────────────────────────────

async def handle_command(phone: str, command: str):
    """Handle keyword commands."""
    profile = get_farm_profile(phone) or {}
    region = profile.get("region", "Ashanti")
    vsd = VSD_CONTACTS.get(region, VSD_CONTACTS["Ashanti"])

    # Handle death count reply (follows reply "3" to daily check)
    state = get_onboarding_state(phone)
    if state == "awaiting_death_count":
        try:
            count = int(command)
            profile["recent_deaths"] = True
            profile["death_count_this_week"] = count
            save_farm_profile(phone, profile)
            clear_onboarding_state(phone)

            await send_whatsapp_message(phone,
                f"❗ *{count} bird deaths recorded.*\n\n"
                f"Your risk score has been updated.\n\n"
                f"Send a voice note from inside the house so I can check "
                f"for respiratory symptoms.\n\n"
                f"If deaths continue, contact your vet today:\n"
                f"📞 {vsd['office']}: {vsd['phone']}"
            )
        except ValueError:
            await send_whatsapp_message(phone,
                "Please reply with a number — how many birds died? (e.g. *5*)"
            )
        return

    # Handle daily check-in numbered replies
    if command in ["1", "2", "3", "4", "5"]:
        await handle_daily_checkin_reply(phone, command)
        return

    # Handle weekly check numbered replies
    if state and state.startswith("weekly_"):
        await handle_weekly_check_reply(phone, command, state)
        return

    # Standard commands
    if command == "RISK":
        profile["current_month"] = datetime.now().month
        result = compute_farm_risk_score(profile)

        await send_whatsapp_message(phone,
            f"{result['emoji']} *Farm Risk Score*\n\n"
            f"*Score:* {result['score']}/100 — *{result['category']}*\n\n"
            f"*{result['monthly_data']['label']}* risk level: "
            f"*{result['monthly_data']['level']}*\n"
            f"{result['monthly_data']['pct']}% of annual Ashanti cases "
            f"occur this month.\n"
            f"_(KNUST Vet Lab, 2018–2021)_\n\n"
            f"*Top action:* {result['top_action']}\n\n"
            f"Send a voice note to check your flock now."
        )

    elif command == "VACC":
        statuses = get_all_vaccination_statuses(profile)
        msg = "💉 *Vaccination Status*\n\n"
        for s in statuses:
            urgency = s.get("urgency", "normal")
            icon = "❗" if urgency == "urgent" else ("🟡" if urgency == "soon" else "✅")
            msg += f"{icon} *{s['disease']}*\n"
            if s.get("next_due_date"):
                msg += f"Next due: {s['next_due_date']}\n"
            if s.get("next_vaccine"):
                msg += f"Vaccine: {s['next_vaccine']}\n"
            msg += "\n"
        msg += "_Reply RISK to see your full farm risk score._"
        await send_whatsapp_message(phone, msg)

    elif command == "BIOSEC":
        result = compute_biosecurity_score(profile)
        msg = (
            f"🔒 *Biosecurity Score*\n\n"
            f"*Your score:* {result['score']}/10 — *{result['grade']}*\n\n"
        )
        if result["improvements"]:
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
            "If birds are dying: *DO NOT move them.* "
            "Contain the flock and call now."
        )

    else:
        await send_whatsapp_message(phone,
            "Send a *voice note* from inside your poultry house to check your flock.\n\n"
            "Or use a command:\n"
            "*RISK* · *VACC* · *BIOSEC* · *HELP*"
        )


# ── AUDIO MESSAGE HANDLER ─────────────────────────────────────────────────────

async def handle_audio_message(phone: str, audio_id: str):
    """Download WhatsApp voice note, run pipeline, send contextualised result."""
    await send_whatsapp_message(phone,
        "🧠 *Analysing your flock...* Please wait about 10 seconds."
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

        # Save to temp file and analyse
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_resp.content)
            tmp_path = tmp.name

        farm_profile = get_farm_profile(phone) or {}

        if not pipeline._loaded:
            pipeline.load_models()

        result = pipeline.analyse_audio(tmp_path, farm_profile=farm_profile)
        os.unlink(tmp_path)

        # Format and send result
        await send_audio_result(phone, result, farm_profile)

        # Save to database
        save_analysis_result(phone, result)

    except Exception as e:
        await send_whatsapp_message(phone,
            "❌ Something went wrong analysing your recording.\n\n"
            "Please try again, or type *HELP* for emergency vet contacts."
        )


async def send_audio_result(phone: str, result: dict, farm_profile: dict):
    """Format the pipeline result into a clear WhatsApp message."""
    status = result.get("status")
    emoji = result.get("emoji", "❓")
    headline = result.get("headline", "")

    if status == "healthy":
        await send_whatsapp_message(phone,
            f"{emoji} *{headline}*\n\n"
            f"{result.get('immediate', '')}\n\n"
            f"Windows analysed: {result.get('windows_analysed', 'N/A')}\n\n"
            f"_Send a voice note anytime to check again._ 🇬🇭"
        )

    elif status == "anomaly_detected":
        # Build rich contextualised message
        msg = f"{emoji} *{headline}*\n\n"
        msg += f"{result.get('immediate', '')}\n\n"

        # Context lines from risk engine
        context = result.get("context_lines", [])
        if context:
            msg += "*Your farm context:*\n"
            for line in context:
                msg += f"{line}\n"
            msg += "\n"

        msg += f"*Next 24 hours:*\n{result.get('next_24h', '')}\n\n"

        # VSD contact
        vsd = result.get("vsd_contact", {})
        if vsd:
            msg += (
                f"📞 *Contact VSD now:*\n"
                f"{vsd.get('office', '')}\n"
                f"{vsd.get('phone', '')}\n\n"
            )

        msg += "_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        await send_whatsapp_message(phone, msg)

    elif status == "inconclusive":
        await send_whatsapp_message(phone,
            f"{emoji} *{headline}*\n\n"
            f"{result.get('immediate', '')}"
        )


def send_main_menu(phone: str, profile: dict):
    """Returning farmer — show main menu."""
    from datetime import datetime
    profile["current_month"] = datetime.now().month
    risk = compute_farm_risk_score(profile)

    return send_whatsapp_message(phone,
        f"👋 *Welcome back to KokoAlert!*\n\n"
        f"Your farm risk score: *{risk['score']}/100 — {risk['category']}* "
        f"{risk['emoji']}\n\n"
        f"Send a *voice note* to check your flock, or use a command:\n"
        f"*RISK* · *VACC* · *BIOSEC* · *HELP*"
    )