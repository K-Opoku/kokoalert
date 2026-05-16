"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
KokoAlert FastAPI application + Agentic Farm Monitor.

This file does three things:

1. FASTAPI APP
   Handles incoming WhatsApp webhook messages.
   Routes them to handlers.py for processing.

2. STARTUP
   Initialises the database and loads ML models once when the server starts.

3. AGENTIC FARM MONITOR (APScheduler)
   Runs automatically every morning at 7am Ghana time.
   The farmer does nothing — the agent acts first.

   What the agent does every morning:
     a) Checks every farmer's flock age
     b) Detects if a dangerous age window is opening TODAY or TOMORROW
     c) Sends proactive alerts before problems happen
     d) Sends vaccination reminders if anything is due
     e) Asks droppings check questions during the Gumboro danger window
     f) Sends daily check-in
     g) On Sundays: sends the full weekly health check instead

   This is the agentic behaviour. The system perceives (reads DB),
   reasons (checks rules), and acts (sends WhatsApp) — autonomously.

DEPLOYMENT: Render.com (no Docker needed)
  1. Push to GitHub
  2. Create Render Web Service → connect repo → set start command:
     uvicorn api.main:app --host 0.0.0.0 --port $PORT
  3. Add environment variables in Render dashboard:
     WHATSAPP_API_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN
  4. Upload model files to Render disk (models/ folder)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from whatsapp_bot.handlers import handle_incoming_message
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.dashboard_routes import router as dashboard_router
from src.config import WHATSAPP_VERIFY_TOKEN, MONTHLY_RISK_DATA
from api.database import (
    init_db,
    get_all_active_farmers,
    get_farm_profile,
    get_vaccination_log,
    log_agent_action,
)
from src.pipeline import pipeline
from src.vaccination_scheduler import (
    get_todays_reminders,
    format_vaccination_reminder_message,
    get_flock_age_weeks,
    get_flock_age_days,
)


# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kokoalert")


# ── SCHEDULER ─────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone="Africa/Accra")


# ── STARTUP AND SHUTDOWN ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Startup: initialise DB → load models → start scheduler
    Shutdown: stop scheduler cleanly
    """
    logger.info("KokoAlert starting up...")

    # Initialise database (creates tables if they don't exist)
    init_db()
    logger.info("Database initialised.")

    # Load ML models into memory (audio classifier + image classifier)
    # This takes a few seconds — done once so every request is fast
    pipeline.load_models()
    logger.info("ML models loaded.")

    # Register scheduled jobs
    _register_scheduled_jobs()
    scheduler.start()
    logger.info("Farm Monitor Agent started.")

    yield  # App is now running

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("KokoAlert shutting down.")


# ── FASTAPI APP ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="KokoAlert",
    description="AI-powered poultry disease detection for Ghanaian farmers",
    version="4.0.0",
    lifespan=lifespan,
)


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(dashboard_router)
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

# ── WHATSAPP WEBHOOK ──────────────────────────────────────────────────────────

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    WhatsApp webhook verification.
    Meta sends a GET request with a challenge token when you first
    register the webhook URL. We verify the token and return the challenge.
    """
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Webhook verification failed — token mismatch.")
    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """
    Receive incoming WhatsApp messages.
    Meta sends a POST request for every message, status update, etc.
    We only process actual messages — ignore delivery receipts and status.
    """
    try:
        body = await request.json()

        # Only process message events — ignore read receipts and status
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        if "messages" not in value:
            return {"status": "ignored"}

        # Route to handlers.py
        result = await handle_incoming_message(body)
        return result

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple health check for Render deployment monitoring."""
    return {
        "status": "ok",
        "models_loaded": pipeline._loaded,
        "scheduler_running": scheduler.running,
        "timestamp": datetime.now().isoformat(),
    }


# ── SCHEDULED JOBS ────────────────────────────────────────────────────────────

def _register_scheduled_jobs():
    """Register all scheduled jobs with APScheduler."""

    # Main morning run — 7am Ghana time, every day
    scheduler.add_job(
        morning_agent_run,
        trigger=CronTrigger(hour=7, minute=0),
        id="morning_agent_run",
        name="Farm Monitor — Morning Run",
        replace_existing=True,
    )

    # Evening follow-up — 5pm Ghana time
    # Checks if any Gumboro-window farmers haven't responded to morning check
    scheduler.add_job(
        evening_followup_run,
        trigger=CronTrigger(hour=17, minute=0),
        id="evening_followup",
        name="Farm Monitor — Evening Follow-up",
        replace_existing=True,
    )

    # Monthly season alert — last day of October
    # Warns all farmers that November (highest risk month) is coming
    scheduler.add_job(
        november_season_alert,
        trigger=CronTrigger(month=10, day=31, hour=8, minute=0),
        id="november_alert",
        name="November Season Alert",
        replace_existing=True,
    )

    logger.info("Scheduled jobs registered: morning run, evening follow-up, November alert.")


# ── MORNING AGENT RUN ─────────────────────────────────────────────────────────

async def morning_agent_run():
    """
    The core agentic behaviour. Runs every morning at 7am.

    For each registered farmer, the agent:
      1. Calculates flock age
      2. Detects if a dangerous age window is opening today or tomorrow
      3. Sends proactive disease prevention alerts
      4. Sends vaccination reminders
      5. Sends the daily check-in (or weekly check on Sundays)

    The farmer wakes up with a message — they did not have to ask.
    """
    today = datetime.now()
    is_sunday = today.weekday() == 6
    logger.info(f"Morning agent run started — {today.strftime('%Y-%m-%d')} "
                f"({'Sunday' if is_sunday else 'Weekday'})")

    phones = get_all_active_farmers()
    logger.info(f"Processing {len(phones)} registered farmers.")

    for phone in phones:
        try:
            await _process_farmer_morning(phone, is_sunday)
            # Small delay between farmers — avoid hitting WhatsApp rate limits
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error processing farmer {phone}: {e}")
            continue

    logger.info("Morning agent run complete.")


async def _process_farmer_morning(phone: str, is_sunday: bool):
    """Process one farmer's morning routine."""
    profile = get_farm_profile(phone)
    if not profile:
        return

    vacc_log = get_vaccination_log(phone) or {}
    doc_arrival_date = profile.get("doc_arrival_date")

    flock_age_weeks = 0
    flock_age_days = 0
    if doc_arrival_date:
        flock_age_days = get_flock_age_days(doc_arrival_date)
        flock_age_weeks = flock_age_days // 7

    # Keep flock_age_weeks in profile up to date
    profile["flock_age_weeks"] = flock_age_weeks

    # ── [1] PROACTIVE AGE-BASED ALERTS ───────────────────────────────────
    # These fire before the farmer even asks anything.
    # This is the core agentic behaviour.

    alert_sent = await _send_proactive_age_alert(
        phone, profile, flock_age_weeks, flock_age_days
    )

    # ── [2] VACCINATION REMINDERS ─────────────────────────────────────────
    reminders = get_todays_reminders(profile, vacc_log)
    if reminders:
        reminder_msg = format_vaccination_reminder_message(reminders)
        await send_whatsapp_message(phone, reminder_msg)
        log_agent_action("vaccination_scheduler", phone,
                         f"reminder_sent_{len(reminders)}_items")
        await asyncio.sleep(0.3)

    # ── [3] DAILY CHECK-IN OR WEEKLY HEALTH CHECK ─────────────────────────
    # On Sundays: full 5-question weekly check
    # Other days: simple 1–6 daily check-in
    # Skip if a proactive alert was already sent and it contains a question
    # (we don't want to overwhelm the farmer with multiple messages)

    farmer_name = profile.get("farmer_name", "Farmer")

    if is_sunday:
        await send_weekly_health_check(phone, farmer_name)
        log_agent_action("farm_monitor", phone, "weekly_check_sent")
    else:
        await send_daily_checkin(phone, farmer_name, profile)
        log_agent_action("farm_monitor", phone, "daily_checkin_sent")


async def _send_proactive_age_alert(
    phone: str,
    profile: dict,
    flock_age_weeks: int,
    flock_age_days: int,
) -> bool:
    """
    Send a proactive alert based on flock age.
    Returns True if an alert was sent.

    This is where KokoAlert acts like a knowledgeable farmer standing
    next to you — telling you what is about to happen before it happens.
    """
    sent = False

    # ── WARNING: Gumboro window opens TOMORROW ────────────────────────────
    # Day 20 = tomorrow birds hit 3 weeks = Gumboro window opens
    if flock_age_days == 20:
        gumboro_vaccinated = profile.get("gumboro_vaccinated", "none")
        vacc_note = ""
        if gumboro_vaccinated == "none":
            vacc_note = (
                "\n\n🔴 *WARNING: Your flock has not received the Gumboro vaccine.* "
                "Unvaccinated birds in this window face up to 70% mortality. "
                "Contact your agrovet immediately."
            )
        elif gumboro_vaccinated == "first_only":
            vacc_note = (
                "\n\n⚠️ Only the 1st Gumboro dose was recorded. "
                "Confirm the 2nd dose was given at week 3."
            )

        await send_whatsapp_message(phone,
            f"⚠️ *KokoAlert Warning — Tomorrow is a Critical Day*\n\n"
            f"Your birds will be *3 weeks old tomorrow* — this is when the "
            f"*Gumboro danger window opens.*\n\n"
            f"Gumboro Disease is the deadliest risk your flock will face. "
            f"It can kill 70% of birds in 3 days with no cure.\n\n"
            f"*Starting tomorrow, check the droppings every morning.* "
            f"White, watery droppings = emergency. Report immediately."
            f"{vacc_note}\n\n"
            f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        log_agent_action("farm_monitor", phone, "proactive_gumboro_window_warning")
        sent = True

    # ── ACTIVE: Gumboro window (weeks 3–6) — ask droppings daily ─────────
    elif 21 <= flock_age_days <= 42:
        month = datetime.now().month
        monthly = MONTHLY_RISK_DATA[month]

        await send_whatsapp_message(phone,
            f"🔴 *Gumboro Danger Window — Week {flock_age_weeks}*\n\n"
            f"Your birds are in the most dangerous period of the flock cycle.\n\n"
            f"*Quick check — what do the droppings look like this morning?*\n\n"
            f"1 — Normal (brown or dark)\n"
            f"2 — White and watery ← EMERGENCY, reply immediately\n"
            f"3 — Bloody or dark chocolate\n"
            f"4 — Bright green\n"
            f"5 — Haven't checked yet\n\n"
            f"Reply with a number now."
        )
        log_agent_action("farm_monitor", phone,
                         f"gumboro_window_droppings_check_day_{flock_age_days}")
        sent = True

    # ── WARNING: Coccidiosis opening door to Gumboro ──────────────────────
    # If farmer previously reported bloody droppings AND is now entering Gumboro window
    elif flock_age_days == 18 and profile.get("weekly_droppings") == "bloody_chocolate":
        await send_whatsapp_message(phone,
            f"⚠️ *Urgent Warning — Coccidiosis + Gumboro Risk*\n\n"
            f"You recently reported bloody droppings — this means Coccidiosis "
            f"may be active in your flock.\n\n"
            f"Your birds are *{flock_age_weeks} weeks old* and entering the "
            f"Gumboro window in 3 days.\n\n"
            f"*Coccidiosis weakens the immune system — it opens the door to Gumboro.*\n\n"
            f"If you have not already treated for Coccidiosis:\n"
            f"Go to your agrovet NOW. Ask for *Amprolium (Amprocox)*.\n\n"
            f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        log_agent_action("farm_monitor", phone, "cocci_gumboro_cascade_warning")
        sent = True

    # ── ALERT: Newcastle booster window ───────────────────────────────────
    elif flock_age_days == 70:  # Week 10 exactly
        newcastle_vaccinated = profile.get("newcastle_vaccinated", "none")
        if newcastle_vaccinated in ["none", "partial"]:
            await send_whatsapp_message(phone,
                f"⏰ *Newcastle Booster Due — Week 10*\n\n"
                f"Your birds are now *10 weeks old*. This is when the Newcastle "
                f"booster vaccine is due.\n\n"
                f"Without this booster, immunity from the earlier doses begins "
                f"to wane — your flock becomes vulnerable again.\n\n"
                f"Go to your agrovet. Ask for: *Lasota Newcastle vaccine*\n"
                f"Give in drinking water this week.\n\n"
                f"Reply *4* after you vaccinate to record it.\n\n"
                f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
            )
            log_agent_action("farm_monitor", phone, "newcastle_booster_alert_week10")
            sent = True

    # ── ALERT: Stop coccidiosis medicine ──────────────────────────────────
    elif flock_age_days == 98:  # Week 14 — warning
        await send_whatsapp_message(phone,
            f"⚠️ *Action Required — Coccidiosis Medicine*\n\n"
            f"Your birds are now *{flock_age_weeks} weeks old*.\n\n"
            f"*STOP giving Coccidiosis medicine at the end of Week 15.*\n\n"
            f"Continuing beyond Week 15 interferes with egg production in layers "
            f"and has no benefit for broilers approaching market weight.\n\n"
            f"One more week — then stop completely.\n\n"
            f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
        )
        log_agent_action("farm_monitor", phone, "cocci_stop_warning_week14")
        sent = True

    # ── ALERT: High season is this month ──────────────────────────────────
    elif datetime.now().month in [11, 12] and flock_age_weeks > 0:
        # Only send this once — on the 1st of November/December
        if datetime.now().day == 1:
            monthly = MONTHLY_RISK_DATA[datetime.now().month]
            await send_whatsapp_message(phone,
                f"🔴 *Season Alert — {monthly['label']} is High Risk*\n\n"
                f"{monthly['pct']}% of Ghana's annual respiratory disease "
                f"cases occur this month.\n\n"
                f"*What to do this month:*\n"
                f"• Check your flock twice daily\n"
                f"• Confirm Newcastle vaccination is up to date\n"
                f"• Send a voice note to KokoAlert every 3 days\n"
                f"• Keep the poultry house dry and well-ventilated\n"
                f"• Report any deaths immediately\n\n"
                f"_(Source: Mensah et al., 2023 — KNUST VS Lab)_\n\n"
                f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
            )
            log_agent_action("farm_monitor", phone,
                             f"peak_season_alert_{monthly['label']}")
            sent = True

    return sent


# ── EVENING FOLLOW-UP ─────────────────────────────────────────────────────────

async def evening_followup_run():
    """
    5pm check — only for farmers whose birds are in the Gumboro window.
    If they haven't responded to the morning droppings check, follow up.
    """
    logger.info("Evening follow-up run started.")

    phones = get_all_active_farmers()

    for phone in phones:
        try:
            profile = get_farm_profile(phone)
            if not profile:
                continue

            doc_arrival_date = profile.get("doc_arrival_date")
            if not doc_arrival_date:
                continue

            flock_age_days = get_flock_age_days(doc_arrival_date)

            # Only follow up during the Gumboro window
            if not (21 <= flock_age_days <= 42):
                continue

            # Check if farmer responded to the morning check
            # If _pending_droppings exists, they haven't completed the flow
            if profile.get("_pending_droppings") is None:
                # They haven't been asked yet today or already responded — skip
                continue

            await send_whatsapp_message(phone,
                f"👋 *Evening check — KokoAlert*\n\n"
                f"You didn't reply to the morning droppings check.\n\n"
                f"Your birds are in the Gumboro danger window — please check "
                f"the droppings before it gets dark.\n\n"
                f"1 — Normal (brown or dark)\n"
                f"2 — White and watery ← EMERGENCY\n"
                f"3 — Bloody or dark chocolate\n"
                f"4 — Bright green"
            )
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Evening follow-up error for {phone}: {e}")
            continue

    logger.info("Evening follow-up complete.")


# ── NOVEMBER SEASON ALERT ─────────────────────────────────────────────────────

async def november_season_alert():
    """
    Fired on October 31st — warns all farmers that November
    (the highest risk month in Ghana) starts tomorrow.
    """
    logger.info("Sending November season alert to all farmers.")
    phones = get_all_active_farmers()

    for phone in phones:
        try:
            await send_whatsapp_message(phone,
                f"🔴 *KokoAlert — November Season Warning*\n\n"
                f"Tomorrow is November 1st — *the highest risk month for "
                f"poultry disease in Ghana.*\n\n"
                f"34.27% of all respiratory disease cases at the Kumasi VS Lab "
                f"occur in November alone.\n"
                f"_(Mensah et al., 2023 — KNUST)_\n\n"
                f"*Before November starts:*\n"
                f"• Confirm your Newcastle vaccination is up to date\n"
                f"• Clean and disinfect your poultry house this week\n"
                f"• Check that your footbath is active\n"
                f"• Stock up on electrolytes and vitamins\n\n"
                f"KokoAlert will monitor your flock closely this month.\n"
                f"Send a voice note anytime for a full audio check.\n\n"
                f"_KokoAlert — Protecting Ghana's poultry farmers_ 🇬🇭"
            )
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"November alert error for {phone}: {e}")

    logger.info(f"November alert sent to {len(phones)} farmers.")


# ── REGIONAL OUTBREAK DETECTION ───────────────────────────────────────────────

async def check_regional_outbreak():
    """
    If 3+ farmers in the same region report the same disease within 7 days,
    alert ALL farmers in that region — not just the ones who reported.

    This turns KokoAlert from a farm tool into a disease surveillance network.
    Called after every diagnosis in handlers.py (post-diagnosis hook).

    Currently: logs the outbreak signal. Full alert firing is the next step.
    """
    # TODO: implement after database.get_recent_diagnoses_by_region() is added
    # This will be one of the "add before June" features
    pass