"""
api/database.py
─────────────────────────────────────────────────────────────────────────────
KokoAlert database layer using SQLite.

SQLite is used because:
  - Zero configuration — no separate database server needed
  - File-based — easy to back up, inspect, and deploy
  - Fast enough for the scale of a competition demo and early production
  - Can be swapped for PostgreSQL later without changing any other module

Tables:
  farm_profiles      — one row per farmer (phone is primary key)
  vaccination_logs   — one row per farmer, stores {vaccine_id: date} as JSON
  analysis_results   — append-only log of every diagnosis
  agent_actions      — append-only log of every agent action
  conversation_state — one row per farmer, stores current WhatsApp state
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

# ── DATABASE LOCATION ─────────────────────────────────────────────────────────
DB_PATH = Path("data/kokoalert.db")


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database. Creates the file if missing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Rows behave like dicts
    return conn


def init_db():
    """
    Create all tables if they don't already exist.
    Call once at API startup in api/main.py.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS farm_profiles (
                phone           TEXT PRIMARY KEY,
                profile_json    TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vaccination_logs (
                phone           TEXT PRIMARY KEY,
                log_json        TEXT NOT NULL DEFAULT '{}',
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                phone           TEXT NOT NULL,
                result_json     TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_actions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                agent           TEXT NOT NULL,
                phone           TEXT NOT NULL,
                action          TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_state (
                phone           TEXT PRIMARY KEY,
                state           TEXT,
                updated_at      TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ── FARM PROFILE ──────────────────────────────────────────────────────────────

def get_farm_profile(phone: str) -> dict | None:
    """
    Retrieve a farmer's profile by phone number.
    Returns None if the farmer hasn't onboarded yet.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT profile_json FROM farm_profiles WHERE phone = ?", (phone,)
        ).fetchone()
        if row:
            return json.loads(row["profile_json"])
        return None
    finally:
        conn.close()


def save_farm_profile(phone: str, profile: dict) -> None:
    """
    Insert or update a farmer's profile.
    Uses UPSERT — safe to call on both new and existing farmers.
    """
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO farm_profiles (phone, profile_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at   = excluded.updated_at
        """, (phone, json.dumps(profile), now, now))
        conn.commit()
    finally:
        conn.close()


# ── VACCINATION LOG ───────────────────────────────────────────────────────────

def get_vaccination_log(phone: str) -> dict:
    """
    Return the vaccination log for a farmer.
    Format: {vaccine_id: "YYYY-MM-DD"}
    Example: {"gumboro_1": "2026-05-10", "newcastle_1": "2026-05-17"}
    Returns {} if no log exists.

    The log is stored separately from farm_profile so it can be reset
    when a new flock arrives without losing other farm data.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT log_json FROM vaccination_logs WHERE phone = ?", (phone,)
        ).fetchone()
        if row:
            return json.loads(row["log_json"])
        return {}
    finally:
        conn.close()


def save_vaccination_log(phone: str, log: dict) -> None:
    """
    Save or replace the vaccination log for a farmer.
    log format: {vaccine_id: "YYYY-MM-DD"}
    Overwrites any existing log for this phone.
    """
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO vaccination_logs (phone, log_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                log_json   = excluded.log_json,
                updated_at = excluded.updated_at
        """, (phone, json.dumps(log), now))
        conn.commit()
    finally:
        conn.close()


# ── ANALYSIS RESULTS ──────────────────────────────────────────────────────────

def save_analysis_result(phone: str, result: dict) -> None:
    """
    Append a diagnosis result to the analysis log.
    This is append-only — never overwrite history.
    Used for: monitoring, trend analysis, competition data story.
    """
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO analysis_results (phone, result_json, created_at) VALUES (?, ?, ?)",
            (phone, json.dumps(result), now)
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_analysis(phone: str, limit: int = 5) -> list[dict]:
    """
    Return the most recent diagnosis results for a farmer.
    Used by the Farm Monitor Agent to detect repeat issues.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT result_json, created_at FROM analysis_results "
            "WHERE phone = ? ORDER BY created_at DESC LIMIT ?",
            (phone, limit)
        ).fetchall()
        return [
            {**json.loads(r["result_json"]), "recorded_at": r["created_at"]}
            for r in rows
        ]
    finally:
        conn.close()


# ── AGENT ACTIONS ─────────────────────────────────────────────────────────────

def log_agent_action(agent: str, phone: str, action: str) -> None:
    """
    Log an action taken by the Farm Monitor Agent or any automated process.
    Append-only. Used for: debugging, audit trail, competition monitoring story.

    Args:
        agent:  "onboarding" | "diagnosis" | "farm_monitor" | "vaccination_scheduler"
        phone:  farmer's WhatsApp number
        action: short string, e.g. "diagnosis_gumboro_High" or "vaccine_reminder_sent"
    """
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO agent_actions (agent, phone, action, created_at) VALUES (?, ?, ?, ?)",
            (agent, phone, action, now)
        )
        conn.commit()
    finally:
        conn.close()


# ── CONVERSATION STATE (WhatsApp state machine) ───────────────────────────────

def get_onboarding_state(phone: str) -> str | None:
    """
    Return the current conversation state for a farmer.
    Returns None if no active state (normal routing applies).

    States used by handlers.py:
      onboarding_0 to onboarding_6   — mid-onboarding
      awaiting_droppings             — waiting for droppings colour reply
      awaiting_behavior              — waiting for behaviour signs reply
      awaiting_image                 — waiting for optional photo
      weekly_0 to weekly_4           — mid-weekly health check
      awaiting_death_count           — waiting for death count number
      awaiting_doc_date              — waiting for DOC arrival date
      awaiting_vaccine_confirm       — waiting for vaccine type confirmation
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT state FROM conversation_state WHERE phone = ?", (phone,)
        ).fetchone()
        if row:
            return row["state"]
        return None
    finally:
        conn.close()


def set_onboarding_state(phone: str, state: str) -> None:
    """Set the conversation state for a farmer."""
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO conversation_state (phone, state, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                state      = excluded.state,
                updated_at = excluded.updated_at
        """, (phone, state, now))
        conn.commit()
    finally:
        conn.close()


def clear_onboarding_state(phone: str) -> None:
    """Clear the conversation state — farmer is back in normal routing."""
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO conversation_state (phone, state, updated_at)
            VALUES (?, NULL, ?)
            ON CONFLICT(phone) DO UPDATE SET
                state      = NULL,
                updated_at = excluded.updated_at
        """, (phone, now))
        conn.commit()
    finally:
        conn.close()


# ── UTILITY ───────────────────────────────────────────────────────────────────

def get_all_active_farmers() -> list[str]:
    """
    Return phone numbers of all registered farmers.
    Used by Farm Monitor Agent to send morning check-ins.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT phone FROM farm_profiles ORDER BY created_at"
        ).fetchall()
        return [r["phone"] for r in rows]
    finally:
        conn.close()


def get_farmers_by_region(region: str) -> list[str]:
    """
    Return farmers in a specific region.
    Used for regional outbreak alerts.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT phone, profile_json FROM farm_profiles"
        ).fetchall()
        phones = []
        for r in rows:
            profile = json.loads(r["profile_json"])
            if profile.get("region") == region:
                phones.append(r["phone"])
        return phones
    finally:
        conn.close()