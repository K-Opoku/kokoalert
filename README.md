# KokoAlert 🐔
### AI-Powered Poultry Disease Detection and Farm Health Management for Ghanaian Smallholder Farmers

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Dashboard-22c55e?style=flat-square)](https://kokoalert-1.onrender.com/dashboard)
[![API Docs](https://img.shields.io/badge/API-Docs-3b82f6?style=flat-square)](https://kokoalert-1.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square)](https://tensorflow.org)

---

## What is KokoAlert?

KokoAlert is a WhatsApp-based AI system that helps Ghanaian poultry farmers detect respiratory diseases early, follow vaccination schedules, and understand their seasonal disease risk, before losses occur.

Farmers send a voice note from inside their poultry house. KokoAlert analyses the audio, asks follow-up questions, and sends back a confirmed disease flag with the exact drug name to ask for at the agrovet. No app installation. No technical knowledge required. Just WhatsApp.

**The problem it solves:** Only 8.9% of Ghanaian poultry farmers have monthly veterinary access. By the time visual symptoms appear, disease has already spread. KokoAlert gives farmers a detection capability that was previously only available at a veterinary clinic.

---

## Three Detection Systems

```
👂 EAR   — CNN audio classifier
           Farmer records 10–30s voice note inside poultry house
           Audio → mel spectrogram → binary CNN → healthy or anomalous

👁️ EYE   — MobileNetV2 image classifier (optional, conditional)
           Triggered only when audio is anomalous
           Droppings photo → 3-class classifier → healthy / coccidiosis / newcastle

🧠 BRAIN — Diagnosis engine
           Combines audio + image + droppings text + behaviour + flock age
           + vaccination status + season → confirmed disease with reasons
           + exact agrovet drug name + urgency level
```

---

## Five Diseases Covered

| Disease | Urgency | Treatable | Key Detection Signal |
|---------|---------|-----------|---------------------|
| Newcastle Disease | 🔴 Emergency | No cure | Bright green droppings + audio anomaly |
| Gumboro (IBD) | 🔴 Emergency | No cure | Age 3–6 weeks + white/watery droppings |
| Coccidiosis | 🟠 Urgent | ✅ Yes | Bloody/chocolate droppings |
| CRD (Amaman) | 🟠 Urgent | ✅ Yes | Audio anomaly + coughing + poor ventilation |
| Fowl Pox | 🟡 Monitor | Supportive | Face/comb lesions |

---

## Disease Cascade Detection

KokoAlert tracks the disease cascade that destroys entire flock cycles:

```
Coccidiosis (weeks 1–15)
  └─ Damages intestinal lining → weakens immune system
       └─ Opens door to Gumboro (weeks 3–6)
            └─ Destroys immune system completely
                 └─ Birds cannot respond to vaccines
                      └─ Newcastle, E.coli, Salmonella follow

CRD (any age, wet/dirty conditions)
  └─ Weakens respiratory tract
       └─ Opens door to Newcastle
```

When Coccidiosis is detected in a bird aged 3–6 weeks, KokoAlert automatically flags the Gumboro cascade risk in the diagnosis message.

---

## How It Works — End-to-End Flow

```
Farmer sends voice note
        │
        ▼
[1] preprocess.py
    load_audio() → peak_normalise() → slice_into_windows() → mel_spectrogram()
    Output: List of (128, 157, 1) spectrograms

        │
        ▼
[2] disease_classifier.py  ← 👂 EAR
    CNN classifies each 5-second window → Any-Two rule → anomaly flag
    Output: {is_anomalous, probability, anomalous_windows, total_windows}

        ├── Healthy → send healthy message → DONE
        │
        └── Anomalous
                │
                ▼
        [3] Ask droppings colour (5 options via WhatsApp)
        [4] Ask behaviour signs (multi-select)
        [5] Ask for optional droppings photo
                │
                ├── Photo sent → image_classifier.py ← 👁️ EYE
                │               MobileNetV2 → {class, confidence}
                │               High confidence (≥0.75) adjusts diagnosis
                │
                └── No photo → text answer used as-is (no degradation)
                        │
                        ▼
        [6] diagnosis_engine.py ← 🧠 BRAIN
            Checks: Gumboro · Newcastle · Coccidiosis · CRD · Fowl Pox
            Each check scores confidence from multiple signals
            Output: {disease, confidence, reasons[], urgency, drug, whatsapp_message}

        [7] Farmer receives diagnosis on WhatsApp:
            - Disease name
            - Confidence level (High / Medium / Low)
            - Numbered reasons why
            - Urgency level (Emergency / Urgent / Monitor)
            - Exact agrovet drug name
            - VSD contact if Newcastle (notifiable disease)
```

---

## Agentic Farm Monitor

KokoAlert runs an autonomous agent every morning at 7am Ghana time. The farmer does nothing — the system acts first.

**What the agent does daily:**
- Calculates every farmer's flock age
- Sends proactive Gumboro danger window alerts (weeks 3–6)
- Sends vaccination reminders ±2 days before each due date
- Sends daily check-in question (1–6 options)
- On Sundays: sends full 5-question weekly health check

**Key proactive alerts:**
- Day 20: "Tomorrow your birds hit 3 weeks — Gumboro window opens"
- Days 21–42: Daily droppings check during the danger window
- Week 10: Newcastle booster reminder
- Week 14: Stop coccidiosis medicine warning
- November 1st: Peak disease season alert (34.27% of annual cases)

---

## Vaccination Schedule

KokoAlert tracks the complete Ghana VSD vaccination schedule and sends automated reminders. The farmer never needs to remember dates.

| Day | Action |
|-----|--------|
| 1 | Glucose + poultry multivitamins in water |
| 7 | 1st Gumboro vaccine (Gumboro Intermediate/Plus) |
| 14 | 1st Newcastle vaccine (Lasota) |
| 21 | 2nd Gumboro vaccine |
| 28 | 2nd Newcastle vaccine |
| 1–105 (Mon) | Coccidiosis medicine 3 days/week |
| 70 | Newcastle booster (Week 10) |
| 84 | Fowl Pox vaccine (Week 12) |
| 98 | Stop coccidiosis medicine warning |
| 105 | STOP coccidiosis medicine (Week 15) |
| 112 | Final Newcastle injection — oil-based inactivated (Week 16) |

---

## Model Performance

| Model | Dataset | Key Metrics |
|-------|---------|-------------|
| Audio CNN Classifier | Bowen University Nigeria (260 clips) + Ghanaian farm audio | AUC 1.00 · Recall 99% · FPR 1.4% |
| Image Classifier (MobileNetV2) | Tanzanian PCR-verified dataset — Zenodo 5801834 | Accuracy 91% · NCD Recall >80% |
| Seasonal Risk Engine | KNUST VS Lab — 1,998 cases 2018–2021 | Rule-based — direct literature extraction |
| Diagnosis Engine | Ayim-Akonor et al. (2020) + Ghana VSD protocols | Rule-based — validated against Ghanaian vet research |

**Audio CNN technical details:**
- Architecture: Conv2D(32) → Conv2D(64) → Conv2D(128) → GAP → Dense(64) → sigmoid
- Input: (128, 157, 1) mel spectrograms — 5-second windows, FMIN=500Hz
- Decision threshold: 0.74 (FPR-optimal calibration)
- Any-Two rule: ≥2 anomalous windows flags the recording

---

## Data Sources

All data sources are Ghanaian or African. The seasonal risk and biosecurity models are grounded entirely in Ghanaian research.

| Source | Type | Ghanaian? | Used For |
|--------|------|-----------|----------|
| Mensah et al. (2023). PAMJ-One Health 12:11 | 1,998 cases, KNUST VS Lab, Kumasi | 🇬🇭 Yes | Seasonal risk weights |
| Ayim-Akonor et al. (2020). BMC Veterinary Research | 76 Ashanti Region farms | 🇬🇭 Yes | Biosecurity scoring rules |
| Ghana VSD Vaccination Schedule | Official government protocol | 🇬🇭 Yes | Vaccination tracker |
| Experienced poultry farmer, Ashanti Region | Primary field interview — validated against Ghana VSD official protocol and Mensah et al. (2023) | 🇬🇭 Yes | Vaccination schedule + disease management protocols |
| Poultry farm, Ashanti Region | Own-collected healthy flock audio | 🇬🇭 Yes | Ghanaian domain fine-tuning |
| Bowen University Nigeria — Mendeley Data | 260 poultry audio clips | 🌍 African (Nigeria) | Audio CNN pre-training |
| Zenodo record 5801834 | 4,536 PCR-verified droppings images, Tanzania | 🌍 African (Tanzania) | Image classifier training |

---

## Project Structure

```
kokoalert/
├── src/
│   ├── config.py                ← constants, vaccination schedule, disease metadata
│   ├── preprocess.py            ← audio → mel spectrogram pipeline
│   ├── disease_classifier.py    ← CNN binary classifier (healthy/sick)
│   ├── image_classifier.py      ← MobileNetV2 droppings classifier
│   ├── diagnosis_engine.py      ← multi-signal disease diagnosis brain
│   ├── vaccination_scheduler.py ← flock age tracking + vaccine reminders
│   ├── pipeline.py              ← orchestrates all ML modules
│   ├── risk_engine.py           ← farm risk scoring
│   └── biosecurity_scorer.py    ← biosecurity scoring (Ayim-Akonor et al.)
│
├── whatsapp_bot/
│   └── handlers.py              ← WhatsApp conversation state machine
│
├── api/
│   ├── main.py                  ← FastAPI app + APScheduler farm monitor agent
│   ├── database.py              ← SQLite layer
│   └── dashboard_routes.py      ← dashboard and analysis API endpoints
│
├── dashboard/
│   └── index.html               ← command center + live diagnosis + model/data
│
├── models/
│   ├── autoencoder.h5           ← trained audio CNN weights
│   ├── droppings_classifier.h5  ← trained MobileNetV2 weights
│   └── threshold.json           ← {threshold: 0.74}
│
├── tests/
│   └── test_system.py           ← 28 system tests, all passing
│
├── requirements.txt
└── Procfile
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web framework | FastAPI |
| ML framework | TensorFlow 2.15+ |
| Audio processing | librosa 0.10+ |
| Scheduler | APScheduler |
| Database | SQLite |
| Deployment | Render.com |
| WhatsApp | Meta WhatsApp Business Cloud API |
| Dashboard | Vanilla HTML/CSS/JS + Chart.js |

---

## Setup

### Prerequisites
- Python 3.10+
- WhatsApp Business API credentials

### Installation

```bash
git clone https://github.com/K-Opoku/kokoalert.git
cd kokoalert
pip install -r requirements.txt
```

### Environment Variables

```
WHATSAPP_API_TOKEN        = your_meta_api_token
WHATSAPP_PHONE_NUMBER_ID  = your_phone_number_id
WHATSAPP_VERIFY_TOKEN     = kokoalert2026
```

### Run locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Run tests

```bash
pytest tests/test_system.py -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/webhook` | WhatsApp message receiver |
| GET | `/webhook` | WhatsApp webhook verification |
| POST | `/api/analyze/audio` | Audio file → spectrogram + P(sick) |
| POST | `/api/analyze/full` | Audio + symptoms + optional image → full diagnosis |
| GET | `/api/dashboard/stats` | Aggregated farm stats |
| GET | `/dashboard` | Interactive dashboard |

Full API documentation: [https://kokoalert-1.onrender.com/docs](https://kokoalert-1.onrender.com/docs)

---

## WhatsApp Commands

Farmers interact entirely through numbered menus. No typing required beyond single-digit replies.

| Command | Action |
|---------|--------|
| Voice note | Triggers full audio analysis + diagnosis flow |
| `RISK` | Farm disease risk score with seasonal context |
| `VACC` | Full vaccination schedule with status |
| `BIOSEC` | Biosecurity score with improvement recommendations |
| `HELP` | Emergency VSD and vet contacts |
| `DOC` | Register new batch of day-old chicks |
| `RESET` | Clear stuck conversation state |

---

## Live Deployment

- **Dashboard:** [https://kokoalert-1.onrender.com/dashboard](https://kokoalert-1.onrender.com/dashboard)
- **API:** [https://kokoalert-1.onrender.com](https://kokoalert-1.onrender.com)
- **API Docs:** [https://kokoalert-1.onrender.com/docs](https://kokoalert-1.onrender.com/docs)

---

## Research Citations

- Mensah, N.K. et al. (2023). Retrospective analysis of respiratory diseases in poultry diagnosed at The Veterinary Services Laboratory, Kumasi, Ghana. *PAMJ-One Health*, 12:11. DOI: 10.11604/pamj-oh.2023.12.11.40752
- Ayim-Akonor, M. et al. (2020). Zoonotic disease exposure risk and biosecurity practices among poultry farmers in the Ashanti Region, Ghana. *BMC Veterinary Research*.
- Ouma, E.A. et al. (2023). Poultry health constraints in Northern Ghana. *Frontiers in Veterinary Science*.
- Tanzanian PCR-verified droppings dataset: Zenodo record 5801834. Nelson Mandela African Institution of Science and Technology.

---

## Team

**Kofi Konadu Opoku** — ML Engineering & System Architecture, KNUST  



---

*KokoAlert  | Ear + Eye + Brain + Agent*  
*Covers: Newcastle · Gumboro · Coccidiosis · CRD · Fowl Pox*  


