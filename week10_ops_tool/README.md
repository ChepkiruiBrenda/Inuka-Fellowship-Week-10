# ⚡ Equipment Failure Risk Monitor
**Week 10 — AI-Powered Operational Tool**  
Built with Streamlit + XGBoost + SHAP · Vibe-coded using Claude (Anthropic)

---

## What it does

Field engineers enter live sensor readings into the dashboard. The model returns an instant failure risk score and a plain-English explanation of which sensor is driving the alert — so technicians know exactly where to look.

- **Risk score** — LOW / MEDIUM / HIGH with recommended action
- **SHAP bar chart** — shows which of the 12 features is pushing the prediction
- **Plain-English explanation** — no data science jargon
- **Auto-trigger** — SHAP explanation fires automatically when risk exceeds 65%

## Model performance (Week 9 results)

| Metric | Score |
|--------|-------|
| Recall (Failure class) | 95.8% — caught 23 of 24 failures |
| ROC-AUC | 0.9961 |
| CV AUC | 0.9994 ± 0.0001 |
| False alarms per 1,200 records | ~18–19 |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/[your-username]/Inuka-Fellowship-Week-9.git
cd week10_ops_tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
# Copy synthetic_predictive_maintenance_data.csv into this folder
# (or set DATA_PATH in .env to its full path)

# 4. Run
streamlit run app.py
```

## File structure

```
week10_ops_tool/
├── app.py              # Main Streamlit dashboard
├── requirements.txt    # Python dependencies
├── PROMPTS.md          # Vibe coding workflow — all prompts used
├── save_model.py       # Training model script
├── .gitignore          # files not committed
└── README.md
```

## Security

- No secrets hardcoded anywhere
- All paths and credentials loaded from `.env` via `python-dotenv`
- Sliders bounded 0.0–1.0 — no raw text input that could be injected
- `.env` excluded from version control via `.gitignore`

## How it was built

See [PROMPTS.md](PROMPTS.md) for the full vibe coding workflow — every prompt sent to Claude, what was generated, and what was reviewed and changed.
