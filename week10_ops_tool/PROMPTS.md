# PROMPTS.md — Vibe Coding Workflow
**Tool:** Equipment Failure Risk Dashboard  
**AI Assistant used:** Claude (Anthropic)  
**Week:** 10 — AI-Assisted Development

---

## What is Vibe Coding?

Vibe Coding is the practice of using AI assistants to generate, debug, and refine code by writing precise prompts — instead of writing every line of syntax manually. The developer's role shifts from writing code to **defining requirements, reviewing outputs, and iterating on logic**.

---

## Prompt 1 — Architecture Definition

**Goal:** Get Claude to plan the full app structure before writing any code.

**Prompt sent:**
```
Act as a Senior Python Engineer specialising in Streamlit production dashboards.

I have a trained XGBoost equipment failure detection model (Week 9 notebook).
The dataset is synthetic_predictive_maintenance_data.csv with these columns:
vibration, process_temp, torque, rotational_speed, tool_wear, pressure, noise, cycle, failure.

I also engineered 4 features in training:
- vib_torque_stress = vibration * torque
- wear_cycle_degradation = tool_wear * cycle
- temp_speed_ratio = process_temp / rotational_speed
- pressure_noise_ratio = pressure / noise

Build me a Streamlit dashboard where:
1. Operators enter live sensor readings via sliders in a sidebar
2. The model runs instantly and shows a failure risk probability
3. A colour-coded badge shows LOW / MEDIUM / HIGH risk with a recommended action
4. SHAP values are computed and displayed as a bar chart showing which sensor is driving the alert
5. A plain-English explanation tells the operator WHY the asset was flagged
6. No API keys or secrets are hardcoded — use .env files

Use an industrial dark theme. The audience is field engineers, not data scientists.
Generate the complete app.py.
```

**What Claude generated:** Full app.py with sidebar sliders, live probability, risk badge, SHAP bar chart, and plain-English driver summary.

**What I reviewed and changed:**
- Added auto-trigger: SHAP explanation now also fires automatically when risk ≥ 65% (not just on button click)
- Added secondary driver display below the primary SHAP explanation
- Refined the `driver_labels` dict to use plain language for each feature name
- Added the sensor reading expander table at the bottom for transparency

---

## Prompt 2 — Feature Engineering Consistency

**Goal:** Make sure the dashboard applies the exact same feature engineering as the Week 9 training pipeline, so predictions are valid.

**Prompt sent:**
```
The model was trained with these 4 engineered features added AFTER the raw sensors.
Make sure the Streamlit app applies the exact same transformations to the user inputs
before calling model.predict_proba(). Show me the function that does this.
```

**What Claude generated:** The `engineer_features()` function that mirrors the notebook pipeline exactly.

**What I reviewed:** Confirmed column order matches `FEATURE_COLS` list used during training. Added the `+ 1e-6` guard on division features to prevent divide-by-zero on edge inputs.

---

## Prompt 3 — Model Caching

**Goal:** Prevent the model from retraining on every Streamlit interaction.

**Prompt sent:**
```
In Streamlit, every slider interaction reruns the entire script.
Wrap the model training in an @st.cache_resource block so it only trains once per session.
The function should load the CSV, apply SMOTE, train XGBoost, and return the model
and SHAP TreeExplainer together.
```

**What Claude generated:** The `load_model()` function with `@st.cache_resource` decorator.

**What I reviewed:** Added the `DATA_PATH` environment variable lookup via `os.getenv()` so the CSV path is configurable without touching code.

---

## Prompt 4 — Industrial Dark Theme

**Goal:** Replace the default Streamlit white theme with something appropriate for a field engineering tool.

**Prompt sent:**
```
The dashboard is used by field engineers on industrial sites — often on laptops in
low-light environments. Replace the default Streamlit styling with:
- Dark background (#0D1117) matching GitHub's dark mode
- Sidebar in a slightly lighter dark (#161B22)
- Risk badge: red background for HIGH, amber for MEDIUM, green for LOW
- Monospace font for headings (IBM Plex Mono or similar)
- No rounded card shadows or SaaS-style decorations
Write the CSS as a single st.markdown() block.
```

**What Claude generated:** The full CSS block in `app.py`.

**What I reviewed:** Removed the default Streamlit ALL-CAPS label treatment. Added `letter-spacing` adjustments so monospace headings don't feel cramped.

---

## Prompt 5 — Security Review

**Goal:** Make sure no secrets or paths are hardcoded.

**Prompt sent:**
```
Review this app.py for security issues:
1. Are any file paths, API keys, or passwords hardcoded?
2. Should I add input validation on the sliders?
3. What should go in .gitignore?
List any issues and fix them.
```

**What Claude generated:** 
- Confirmed sliders are bounded (0.0–1.0) so no injection risk
- Added `os.getenv("DATA_PATH", "synthetic_predictive_maintenance_data.csv")` for the CSV path
- Generated `.env.example` and `.gitignore` content

---

## Prompt 6 — Debugging: SHAP shape mismatch

**Goal:** Fix a runtime error that appeared during testing.

**Error encountered:**
```
ValueError: shap_values has shape (1, 12) but X_input has shape (1, 12) — indexing error on [0]
```

**Prompt sent:**
```
I'm getting a shape mismatch when indexing shap_values. The explainer returns a 2D array
for a single-row input. How should I index it correctly for a binary XGBoost classifier?
```

**What Claude explained:** For XGBoost binary classifiers, `shap_values` returns shape `(n_samples, n_features)` directly — index as `shap_vals[0]` not `shap_vals[0][1]`.

**Fix applied:** Changed `sv = shap_vals[0][1]` to `sv = shap_vals[0]` in the SHAP section.

---

## Key Vibe Coding Lessons Learned

| Lesson | Detail |
|--------|--------|
| **Be specific about data schema** | Naming every column in the prompt prevents Claude from guessing wrong feature names |
| **State the audience** | "field engineers, not data scientists" changed the entire tone of labels and explanations |
| **Ask for security review separately** | Claude doesn't flag hardcoded paths unless you explicitly ask |
| **Iterate on one feature at a time** | Asking for the full app in one prompt worked, but refinements (caching, theming, SHAP fix) each needed their own focused prompt |
| **Review before trusting** | The SHAP indexing bug would have silently returned wrong values — always test edge cases manually |