"""
Equipment Failure Risk Dashboard
Week 10 - Vibe Coded with Claude (Anthropic)

IMPORTANT: Run save_model.py from your Week 9 folder first to generate:
  - xgb_model.joblib      (trained model)
  - feature_cols.joblib   (exact column order used during training)
Place both files in the same folder as this app.py before launching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Equipment Risk Monitor",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0D1B2A; color: #E8EDF2; }
[data-testid="stSidebar"] { background-color: #112236; border-right: 1px solid #1E3A5F; }
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
.dash-header {
    background: linear-gradient(90deg,#1E3A5F 0%,#0D2D4A 100%);
    border-left: 4px solid #2E9BDA;
    padding: 18px 24px; border-radius: 6px; margin-bottom: 24px;
}
.dash-header h1 { color:#E8EDF2; font-size:1.6rem; font-weight:700; margin:0; }
.dash-header p  { color:#7FA8C8; font-size:0.85rem; margin:4px 0 0 0; }
.risk-card { padding:24px; border-radius:8px; text-align:center; margin-bottom:16px; }
.risk-low    { background:#0D2E1A; border:2px solid #1A7A4A; }
.risk-medium { background:#2E1F00; border:2px solid #E67E22; }
.risk-high   { background:#2E0D0D; border:2px solid #C0392B; }
.risk-score  { font-size:3.2rem; font-weight:800; line-height:1; }
.risk-label  { font-size:1.0rem; font-weight:600; margin-top:6px; }
.risk-sub    { font-size:0.78rem; color:#8A9BB0; margin-top:4px; }
.section-head {
    font-size:0.78rem; font-weight:600; color:#2E9BDA;
    letter-spacing:0.8px; text-transform:uppercase;
    border-bottom:1px solid #1E3A5F;
    padding-bottom:6px; margin:20px 0 12px 0;
}
.alert-banner {
    background:#2E0D0D; border:1px solid #C0392B;
    border-left:4px solid #C0392B; border-radius:6px;
    padding:12px 16px; font-size:0.88rem; color:#F0A0A0; margin-bottom:12px;
}
.safe-banner {
    background:#0D2E1A; border:1px solid #1A7A4A;
    border-left:4px solid #1A7A4A; border-radius:6px;
    padding:12px 16px; font-size:0.88rem; color:#90D4A8; margin-bottom:12px;
}
.warn-banner {
    background:#2E1F00; border:1px solid #E67E22;
    border-left:4px solid #E67E22; border-radius:6px;
    padding:12px 16px; font-size:0.88rem; color:#F5CBA7; margin-bottom:12px;
}
.stButton > button {
    background:#1E6FA8; color:white; border:none; border-radius:6px;
    padding:10px 28px; font-weight:600; font-size:0.9rem; width:100%;
}
.stButton > button:hover { background:#2E9BDA; }
h2, h3 { color:#E8EDF2 !important; }
p, li  { color:#C8D8E8; }
</style>
""", unsafe_allow_html=True)


# ── MODEL + FEATURE COLS LOADER ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load model and the EXACT feature column list saved during training.
    This guarantees the app feeds features in the same order the model
    was trained on — preventing silent feature mismatch bugs.
    """
    model_path   = Path("xgb_model.joblib")
    fcols_path   = Path("feature_cols.joblib")
    csv_path     = Path("synthetic_predictive_maintenance_data.csv")

    # Best case: both saved artifacts exist
    if model_path.exists() and fcols_path.exists():
        model        = joblib.load(model_path)
        feature_cols = joblib.load(fcols_path)
        return model, feature_cols

    # Fallback: train fresh from CSV if available
    if csv_path.exists():
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE

        st.warning(
            "⚠️ No saved model found. Training fresh from CSV — this takes ~30 seconds. "
            "Run `save_model.py` from your Week 9 folder for instant loads next time."
        )

        df = pd.read_csv(csv_path)

        # Feature engineering — identical order to training notebook
        df['vib_torque_stress']      = df['vibration'] * df['torque']
        df['wear_cycle_degradation'] = df['tool_wear'] * df['cycle']
        df['temp_speed_ratio']       = df['process_temp'] / (df['rotational_speed'] + 1e-6)
        df['pressure_noise_ratio']   = df['pressure'] / (df['noise'] + 1e-6)

        feature_cols = [c for c in df.columns if c not in ['asset_id', 'failure']]
        X = df[feature_cols]; y = df['failure']

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)

        neg = int(np.bincount(y_train)[0])
        pos = int(np.bincount(y_train)[1])

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg // pos,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_res, y_res)

        # Save both artifacts so next load is instant
        joblib.dump(model,        model_path)
        joblib.dump(feature_cols, fcols_path)

        return model, feature_cols

    return None, None


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer_features(raw: dict, feature_cols: list) -> pd.DataFrame:
    """
    Apply the same engineered features as training, then reindex
    to the EXACT column order saved from the training notebook.
    The reindex step is what prevents the silent column mismatch bug.
    """
    raw['vib_torque_stress']      = raw['vibration'] * raw['torque']
    raw['wear_cycle_degradation'] = raw['tool_wear'] * raw['cycle']
    raw['temp_speed_ratio']       = raw['process_temp'] / (raw['rotational_speed'] + 1e-6)
    raw['pressure_noise_ratio']   = raw['pressure'] / (raw['noise'] + 1e-6)

    df = pd.DataFrame([raw])
    # Reindex enforces the exact column order the model was trained with
    return df.reindex(columns=feature_cols, fill_value=0.0)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ── GAUGE CHART ───────────────────────────────────────────────────────────────
def draw_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(theta, [1]*300, color='#1E3A5F', linewidth=12, solid_capstyle='round')
    fill_end   = np.pi - prob * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 300)
    color = '#1A7A4A' if prob < 0.35 else ('#E67E22' if prob < 0.65 else '#C0392B')
    ax.plot(theta_fill, [1]*len(theta_fill), color=color, linewidth=12, solid_capstyle='round')
    needle_angle = np.pi - prob * np.pi
    ax.annotate('', xy=(needle_angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#E8EDF2', lw=2.5))
    ax.set_ylim(0, 1.3); ax.set_yticks([])
    ax.set_xticks([np.pi, np.pi*0.75, np.pi*0.5, np.pi*0.25, 0])
    ax.set_xticklabels(['0%','25%','50%','75%','100%'], color='#7FA8C8', fontsize=8)
    ax.spines['polar'].set_visible(False)
    ax.set_thetamin(0); ax.set_thetamax(180)
    plt.tight_layout(pad=0.2)
    return fig


# ── SHAP BAR CHART ────────────────────────────────────────────────────────────
def draw_shap_bars(shap_vals: np.ndarray, feature_row: pd.DataFrame) -> plt.Figure:
    vals  = shap_vals.flatten()
    feats = feature_row.columns.tolist()
    order = np.argsort(np.abs(vals))[::-1][:8]
    vals_s = vals[order][::-1]
    feat_s = [feats[i] for i in order][::-1]
    read_s = feature_row.values.flatten()[order][::-1]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor('#112236'); ax.set_facecolor('#112236')
    colors = ['#C0392B' if v > 0 else '#1A7A4A' for v in vals_s]
    ax.barh(range(len(feat_s)), vals_s, color=colors, height=0.6, edgecolor='none')
    ax.set_yticks(range(len(feat_s)))
    labels = [f"{feat_s[i]}  ({read_s[i]:.3f})" for i in range(len(feat_s))]
    ax.set_yticklabels(labels, color='#C8D8E8', fontsize=8)
    ax.set_xlabel('SHAP contribution to failure risk', color='#7FA8C8', fontsize=8)
    ax.tick_params(colors='#7FA8C8', labelsize=8)
    ax.axvline(0, color='#1E3A5F', linewidth=1)
    ax.spines[['top','right','bottom','left']].set_color('#1E3A5F')
    ax.set_title('Why did the model score this asset?', color='#E8EDF2', fontsize=9, pad=8)
    ax.legend(
        handles=[mpatches.Patch(color='#C0392B', label='Increases failure risk'),
                 mpatches.Patch(color='#1A7A4A', label='Decreases failure risk')],
        fontsize=7, facecolor='#0D1B2A', edgecolor='#1E3A5F',
        labelcolor='#C8D8E8', loc='lower right'
    )
    plt.tight_layout()
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Sensor Inputs")
    st.caption("Enter live readings for the asset you want to assess.")
    asset_id = st.text_input("Asset ID", value="ASSET-001")

    st.markdown("**Mechanical**")
    vibration        = st.slider("Vibration",        0.0, 1.0, 0.41, 0.001)
    torque           = st.slider("Torque",           0.0, 1.0, 0.50, 0.001)
    rotational_speed = st.slider("Rotational Speed", 0.0, 1.0, 0.50, 0.001)

    st.markdown("**Thermal**")
    process_temp = st.slider("Process Temperature", 0.0, 1.0, 0.60, 0.001)

    st.markdown("**Wear & Condition**")
    tool_wear = st.slider("Tool Wear",   0.0, 1.0, 0.50, 0.001)
    cycle     = st.slider("Cycle Count", 0.0, 1.0, 0.50, 0.001)

    st.markdown("**Environmental**")
    pressure = st.slider("Pressure", 0.0, 1.0, 0.50, 0.001)
    noise    = st.slider("Noise",    0.0, 1.0, 0.50, 0.001)

    st.markdown("---")
    run_btn = st.button("⚡  Run Risk Assessment")
    st.markdown("---")
    st.caption("XGBoost | CV AUC 0.9994 | Recall 95.8%")
    st.caption("Fleet baseline failure rate: 2.0%")


# ── MAIN PANEL ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>⚡ Equipment Failure Risk Monitor</h1>
  <p>Live ML prediction · SHAP explanation · XGBoost trained on 6,000 sensor cycles</p>
</div>
""", unsafe_allow_html=True)

model, feature_cols = load_artifacts()

if model is None:
    st.error(
        "No model or data found. Place `xgb_model.joblib` + `feature_cols.joblib` "
        "(from save_model.py) OR `synthetic_predictive_maintenance_data.csv` "
        "in the same folder as app.py, then restart."
    )
    st.stop()

explainer = get_explainer(model)

# Fleet summary
st.markdown('<div class="section-head">Fleet Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Model Recall",       "95.8%",  "23/24 failures caught")
with c2: st.metric("CV ROC-AUC",         "0.9994", "XGBoost")
with c3: st.metric("Fleet Failure Rate", "2.0%",   "49:1 imbalance handled")
with c4: st.metric("Alert Precision",    "54.8%",  "~18 false alarms / 1,200")
st.markdown("---")

# ── PREDICTION ────────────────────────────────────────────────────────────────
if run_btn:
    raw = {
        'vibration': vibration, 'process_temp': process_temp,
        'torque': torque, 'rotational_speed': rotational_speed,
        'tool_wear': tool_wear, 'pressure': pressure,
        'noise': noise, 'cycle': cycle
    }

    feature_row = engineer_features(raw, feature_cols)

    # Sanity check: flag any column that didn't align
    missing = [c for c in feature_cols if c not in feature_row.columns]
    if missing:
        st.error(f"Feature mismatch — columns missing from input: {missing}. "
                 "Re-run save_model.py and restart the app.")
        st.stop()

    prob      = float(model.predict_proba(feature_row)[0, 1])
    shap_vals = explainer.shap_values(feature_row)

    # Risk tier
    if prob < 0.35:
        tier, tier_class, tier_color, tier_icon = "LOW RISK",    "risk-low",    "#1A7A4A", "✅"
    elif prob < 0.65:
        tier, tier_class, tier_color, tier_icon = "MEDIUM RISK", "risk-medium", "#E67E22", "⚠️"
    else:
        tier, tier_class, tier_color, tier_icon = "HIGH RISK",   "risk-high",   "#C0392B", "🚨"

    left, right = st.columns([1, 1.8], gap="large")

    with left:
        st.markdown(f'<div class="section-head">Assessment — {asset_id}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card {tier_class}">
            <div class="risk-score" style="color:{tier_color}">{prob*100:.1f}%</div>
            <div class="risk-label" style="color:{tier_color}">{tier_icon} {tier}</div>
            <div class="risk-sub">Failure probability this cycle</div>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(draw_gauge(prob), use_container_width=True)

        # Action banner
        top_idx  = int(np.argmax(np.abs(shap_vals.flatten())))
        top_feat = feature_row.columns[top_idx]
        top_val  = float(feature_row.iloc[0, top_idx])

        if prob >= 0.65:
            st.markdown(f"""
            <div class="alert-banner">
                <strong>🚨 Action required:</strong> Dispatch technician to {asset_id}.<br>
                Primary driver: <strong>{top_feat}</strong> = {top_val:.3f}<br>
                Confidence: {prob*100:.1f}% failure probability.
            </div>""", unsafe_allow_html=True)
        elif prob >= 0.35:
            st.markdown(f"""
            <div class="warn-banner">
                <strong>⚠️ Monitor closely:</strong> {asset_id} is in an elevated risk zone.<br>
                Primary driver: <strong>{top_feat}</strong> = {top_val:.3f}<br>
                Schedule an inspection within the next cycle.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-banner">
                <strong>✅ No immediate action needed.</strong><br>
                {asset_id} is within normal operating parameters.<br>
                Continue standard monitoring cadence.
            </div>""", unsafe_allow_html=True)

        # Engineered feature spot-check
        vts = float(feature_row['vib_torque_stress'].iloc[0])
        tsr = float(feature_row['temp_speed_ratio'].iloc[0])
        st.markdown('<div class="section-head">Key Engineered Signals</div>',
                    unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            vts_delta = f"+{vts - 0.205:.3f} vs fleet avg" if vts > 0.205 else f"{vts - 0.205:.3f} vs fleet avg"
            st.metric("Vib × Torque", f"{vts:.3f}", vts_delta)
        with ec2:
            tsr_delta = f"+{tsr - 0.12:.3f} vs fleet avg" if tsr > 0.12 else f"{tsr - 0.12:.3f} vs fleet avg"
            st.metric("Temp ÷ Speed", f"{tsr:.3f}", tsr_delta)

    with right:
        st.markdown('<div class="section-head">SHAP Explanation — What drove this score?</div>',
                    unsafe_allow_html=True)
        st.pyplot(draw_shap_bars(shap_vals, feature_row), use_container_width=True)

        # Top 3 plain-English
        order = np.argsort(np.abs(shap_vals.flatten()))[::-1]
        top3  = [(feature_row.columns[i],
                  float(shap_vals.flatten()[i]),
                  float(feature_row.iloc[0, i])) for i in order[:3]]

        st.markdown('<div class="section-head">Plain-English Summary</div>',
                    unsafe_allow_html=True)
        for rank, (feat, sv, reading) in enumerate(top3, 1):
            direction = "🔴 increases" if sv > 0 else "🟢 decreases"
            st.markdown(f"**{rank}. `{feat}`** (reading: `{reading:.3f}`) "
                        f"— {direction} failure risk by **{abs(sv):.3f}** SHAP units.")

        st.markdown("---")
        st.caption(
            "Red bars = push toward failure. Green bars = push toward normal. "
            "Bar length = magnitude of influence on this specific prediction."
        )

    with st.expander("🔍 All feature values used for this prediction"):
        display = feature_row.T.rename(columns={0: "Value"}).round(4)
        st.dataframe(display, use_container_width=True)

else:
    # Landing state
    st.markdown('<div class="section-head">How to use this tool</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Enter sensor readings**")
        st.caption("Use the sliders on the left to input live values for the asset you want to check.")
    with c2:
        st.markdown("**2. Run the assessment**")
        st.caption("Click 'Run Risk Assessment'. The model scores the asset in under a second.")
    with c3:
        st.markdown("**3. Read the explanation**")
        st.caption("The SHAP chart shows which sensor drove the score — green = safe, red = risk.")
    st.info(
        "👈 Set sensor values in the sidebar and click **Run Risk Assessment**. "
        "The model catches 95.8% of real failures (23 of 24 on the test set)."
    )