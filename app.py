import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Autonomous Car Diagnosis Robot System", page_icon="🤖", layout="wide")

def clean_and_score(raw_df):
    df = raw_df.copy()
    for col in ["max_safe_repair_cost", "estimated_repair_cost", "sensor_confidence", "anomaly_deviation_pct", "occurrence_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    df["occurrence_count"] = df.get("occurrence_count", pd.Series(0, index=df.index)).fillna(0)
    df = df.dropna(subset=["fault_name", "max_safe_repair_cost", "estimated_repair_cost", "sensor_confidence"]).copy()
    df = df[(df["max_safe_repair_cost"] > 0) & (df["estimated_repair_cost"] >= 0)].copy()
    if df["anomaly_deviation_pct"].dropna().max() <= 1.0:
        df["anomaly_deviation_pct"] = (df["anomaly_deviation_pct"] * 100).round(2)
    calc_d = ((df["max_safe_repair_cost"] - df["estimated_repair_cost"]) / df["max_safe_repair_cost"] * 100).round(2)
    df["anomaly_deviation_pct"] = df["anomaly_deviation_pct"].fillna(calc_d).round(2)
    df["preventive_savings"] = (df["max_safe_repair_cost"] - df["estimated_repair_cost"]).clip(lower=0).round(2)
    def mm(s):
        return pd.Series(0.0, index=s.index) if s.max() == s.min() else (s - s.min()) / (s.max() - s.min()) * 100
    df["anomaly_score"] = df["anomaly_deviation_pct"].clip(0, 100).round(2)
    df["savings_score"] = mm(df["preventive_savings"]).round(2)
    df["confidence_score"] = (df["sensor_confidence"].clip(0, 5) / 5 * 100).round(2)
    df["frequency_score"] = mm(df["occurrence_count"].clip(lower=0)).round(2)
    df["priority_score"] = (0.30 * df["anomaly_score"] + 0.25 * df["savings_score"] + 0.25 * df["confidence_score"] + 0.20 * df["frequency_score"]).round(2)
    df["severity_category"] = pd.cut(df["priority_score"], bins=[-0.01, 39.999, 59.999, 74.999, 100], labels=["Low", "Moderate", "High", "Critical"]).astype(str)
    return df

@st.cache_data
def load_default_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base_dir, "dashboard_data.csv")
    if os.path.exists(p):
        return clean_and_score(pd.read_csv(p))
    return pd.DataFrame()

st.markdown("### 🤖 Autonomous Car Diagnosis Robot System  `Live Portfolio App`")
st.caption("Multi-Factor OBD-II Fault Severity Scoring, Preventive Repair Cost Comparison & Smart Maintenance Triage")

tab1, tab2, tab3 = st.tabs(["📊 Live Robot Telemetry Dashboard", "🧮 Diagnostic Scoring Formula", "🚀 Permanent Link & Portfolio Guide"])

with tab1:
    st.sidebar.header("📂 Telemetry Dataset Source")
    up_file = st.sidebar.file_uploader("Upload OBD-II Robot CSV File", type=["csv"])
    df = clean_and_score(pd.read_csv(up_file)) if up_file is not None else load_default_data()

    st.sidebar.header("🔍 Subsystem & Fault Filters")
    if "kw" not in st.session_state:
        st.session_state.kw = "Engine"
    c_q1, c_q2, c_q3, c_q4, c_q5 = st.sidebar.columns(5)
    if c_q1.button("Engine"): st.session_state.kw = "Engine"
    if c_q2.button("Brakes"): st.session_state.kw = "Brakes"
    if c_q3.button("Battery"): st.session_state.kw = "Battery"
    if c_q4.button("Cooling"): st.session_state.kw = "Cooling"
    if c_q5.button("All"): st.session_state.kw = ""

    keyword = st.sidebar.text_input("Search DTC code or subsystem keyword", value=st.session_state.kw)
    max_p = int(df["estimated_repair_cost"].max()) if not df.empty else 75000
    default_b = 15000 if st.session_state.kw == "Engine" else max_p
    max_budget = st.sidebar.slider("Max Immediate Repair Budget (₹)", min_value=1000, max_value=max(20000, max_p), value=min(default_b, max(20000, max_p)), step=500)
    min_confidence = st.sidebar.slider("Min Robot Sensor Confidence (★)", 0.0, 5.0, 4.0, 0.1)
    cat_tier = st.sidebar.selectbox("Fault Severity Tier", ["ALL", "Critical", "High", "Moderate", "Low"])
    sort_by = st.sidebar.selectbox("Sort Diagnostic Triage By", ["Highest Priority Score First (Default)", "Estimated Repair Cost: Low to High", "Highest Anomaly Deviation % First", "Highest Preventive Savings (₹)", "Highest Sensor Confidence (★)"])

    filtered = df[(df["estimated_repair_cost"] <= max_budget) & (df["sensor_confidence"] >= min_confidence)].copy()
    if cat_tier != "ALL":
        filtered = filtered[filtered["severity_category"] == cat_tier]
    if keyword.strip():
        filtered = filtered[filtered["fault_name"].astype(str).str.contains(keyword.strip(), case=False, na=False)]
    filtered = filtered.drop_duplicates(subset=["fault_name"])

    if sort_by == "Estimated Repair Cost: Low to High": filtered = filtered.sort_values("estimated_repair_cost", ascending=True)
    elif sort_by == "Highest Anomaly Deviation % First": filtered = filtered.sort_values("anomaly_deviation_pct", ascending=False)
    elif sort_by == "Highest Preventive Savings (₹)": filtered = filtered.sort_values("preventive_savings", ascending=False)
    elif sort_by == "Highest Sensor Confidence (★)": filtered = filtered.sort_values(["sensor_confidence", "priority_score"], ascending=[False, False])
    else: filtered = filtered.sort_values(["priority_score", "sensor_confidence"], ascending=[False, False])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Detected Faults", f"{len(filtered):,}", f"Out of {len(df)} diagnostic codes")
    k2.metric("Avg Anomaly Deviation", f"{filtered['anomaly_deviation_pct'].mean():.1f}%" if not filtered.empty else "—", "Across filtered subsystems")
    k3.metric("Peak Priority Score", f"{filtered['priority_score'].max():.2f}" if not filtered.empty else "—", "Weighted 0-100 urgency scale")
    k4.metric("Max Preventive Savings", f"₹{filtered['preventive_savings'].max():,.0f}" if not filtered.empty else "—", "Avoided breakdown cost")

    if filtered.empty:
        st.warning("🔍 No matching diagnostic faults found. Try clearing the subsystem keyword filter or raising your repair budget (₹).")
    else:
        best = filtered.sort_values(["priority_score", "sensor_confidence"], ascending=[False, False]).iloc[0]
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1e1b4b, #312e81, #0f172a); color: white; padding: 20px; border-radius: 16px; margin-bottom: 20px;">
            <span style="background: rgba(251,191,36,0.2); color: #fde68a; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;">🚨 Top Priority Robot Diagnostic Action</span>
            <h3 style="color: white; margin-top: 10px; margin-bottom: 6px;">{best["fault_name"]}</h3>
            <p style="font-size: 14px; color: #e0e7ff; margin-bottom: 12px;">
                Early Repair Cost: <b style="color:white;">₹{best["estimated_repair_cost"]:,.0f}</b> <del style="color:#a5b4fc;">₹{best["max_safe_repair_cost"]:,.0f} (Breakdown Cost)</del> &nbsp;|&nbsp; 
                Preventive Savings: <b style="color:#6ee7b7;">₹{best["preventive_savings"]:,.0f} ({best["anomaly_deviation_pct"]:.1f}% Anomaly)</b> &nbsp;|&nbsp; 
                Sensor Confidence: <b style="color:#fcd34d;">★ {best["sensor_confidence"]:.1f} / 5 ({int(best["occurrence_count"]):,} logs)</b> &nbsp;|&nbsp; 
                Priority Score: <b style="color:#34d399; font-size:18px;">{best["priority_score"]:.2f} ({best["severity_category"]})</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.progress(min(100, int(best["anomaly_score"])), text=f"Anomaly (30%): {best['anomaly_score']:.0f}/100")
        b2.progress(min(100, int(best["savings_score"])), text=f"Savings (25%): {best['savings_score']:.0f}/100")
        b3.progress(min(100, int(best["confidence_score"])), text=f"Confidence (25%): {best['confidence_score']:.0f}/100")
        b4.progress(min(100, int(best["frequency_score"])), text=f"Frequency (20%): {best['frequency_score']:.0f}/100")

        st.subheader("📋 Robot Diagnostic Triage & Repair Recommendations")
        cols = ["fault_name", "max_safe_repair_cost", "estimated_repair_cost", "preventive_savings", "anomaly_deviation_pct", "sensor_confidence", "priority_score", "severity_category"]
        st.dataframe(filtered[cols], hide_index=True)

        st.subheader("📊 Visual Telemetry Analytics")
        ch1, ch2, ch3 = st.columns(3)
        order = ["Critical", "High", "Moderate", "Low"]
        cmap = {"Critical": "#ef4444", "High": "#f59e0b", "Moderate": "#3b82f6", "Low": "#10b981"}
        with ch1:
            cat_counts = filtered["severity_category"].value_counts().reindex(order, fill_value=0).reset_index()
            cat_counts.columns = ["Severity Category", "Faults"]
            fig1 = px.bar(cat_counts, x="Severity Category", y="Faults", color="Severity Category", color_discrete_map=cmap, title="Fault Severity Distribution")
            st.plotly_chart(fig1, key="c1")
        with ch2:
            fig2 = px.histogram(filtered, x="anomaly_deviation_pct", nbins=10, title="Distribution of Sensor Anomaly Deviations (%)")
            st.plotly_chart(fig2, key="c2")
        with ch3:
            fig3 = px.scatter(filtered, x="sensor_confidence", y="priority_score", color="severity_category", color_discrete_map=cmap, hover_data=["fault_name"], title="Sensor Confidence vs. Priority Score")
            st.plotly_chart(fig3, key="c3")

with tab2:
    st.subheader("Multi-Factor Diagnostic Priority Scoring & Telemetry Cleaning Pipeline")
    st.info("Priority Score = 0.30 × Anomaly Score + 0.25 × MinMax(Preventive Savings) + 0.25 × (Sensor Confidence/5 × 100) + 0.20 × MinMax(Fault Frequency)")
    st.markdown("- **Critical Severity:** Score ≥ 75\n- **High Severity:** Score 60 – 74.99\n- **Moderate Severity:** Score 40 – 59.99\n- **Low Severity:** Score < 40")

with tab3:
    st.subheader("🚀 Permanent Hosting & Resume Bullets")
    st.code("""Autonomous Car Diagnosis Robot & Predictive Maintenance System | Python, Pandas, Plotly, Streamlit
• Engineered an automated OBD-II telemetry cleaning pipeline in Python/Pandas to parse raw sensor logs, strip formatting artifacts, and compute preventive repair cost savings across vehicle subsystems.
• Designed a 4-factor weighted Diagnostic Priority Scoring algorithm (30% Sensor Anomaly Deviation, 25% Min-Max Preventive Cost Savings, 25% Normalized Robot Sensor Confidence, 20% Fault Frequency) to classify faults into 4 severity tiers.
• Built and deployed an interactive Streamlit diagnostic dashboard enabling real-time DTC/subsystem search, repair budget filtering, deduplicated fault triage ranking, and visual telemetry analytics.""", language="text")