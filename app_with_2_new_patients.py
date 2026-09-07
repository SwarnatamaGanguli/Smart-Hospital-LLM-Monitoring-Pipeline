"""
Smart Hospital LLM - Streamlit Dashboard
-----------------------------------------
Connects the existing project components:
    synthetic vitals CSV -> rule-based detector -> patient context -> Ollama LLM

This UI is a prototype for research/demo use. It does not replace clinical
judgment or a validated hospital monitoring system.
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from vitals_detector import VitalsDetector
from patient_context import PatientContext, PatientRegistry


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Smart Hospital LLM",
    page_icon="🏥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Patient registry
# ---------------------------------------------------------------------------

def build_registry() -> PatientRegistry:
    registry = PatientRegistry()
    registry.add(
        PatientContext(
            patient_id="P001",
            age=67,
            sex="male",
            admitting_diagnosis="COPD exacerbation",
            known_conditions=["COPD", "hypertension"],
            notes="Baseline SpO2 runs slightly lower than typical due to COPD.",
        )
    )
    registry.add(
        PatientContext(
            patient_id="P002",
            age=74,
            sex="female",
            admitting_diagnosis="Suspected arrhythmia, telemetry monitoring",
            known_conditions=["atrial fibrillation (history)", "type 2 diabetes"],
            notes="Prone to sudden rate swings; prior episode of symptomatic bradycardia.",
        )
    )
    registry.add(
        PatientContext(
            patient_id="P003",
            age=52,
            sex="male",
            admitting_diagnosis="Post-operative recovery, day 1 (abdominal surgery)",
            known_conditions=["on opioid-based pain management"],
            notes="Watch for opioid-related respiratory depression - slow, sustained desaturation.",
        )
    )
    return registry


PATIENT_LABELS = {
    "P001": "P001 - COPD exacerbation",
    "P002": "P002 - Arrhythmia monitoring",
    "P003": "P003 - Post-op recovery",
}


# ---------------------------------------------------------------------------
# Cached detection: deterministic and independent of Ollama
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_vitals_and_events(csv_path: str):
    df = pd.read_csv(csv_path)
    detector = VitalsDetector()
    events = detector.run_on_csv(csv_path)
    return df, events


# ---------------------------------------------------------------------------
# Optional LLM reasoning
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_reasoning_layer(model: str):
    # Imported lazily so the dashboard can still show the monitoring layer
    # when LangChain/Ollama is not installed or Ollama is unavailable.
    from reasoning_layer import ReasoningLayer

    return ReasoningLayer(model=model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def severity_icon(severity: str) -> str:
    return {
        "critical": "🔴",
        "warning": "🟡",
        "info": "🔵",
    }.get(severity, "⚪")


def urgency_icon(urgency: str) -> str:
    return {
        "urgent": "🔴",
        "review_soon": "🟡",
        "routine": "🟢",
    }.get(urgency, "⚪")


def event_to_row(event):
    return {
        "Severity": f"{severity_icon(event.severity)} {event.severity.upper()}",
        "Event": event.kind,
        "Start": event.start_time,
        "End": event.end_time,
        "Description": event.description,
        "Detections merged": event.merge_count,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

registry = build_registry()

st.sidebar.title("🏥 Smart Hospital LLM")
st.sidebar.caption("Research prototype")

selected_patient_id = st.sidebar.selectbox(
    "Patient",
    options=list(PATIENT_LABELS.keys()),
    format_func=lambda pid: PATIENT_LABELS.get(pid, pid),
)

default_csv_for_patient = str(BASE_DIR / f"patient_{selected_patient_id}_vitals.csv")

with st.sidebar.expander("Advanced: custom CSV path"):
    csv_override = st.text_input(
        "Vitals CSV",
        value="",
        placeholder=default_csv_for_patient,
        help="Leave blank to use the default file for the selected patient.",
    )

csv_path = csv_override.strip() if csv_override.strip() else default_csv_for_patient

model_name = st.sidebar.text_input(
    "Ollama model",
    value="llama3.2",
    help="The local Ollama model used by the reasoning layer.",
)

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

st.title("🏥 Smart Hospital — LLM-Enhanced Patient Monitoring")
st.markdown(
    "**Pipeline:** Patient vitals → rule-based detection → patient context → "
    "local LLM explanation → clinician-facing dashboard"
)

if not Path(csv_path).exists():
    st.error(f"Vitals file not found: {csv_path}")
    st.info("Place patient_P001_vitals.csv beside app.py, or enter another valid CSV path in the sidebar.")
    st.stop()

try:
    df, events = load_vitals_and_events(csv_path)
except Exception as exc:
    st.error(f"Could not load or analyze the vitals file: {exc}")
    st.stop()

patient_id = str(df["patient_id"].iloc[0]) if "patient_id" in df.columns and not df.empty else "Unknown"
patient = registry.get(patient_id)


# ---------------------------------------------------------------------------
# Hospital overview metrics
# ---------------------------------------------------------------------------

critical_count = sum(e.severity == "critical" for e in events)
warning_count = sum(e.severity == "warning" for e in events)
info_count = sum(e.severity == "info" for e in events)

latest = df.iloc[-1] if not df.empty else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Patient", patient_id)
with c2:
    st.metric("Critical events", critical_count)
with c3:
    st.metric("Warning events", warning_count)
with c4:
    st.metric("Total episodes", len(events))


# ---------------------------------------------------------------------------
# Patient context
# ---------------------------------------------------------------------------

if patient:
    with st.expander("👤 Patient context", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Age", patient.age)
        p2.metric("Sex", patient.sex)
        p3.metric("Diagnosis", patient.admitting_diagnosis)
        p4.metric("Conditions", ", ".join(patient.known_conditions))
        if patient.notes:
            st.caption(f"Notes: {patient.notes}")
else:
    st.warning(f"No patient context is registered for {patient_id}.")


# ---------------------------------------------------------------------------
# Live/latest vitals
# ---------------------------------------------------------------------------

st.subheader("📈 Latest available vitals")

if latest is not None:
    v1, v2, v3 = st.columns(3)
    v1.metric("Heart Rate", f"{latest['heart_rate']:.1f} bpm" if pd.notna(latest["heart_rate"]) else "Missing")
    v2.metric("SpO₂", f"{latest['spo2']:.1f} %" if pd.notna(latest["spo2"]) else "Missing")
    v3.metric("Respiratory Rate", f"{latest['resp_rate']:.1f} /min" if pd.notna(latest["resp_rate"]) else "Missing")

chart_df = df.copy()
chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
chart_df = chart_df.set_index("timestamp")

st.line_chart(chart_df[["heart_rate", "spo2", "resp_rate"]], height=360)


# ---------------------------------------------------------------------------
# Detected episodes
# ---------------------------------------------------------------------------

st.subheader("🚨 Detected episodes")

if not events:
    st.success("No rule-based events detected in this data stream.")
else:
    event_rows = pd.DataFrame([event_to_row(e) for e in events])
    st.dataframe(event_rows, use_container_width=True, hide_index=True)

    event_labels = [
        f"{severity_icon(e.severity)} {e.kind} — {e.start_time}"
        for e in events
    ]
    selected_idx = st.selectbox(
        "Select an episode to inspect",
        options=range(len(events)),
        format_func=lambda i: event_labels[i],
    )
    selected_event = events[selected_idx]

    st.markdown("### Event details")
    d1, d2, d3 = st.columns(3)
    d1.metric("Rule severity", selected_event.severity.upper())
    d2.metric("Event type", selected_event.kind)
    d3.metric("Merged detections", selected_event.merge_count)

    st.write(selected_event.description)
    with st.expander("Raw event data"):
        st.json(
            {
                "patient_id": selected_event.patient_id,
                "kind": selected_event.kind,
                "severity": selected_event.severity,
                "start_time": selected_event.start_time,
                "end_time": selected_event.end_time,
                "signals": selected_event.signals,
                "description": selected_event.description,
                "merge_count": selected_event.merge_count,
            }
        )

    # -----------------------------------------------------------------------
    # LLM reasoning for the selected event
    # -----------------------------------------------------------------------

    st.markdown("### 🧠 LLM reasoning")
    st.caption(
        "The LLM explains an event already flagged by deterministic rules. "
        "It is not used here to decide whether the event should have been flagged."
    )

    if st.button("Generate explanation with Ollama", type="primary"):
        try:
            with st.spinner(f"Asking local Ollama model '{model_name}'..."):
                reasoning = build_reasoning_layer(model_name)
                result = reasoning.explain(selected_event, patient)

            urgency = result.get("urgency", "unknown")
            st.markdown(f"#### {urgency_icon(urgency)} Suggested urgency: `{urgency}`")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Explanation**")
                st.write(result.get("explanation", "No explanation returned."))
            with r2:
                st.markdown("**Patient-context note**")
                st.write(result.get("context_note", "No context note returned."))

            with st.expander("Raw LLM response"):
                st.json(result)

        except Exception as exc:
            st.error(
                "The deterministic monitoring dashboard is working, but the LLM reasoning call failed."
            )
            st.code(str(exc))
            st.info(
                "Check that Ollama is running and that the selected model is available, "
                "for example: `ollama list`."
            )


# ---------------------------------------------------------------------------
# Safety / architecture note
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Prototype only — synthetic/research data. This application is not a medical device "
    "and does not replace qualified clinical judgment."
)
