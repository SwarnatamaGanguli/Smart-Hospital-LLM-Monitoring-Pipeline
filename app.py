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

import altair as alt
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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 360px !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        overflow-wrap: break-word;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #131B2E;
        border: 1px solid #223049;
        border-radius: 8px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        color: #8B98B0;
    }

    /* Tabs */
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.6rem 1.1rem;
    }
    [data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        background-color: #2DD4BF !important;
        height: 3px;
    }
    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        background-color: #223049;
    }

    /* Severity / urgency badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .badge-critical { background: rgba(242, 92, 92, 0.15); color: #F25C5C; border: 1px solid rgba(242, 92, 92, 0.4); }
    .badge-warning  { background: rgba(245, 166, 35, 0.15); color: #F5A623; border: 1px solid rgba(245, 166, 35, 0.4); }
    .badge-info     { background: rgba(69, 123, 169, 0.15); color: #6BA3D6; border: 1px solid rgba(69, 123, 169, 0.4); }
    .badge-urgent      { background: rgba(242, 92, 92, 0.15); color: #F25C5C; border: 1px solid rgba(242, 92, 92, 0.4); }
    .badge-review_soon { background: rgba(245, 166, 35, 0.15); color: #F5A623; border: 1px solid rgba(245, 166, 35, 0.4); }
    .badge-routine     { background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); }

    .agree-banner {
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        font-size: 0.88rem;
        margin: 0.6rem 0 1rem 0;
    }
    .agree-ok        { background: rgba(52, 211, 153, 0.10); border: 1px solid rgba(52, 211, 153, 0.35); color: #34D399; }
    .agree-escalated { background: rgba(245, 166, 35, 0.10); border: 1px solid rgba(245, 166, 35, 0.35); color: #F5A623; }
    .agree-downgraded{ background: rgba(242, 92, 92, 0.12); border: 1px solid rgba(242, 92, 92, 0.45); color: #F25C5C; }

    /* Vital readouts */
    .vital-readout {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.6rem;
        color: #E7ECF5;
    }
    .vital-label {
        color: #8B98B0;
        font-size: 0.85rem;
    }
    .context-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E7ECF5;
        line-height: 1.35;
        margin-top: 0.15rem;
    }

    footer[data-testid="stFooter"] { visibility: hidden; }

    .app-footer {
        margin-top: 2.2rem;
        padding-top: 1rem;
        border-top: 1px solid #223049;
        color: #8B98B0;
        font-size: 0.82rem;
    }
    .app-footer strong {
        color: #E7ECF5;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


SEVERITY_RANK = {"info": 1, "warning": 2, "critical": 3}
URGENCY_RANK = {"routine": 1, "review_soon": 2, "urgent": 3}


def compare_severity_urgency(severity: str, urgency: str):
    """
    Returns ("agree" | "escalated" | "downgraded" | "unknown", message).

    - "escalated": the LLM suggested a higher urgency than the rule severity.
      Not a problem on its own - often means patient context legitimately
      raised the stakes (see the COPD example) - but worth surfacing so
      staff know the AI added weight beyond the raw rule.
    - "downgraded": the LLM suggested a LOWER urgency than the rule severity.
      This should not happen, since the reasoning layer is explicitly
      instructed never to downgrade a rule-flagged event. Flagged distinctly
      because it may indicate the model didn't follow its instructions.
    """
    sev_rank = SEVERITY_RANK.get(severity)
    urg_rank = URGENCY_RANK.get(urgency)
    if sev_rank is None or urg_rank is None:
        return "unknown", "Could not compare rule severity to LLM urgency."

    if urg_rank == sev_rank:
        return "agree", "LLM urgency matches the rule-based severity."
    if urg_rank > sev_rank:
        return (
            "escalated",
            f"LLM suggested a higher urgency ({urgency}) than the rule severity "
            f"({severity}) - likely due to patient context. Review the context note.",
        )
    return (
        "downgraded",
        f"LLM suggested a LOWER urgency ({urgency}) than the rule severity "
        f"({severity}). The reasoning layer is instructed never to downgrade a "
        f"flagged event - treat the rule's severity as authoritative and review manually.",
    )


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

if st.sidebar.button("🔄 Refresh data", width="stretch"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

st.markdown("## 🏥 Smart Hospital — LLM-Enhanced Patient Monitoring")
st.caption("Patient vitals → rule-based detection → patient context → local LLM explanation")

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
# Tabbed layout: Monitor / Episodes & Reasoning
# ---------------------------------------------------------------------------

tab_monitor, tab_episodes = st.tabs(["🖥️  Monitor", "🚨  Episodes & Reasoning"])

with tab_monitor:
    # -----------------------------------------------------------------------
    # Patient context
    # -----------------------------------------------------------------------

    if patient:
        with st.expander("👤 Patient context", expanded=True):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Age", patient.age)
            p2.metric("Sex", patient.sex)
            with p3:
                st.markdown('<div class="vital-label">Diagnosis</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="context-text">{patient.admitting_diagnosis}</div>', unsafe_allow_html=True)
            with p4:
                st.markdown('<div class="vital-label">Conditions</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="context-text">{", ".join(patient.known_conditions)}</div>',
                    unsafe_allow_html=True,
                )
            if patient.notes:
                st.caption(f"Notes: {patient.notes}")
    else:
        st.warning(f"No patient context is registered for {patient_id}.")

    # -----------------------------------------------------------------------
    # Live/latest vitals
    # -----------------------------------------------------------------------

    st.subheader("📈 Latest available vitals")

    if latest is not None:
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown('<div class="vital-label">Heart Rate</div>', unsafe_allow_html=True)
            hr_text = f"{latest['heart_rate']:.1f} bpm" if pd.notna(latest["heart_rate"]) else "—"
            st.markdown(f'<div class="vital-readout">{hr_text}</div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="vital-label">SpO₂</div>', unsafe_allow_html=True)
            spo2_text = f"{latest['spo2']:.1f} %" if pd.notna(latest["spo2"]) else "—"
            st.markdown(f'<div class="vital-readout">{spo2_text}</div>', unsafe_allow_html=True)
        with v3:
            st.markdown('<div class="vital-label">Respiratory Rate</div>', unsafe_allow_html=True)
            rr_text = f"{latest['resp_rate']:.1f} /min" if pd.notna(latest["resp_rate"]) else "—"
            st.markdown(f'<div class="vital-readout">{rr_text}</div>', unsafe_allow_html=True)
        st.write("")

    chart_df = df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])

    SEVERITY_COLORS = {"critical": "#F25C5C", "warning": "#F5A623", "info": "#6BA3D6"}
    SIGNAL_COLORS = {"heart_rate": "#2DD4BF", "spo2": "#8B5CF6", "resp_rate": "#F5A623"}

    plot_df = chart_df.melt(
        id_vars="timestamp",
        value_vars=["heart_rate", "spo2", "resp_rate"],
        var_name="signal",
        value_name="value",
    )

    lines = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color(
                "signal:N",
                title="Signal",
                scale=alt.Scale(
                    domain=list(SIGNAL_COLORS.keys()),
                    range=list(SIGNAL_COLORS.values()),
                ),
            ),
            tooltip=["timestamp:T", "signal:N", "value:Q"],
        )
    )

    if events:
        episodes_df = pd.DataFrame(
            [
                {
                    "start": pd.to_datetime(e.start_time),
                    "end": pd.to_datetime(e.end_time),
                    "severity": e.severity,
                    "kind": e.kind,
                    "description": e.description,
                }
                for e in events
            ]
        )
        # give point-in-time episodes a minimum visible width
        min_width = pd.Timedelta(seconds=5)
        too_narrow = (episodes_df["end"] - episodes_df["start"]) < min_width
        episodes_df.loc[too_narrow, "end"] = episodes_df.loc[too_narrow, "start"] + min_width

        bands = (
            alt.Chart(episodes_df)
            .mark_rect(opacity=0.22)
            .encode(
                x="start:T",
                x2="end:T",
                color=alt.Color(
                    "severity:N",
                    title="Episode severity",
                    scale=alt.Scale(
                        domain=list(SEVERITY_COLORS.keys()),
                        range=list(SEVERITY_COLORS.values()),
                    ),
                ),
                tooltip=["kind:N", "severity:N", "start:T", "end:T", "description:N"],
            )
        )
        combined = (bands + lines).resolve_scale(color="independent")
    else:
        combined = lines

    chart_theme = combined.configure_axis(
        gridColor="#223049", domainColor="#223049", labelColor="#8B98B0", titleColor="#8B98B0"
    ).configure_view(strokeWidth=0)

    st.altair_chart(chart_theme.properties(height=360).interactive(), width="stretch", theme=None)
    st.caption("Shaded bands mark detected episodes (red = critical, orange = warning) — hover for details.")


with tab_episodes:
    # -----------------------------------------------------------------------
    # Detected episodes
    # -----------------------------------------------------------------------

    st.subheader("🚨 Detected episodes")

    if not events:
        st.success("No rule-based events detected in this data stream.")
    else:
        event_rows = pd.DataFrame([event_to_row(e) for e in events])
        st.dataframe(event_rows, width="stretch", hide_index=True)

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

        st.markdown("#### Event details")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown('<div class="vital-label">Rule severity</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge badge-{selected_event.severity}">{selected_event.severity.upper()}</span>',
                unsafe_allow_html=True,
            )
        d2.metric("Event type", selected_event.kind)
        d3.metric("Merged detections", selected_event.merge_count)

        st.write("")
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

        # ---------------------------------------------------------------
        # LLM reasoning for the selected event
        # ---------------------------------------------------------------

        st.markdown("#### 🧠 LLM reasoning")
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
                st.markdown(
                    f'Suggested urgency: <span class="badge badge-{urgency}">{urgency.upper()}</span>',
                    unsafe_allow_html=True,
                )

                agreement, agreement_msg = compare_severity_urgency(selected_event.severity, urgency)
                banner_class = {
                    "agree": "agree-ok",
                    "escalated": "agree-escalated",
                    "downgraded": "agree-downgraded",
                    "unknown": "agree-escalated",
                }[agreement]
                banner_icon = {
                    "agree": "✓",
                    "escalated": "⬆",
                    "downgraded": "⚠",
                    "unknown": "?",
                }[agreement]
                st.markdown(
                    f'<div class="agree-banner {banner_class}">{banner_icon} {agreement_msg}</div>',
                    unsafe_allow_html=True,
                )

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
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        Prototype only — synthetic/research data. This application is not a medical device
        and does not replace qualified clinical judgment.<br>
        Built by <strong>Swarnatama Ganguli</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
