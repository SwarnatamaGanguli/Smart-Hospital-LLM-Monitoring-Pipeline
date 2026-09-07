"""
Multi-Patient Vitals Generator
--------------------------------
Generates synthetic vitals CSVs for several patients, each with a
different baseline and a different clinical picture - so the dashboard
has more than one type of case to show.

Run this once to (re)generate all patient CSVs before using app.py's
patient selector.
"""

from datetime import datetime

from vitals_simulator import PatientVitalsSimulator, AnomalyEvent


def generate_p001_copd():
    """P001: COPD patient - gradual desaturation, then tachycardia, then a sensor dropout."""
    sim = PatientVitalsSimulator(
        patient_id="P001",
        baseline_hr=78,
        baseline_spo2=98,
        baseline_rr=16,
        start_time=datetime(2026, 9, 7, 8, 0, 0),
        seed=42,
    )
    sim.add_event(AnomalyEvent(kind="desaturation", start_sec=360, duration_sec=180, severity=0.9))
    sim.add_event(AnomalyEvent(kind="tachycardia", start_sec=780, duration_sec=120, severity=0.8))
    sim.add_event(AnomalyEvent(kind="sensor_dropout", start_sec=1020, duration_sec=30))
    return sim


def generate_p002_cardiac():
    """P002: Cardiac/arrhythmia risk patient - a bradycardia dip, then a tachycardia spike."""
    sim = PatientVitalsSimulator(
        patient_id="P002",
        baseline_hr=82,
        baseline_spo2=97,
        baseline_rr=15,
        start_time=datetime(2026, 9, 7, 8, 0, 0),
        seed=7,
    )
    # Sudden bradycardia episode - possible conduction issue
    sim.add_event(AnomalyEvent(kind="bradycardia", start_sec=300, duration_sec=150, severity=1.0))
    # Later, a sudden tachycardia spike - possible arrhythmia
    sim.add_event(AnomalyEvent(kind="tachycardia", start_sec=900, duration_sec=100, severity=1.0))
    return sim


def generate_p003_postop():
    """P003: Post-op patient - slow respiratory-depression-style desaturation (opioid risk pattern)."""
    sim = PatientVitalsSimulator(
        patient_id="P003",
        baseline_hr=88,
        baseline_spo2=96,
        baseline_rr=14,
        start_time=datetime(2026, 9, 7, 8, 0, 0),
        seed=19,
    )
    # Slow, prolonged desaturation - classic opioid-related respiratory depression pattern
    sim.add_event(AnomalyEvent(kind="desaturation", start_sec=500, duration_sec=400, severity=0.7))
    sim.add_event(AnomalyEvent(kind="sensor_dropout", start_sec=1100, duration_sec=40))
    return sim


if __name__ == "__main__":
    generators = {
        "P001": generate_p001_copd,
        "P002": generate_p002_cardiac,
        "P003": generate_p003_postop,
    }

    for patient_id, build_sim in generators.items():
        sim = build_sim()
        path = sim.to_csv(f"patient_{patient_id}_vitals.csv", duration_sec=1200, interval_sec=5)
        print(f"Generated {path}")
