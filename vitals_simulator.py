"""
Synthetic Patient Vitals Simulator
-----------------------------------
Generates a realistic, timestamped stream of patient vital signs
(Heart Rate, SpO2, Respiratory Rate) with natural variation, and
allows injecting specific clinical anomaly events (desaturation,
tachycardia, bradycardia, sensor dropout) at chosen times.

This is meant to feed downstream rule-based detection and LLM
reasoning layers in the smart-hospital-llm project.
"""

import csv
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ---------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """A clinical event to inject into the vitals stream."""
    kind: str              # "desaturation", "tachycardia", "bradycardia", "sensor_dropout"
    start_sec: int          # seconds from stream start
    duration_sec: int       # how long the event lasts
    severity: float = 1.0    # 0.0-1.0, scales how extreme the change is


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------

class PatientVitalsSimulator:
    """
    Simulates a single patient's HR, SpO2, and RR over time.

    Baseline values drift gently and have small random noise, like a
    real monitor. Injected AnomalyEvents temporarily push one or more
    signals away from baseline, then let them recover.
    """

    def __init__(
        self,
        patient_id: str = "P001",
        baseline_hr: float = 78,
        baseline_spo2: float = 98,
        baseline_rr: float = 16,
        start_time: datetime | None = None,
        seed: int | None = None,
    ):
        self.patient_id = patient_id
        self.baseline_hr = baseline_hr
        self.baseline_spo2 = baseline_spo2
        self.baseline_rr = baseline_rr
        self.start_time = start_time or datetime.now()
        self.events: list[AnomalyEvent] = []
        if seed is not None:
            random.seed(seed)

        # slow drift state so values wander naturally instead of
        # snapping back to baseline every second
        self._hr_drift = 0.0
        self._spo2_drift = 0.0
        self._rr_drift = 0.0

    def add_event(self, event: AnomalyEvent):
        self.events.append(event)
        return self

    # -- internal helpers -------------------------------------------------

    def _active_events(self, elapsed_sec: int):
        return [
            e for e in self.events
            if e.start_sec <= elapsed_sec < e.start_sec + e.duration_sec
        ]

    def _event_progress(self, event: AnomalyEvent, elapsed_sec: int) -> float:
        """0 -> 1 -> 0 shape (ramp up, hold, ramp down) across the event."""
        t = elapsed_sec - event.start_sec
        d = event.duration_sec
        ramp = max(1, d // 4)  # ramp over first/last quarter of duration
        if t < ramp:
            return t / ramp
        if t > d - ramp:
            return max(0.0, (d - t) / ramp)
        return 1.0

    def _step_drift(self, current: float, max_step: float) -> float:
        current += random.uniform(-max_step, max_step)
        return max(-1.0, min(1.0, current))  # clamp drift to [-1, 1]

    # -- main generation ----------------------------------------------------

    def _reading_at(self, elapsed_sec: int) -> dict:
        # gentle natural drift + small noise
        self._hr_drift = self._step_drift(self._hr_drift, 0.05)
        self._spo2_drift = self._step_drift(self._spo2_drift, 0.03)
        self._rr_drift = self._step_drift(self._rr_drift, 0.04)

        hr = self.baseline_hr + self._hr_drift * 4 + random.gauss(0, 0.6)
        spo2 = self.baseline_spo2 + self._spo2_drift * 1.0 + random.gauss(0, 0.2)
        rr = self.baseline_rr + self._rr_drift * 1.5 + random.gauss(0, 0.3)

        dropout = False

        for event in self._active_events(elapsed_sec):
            progress = self._event_progress(event, elapsed_sec)
            s = event.severity

            if event.kind == "desaturation":
                spo2 -= progress * s * 12          # e.g. 98 -> ~86 at severity 1
                hr += progress * s * 15             # compensatory tachycardia
            elif event.kind == "tachycardia":
                hr += progress * s * 55             # e.g. 78 -> ~133
            elif event.kind == "bradycardia":
                hr -= progress * s * 35             # e.g. 78 -> ~43
            elif event.kind == "sensor_dropout":
                dropout = True

        reading = {
            "patient_id": self.patient_id,
            "timestamp": (self.start_time + timedelta(seconds=elapsed_sec)).isoformat(),
            "elapsed_sec": elapsed_sec,
            "heart_rate": None if dropout else round(max(20, hr), 1),
            "spo2": None if dropout else round(max(50, min(100, spo2)), 1),
            "resp_rate": None if dropout else round(max(4, rr), 1),
        }
        return reading

    def stream(self, duration_sec: int, interval_sec: int = 5):
        """Yield readings from t=0 to t=duration_sec, every interval_sec."""
        for elapsed in range(0, duration_sec, interval_sec):
            yield self._reading_at(elapsed)

    def to_csv(self, path: str, duration_sec: int, interval_sec: int = 5):
        readings = list(self.stream(duration_sec, interval_sec))
        fieldnames = ["patient_id", "timestamp", "elapsed_sec", "heart_rate", "spo2", "resp_rate"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(readings)
        return path


# ---------------------------------------------------------------------
# Example usage: generate a 20-minute stream with two injected events
# ---------------------------------------------------------------------

if __name__ == "__main__":
    sim = PatientVitalsSimulator(
        patient_id="P001",
        baseline_hr=78,
        baseline_spo2=98,
        baseline_rr=16,
        start_time=datetime(2026, 9, 7, 8, 0, 0),
        seed=42,
    )

    # Event 1: gradual desaturation (e.g. COPD-related), 6-9 min mark
    sim.add_event(AnomalyEvent(kind="desaturation", start_sec=360, duration_sec=180, severity=0.9))

    # Event 2: sudden tachycardia, 13-15 min mark
    sim.add_event(AnomalyEvent(kind="tachycardia", start_sec=780, duration_sec=120, severity=0.8))

    # Event 3: brief sensor dropout, 17 min mark
    sim.add_event(AnomalyEvent(kind="sensor_dropout", start_sec=1020, duration_sec=30))

    out_path = sim.to_csv("patient_P001_vitals.csv", duration_sec=1200, interval_sec=5)
    print(f"Generated synthetic vitals stream -> {out_path}")

    # print a few sample rows around the desaturation event
    print("\nSample readings around the desaturation event (t=340-420s):")
    for r in sim.stream(duration_sec=1200, interval_sec=5):
        if 340 <= r["elapsed_sec"] <= 420:
            print(r)
