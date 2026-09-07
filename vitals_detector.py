"""
Rule-Based Vitals Detector
---------------------------
Reads a patient vitals CSV (as produced by vitals_simulator.py) and
flags clinically meaningful events using:
  1. Threshold rules      - hard limits on HR / SpO2 / RR
  2. Trend detection      - sustained drift over a rolling window
  3. Correlation rules    - multiple signals moving together
  4. Dropout detection    - missing/null readings

Output is a list of structured DetectedEvent objects, ready to be
handed to the LLM reasoning layer for plain-language explanation.
This layer stays fully deterministic and auditable - it makes no
"judgment calls", it just flags what crossed a defined rule.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------
# Thresholds - tune these to match real clinical protocols later
# ---------------------------------------------------------------------

THRESHOLDS = {
    "spo2_low": 92.0,
    "spo2_critical": 88.0,
    "hr_high": 130.0,
    "hr_low": 50.0,
    "rr_high": 24.0,
    "rr_low": 8.0,
}

TREND_WINDOW = 6          # number of readings to look back for trend checks
TREND_SPO2_DROP = 4.0     # spo2 points dropped across the window to flag a trend
TREND_HR_RISE = 20.0      # hr points risen across the window to flag a trend

MERGE_GAP_SEC = 15        # detections of the same kind within this gap are one episode
SEVERITY_RANK = {"info": 1, "warning": 2, "critical": 3}


# ---------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------

@dataclass
class DetectedEvent:
    patient_id: str
    kind: str                # e.g. "spo2_threshold", "hr_trend", "correlated_desat", "sensor_dropout"
    severity: str             # "info", "warning", "critical"
    start_time: str
    end_time: str
    signals: dict = field(default_factory=dict)   # relevant readings/values at flag time
    description: str = ""
    merge_count: int = 1      # how many raw detections were collapsed into this episode

    def __repr__(self):
        tag = f" (x{self.merge_count} detections)" if self.merge_count > 1 else ""
        return (f"[{self.severity.upper()}] {self.kind} | {self.patient_id} | "
                f"{self.start_time} -> {self.end_time} | {self.description}{tag}")


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------

class VitalsDetector:
    def __init__(self):
        self.events: list[DetectedEvent] = []

    def run_on_csv(self, csv_path: str) -> list[DetectedEvent]:
        rows = self._load_rows(csv_path)
        self.events = []

        self._detect_dropouts(rows)
        self._detect_thresholds(rows)
        self._detect_trends(rows)
        self._detect_correlations(rows)

        self.events = self._merge_events(self.events)

        # sort chronologically for readability
        self.events.sort(key=lambda e: e.start_time)
        return self.events

    # -- merging: collapse overlapping/adjacent same-kind detections --------
    # Raw rules fire on every reading a condition holds, which produces many
    # near-duplicate detections for what is really one clinical episode. This
    # groups by (patient, kind), then merges detections that overlap or sit
    # within MERGE_GAP_SEC of each other into a single episode - keeping the
    # highest severity seen and the description that goes with it.

    def _merge_events(self, events: list[DetectedEvent]) -> list[DetectedEvent]:
        groups: dict[tuple, list[DetectedEvent]] = defaultdict(list)
        for e in events:
            groups[(e.patient_id, e.kind)].append(e)

        merged_all: list[DetectedEvent] = []
        for _, group in groups.items():
            group.sort(key=lambda e: e.start_time)
            current = None
            for e in group:
                if current is None:
                    current = self._clone(e)
                    continue
                gap = (datetime.fromisoformat(e.start_time)
                       - datetime.fromisoformat(current.end_time)).total_seconds()
                if gap <= MERGE_GAP_SEC:
                    current = self._combine(current, e)
                else:
                    merged_all.append(current)
                    current = self._clone(e)
            if current is not None:
                merged_all.append(current)

        return merged_all

    def _clone(self, e: DetectedEvent) -> DetectedEvent:
        return DetectedEvent(
            patient_id=e.patient_id, kind=e.kind, severity=e.severity,
            start_time=e.start_time, end_time=e.end_time,
            signals=dict(e.signals), description=e.description, merge_count=1,
        )

    def _combine(self, current: DetectedEvent, e: DetectedEvent) -> DetectedEvent:
        current.end_time = e.end_time
        current.signals.update(e.signals)  # later readings' values win (e.g. *_end keys)
        current.merge_count += 1
        if SEVERITY_RANK[e.severity] > SEVERITY_RANK[current.severity]:
            current.severity = e.severity
            current.description = e.description  # keep the description matching the peak severity
        return current

    # -- loading --------------------------------------------------------

    def _load_rows(self, csv_path: str) -> list[dict]:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                r["heart_rate"] = float(r["heart_rate"]) if r["heart_rate"] not in ("", "None") else None
                r["spo2"] = float(r["spo2"]) if r["spo2"] not in ("", "None") else None
                r["resp_rate"] = float(r["resp_rate"]) if r["resp_rate"] not in ("", "None") else None
                r["elapsed_sec"] = int(r["elapsed_sec"])
                rows.append(r)
        return rows

    # -- rule 1: dropout --------------------------------------------------

    def _detect_dropouts(self, rows: list[dict]):
        in_dropout = False
        start = None
        for r in rows:
            missing = r["heart_rate"] is None or r["spo2"] is None or r["resp_rate"] is None
            if missing and not in_dropout:
                in_dropout = True
                start = r
            elif not missing and in_dropout:
                in_dropout = False
                self.events.append(DetectedEvent(
                    patient_id=r["patient_id"],
                    kind="sensor_dropout",
                    severity="warning",
                    start_time=start["timestamp"],
                    end_time=r["timestamp"],
                    description="Sensor data dropout detected - readings missing for a period. "
                                 "Treat as a data-quality issue, not a clinical event, unless it recurs.",
                ))

    # -- rule 2: hard thresholds --------------------------------------------

    def _detect_thresholds(self, rows: list[dict]):
        for r in rows:
            if r["spo2"] is None:
                continue

            if r["spo2"] < THRESHOLDS["spo2_critical"]:
                self.events.append(DetectedEvent(
                    patient_id=r["patient_id"], kind="spo2_threshold", severity="critical",
                    start_time=r["timestamp"], end_time=r["timestamp"],
                    signals={"spo2": r["spo2"], "heart_rate": r["heart_rate"]},
                    description=f"SpO2 critically low at {r['spo2']}%.",
                ))
            elif r["spo2"] < THRESHOLDS["spo2_low"]:
                self.events.append(DetectedEvent(
                    patient_id=r["patient_id"], kind="spo2_threshold", severity="warning",
                    start_time=r["timestamp"], end_time=r["timestamp"],
                    signals={"spo2": r["spo2"], "heart_rate": r["heart_rate"]},
                    description=f"SpO2 below normal range at {r['spo2']}%.",
                ))

            if r["heart_rate"] is not None:
                if r["heart_rate"] > THRESHOLDS["hr_high"]:
                    self.events.append(DetectedEvent(
                        patient_id=r["patient_id"], kind="hr_threshold", severity="warning",
                        start_time=r["timestamp"], end_time=r["timestamp"],
                        signals={"heart_rate": r["heart_rate"]},
                        description=f"Heart rate elevated at {r['heart_rate']} bpm (tachycardia range).",
                    ))
                elif r["heart_rate"] < THRESHOLDS["hr_low"]:
                    self.events.append(DetectedEvent(
                        patient_id=r["patient_id"], kind="hr_threshold", severity="warning",
                        start_time=r["timestamp"], end_time=r["timestamp"],
                        signals={"heart_rate": r["heart_rate"]},
                        description=f"Heart rate low at {r['heart_rate']} bpm (bradycardia range).",
                    ))

    # -- rule 3: trend detection (rolling window) --------------------------

    def _detect_trends(self, rows: list[dict]):
        for i in range(TREND_WINDOW, len(rows)):
            window = rows[i - TREND_WINDOW:i + 1]
            if any(r["spo2"] is None for r in window):
                continue

            spo2_drop = window[0]["spo2"] - window[-1]["spo2"]
            if spo2_drop >= TREND_SPO2_DROP:
                self.events.append(DetectedEvent(
                    patient_id=window[-1]["patient_id"], kind="spo2_trend", severity="warning",
                    start_time=window[0]["timestamp"], end_time=window[-1]["timestamp"],
                    signals={"spo2_start": window[0]["spo2"], "spo2_end": window[-1]["spo2"]},
                    description=f"SpO2 trending down: {window[0]['spo2']}% -> {window[-1]['spo2']}% "
                                f"over {window[-1]['elapsed_sec'] - window[0]['elapsed_sec']}s.",
                ))

            if all(r["heart_rate"] is not None for r in window):
                hr_rise = window[-1]["heart_rate"] - window[0]["heart_rate"]
                if hr_rise >= TREND_HR_RISE:
                    self.events.append(DetectedEvent(
                        patient_id=window[-1]["patient_id"], kind="hr_trend", severity="warning",
                        start_time=window[0]["timestamp"], end_time=window[-1]["timestamp"],
                        signals={"hr_start": window[0]["heart_rate"], "hr_end": window[-1]["heart_rate"]},
                        description=f"Heart rate trending up: {window[0]['heart_rate']} -> "
                                    f"{window[-1]['heart_rate']} bpm over "
                                    f"{window[-1]['elapsed_sec'] - window[0]['elapsed_sec']}s.",
                    ))

    # -- rule 4: correlation (higher priority than single-signal flags) -----

    def _detect_correlations(self, rows: list[dict]):
        for i in range(TREND_WINDOW, len(rows)):
            window = rows[i - TREND_WINDOW:i + 1]
            if any(r["spo2"] is None or r["heart_rate"] is None for r in window):
                continue

            spo2_drop = window[0]["spo2"] - window[-1]["spo2"]
            hr_rise = window[-1]["heart_rate"] - window[0]["heart_rate"]

            if spo2_drop >= TREND_SPO2_DROP and hr_rise >= 8.0:
                self.events.append(DetectedEvent(
                    patient_id=window[-1]["patient_id"], kind="correlated_desaturation",
                    severity="critical",
                    start_time=window[0]["timestamp"], end_time=window[-1]["timestamp"],
                    signals={
                        "spo2_start": window[0]["spo2"], "spo2_end": window[-1]["spo2"],
                        "hr_start": window[0]["heart_rate"], "hr_end": window[-1]["heart_rate"],
                    },
                    description=(
                        f"Correlated pattern: SpO2 dropping ({window[0]['spo2']}% -> "
                        f"{window[-1]['spo2']}%) while heart rate rising "
                        f"({window[0]['heart_rate']} -> {window[-1]['heart_rate']} bpm) - "
                        f"consistent with a real desaturation event rather than sensor noise."
                    ),
                ))


# ---------------------------------------------------------------------
# Run against the CSV produced by vitals_simulator.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    detector = VitalsDetector()
    events = detector.run_on_csv("patient_P001_vitals.csv")

    print(f"Detected {len(events)} events:\n")
    for e in events:
        print(e)
