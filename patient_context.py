"""
Patient Context
-----------------
A minimal structured profile for a patient, used to give the LLM
reasoning layer relevant background when explaining a detected event.

This is intentionally simple (a dict-backed store) - no database needed
yet. Swap this for a real lookup (Postgres, EHR system, etc.) later
without changing how the reasoning layer consumes it.
"""

from dataclasses import dataclass, field


@dataclass
class PatientContext:
    patient_id: str
    age: int
    sex: str
    admitting_diagnosis: str
    known_conditions: list[str] = field(default_factory=list)
    notes: str = ""

    def as_prompt_text(self) -> str:
        """Plain-language summary suitable for inserting into an LLM prompt."""
        conditions = ", ".join(self.known_conditions) if self.known_conditions else "none documented"
        text = (
            f"Patient {self.patient_id}: {self.age}-year-old {self.sex}. "
            f"Admitting diagnosis: {self.admitting_diagnosis}. "
            f"Known conditions: {conditions}."
        )
        if self.notes:
            text += f" Additional notes: {self.notes}"
        return text


# ---------------------------------------------------------------------
# Simple in-memory patient registry (stand-in for a real EHR lookup)
# ---------------------------------------------------------------------

class PatientRegistry:
    def __init__(self):
        self._patients: dict[str, PatientContext] = {}

    def add(self, patient: PatientContext):
        self._patients[patient.patient_id] = patient
        return self

    def get(self, patient_id: str) -> PatientContext | None:
        return self._patients.get(patient_id)


if __name__ == "__main__":
    registry = PatientRegistry()
    registry.add(PatientContext(
        patient_id="P001",
        age=67,
        sex="male",
        admitting_diagnosis="COPD exacerbation",
        known_conditions=["COPD", "hypertension"],
        notes="Baseline SpO2 runs slightly lower than typical due to COPD.",
    ))

    p = registry.get("P001")
    print(p.as_prompt_text())
