"""
Reasoning Layer (LangChain + Ollama)
--------------------------------------
Takes a DetectedEvent (from vitals_detector.py) plus a PatientContext
(from patient_context.py) and asks a local LLM, via Ollama, to produce
a plain-language explanation and a suggested urgency level.

IMPORTANT - this layer only EXPLAINS what the rule-based detector
already flagged. It does not decide on its own whether something is
worth flagging - that stays with the deterministic detection layer.

Requires Ollama running locally with the target model pulled, e.g.:
    ollama pull llama3.2
    ollama serve   (usually runs automatically after install)

Requires: pip install langchain-ollama --break-system-packages
"""

import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from vitals_detector import DetectedEvent
from patient_context import PatientContext


SYSTEM_PROMPT = """You are a clinical monitoring assistant supporting hospital staff.

You will be given:
1. A detected event from a deterministic, rule-based monitoring system
   (thresholds, trends, and signal-correlation rules already decided
   this event is worth flagging - your job is NOT to second-guess
   whether it should have been flagged).
2. Relevant patient context.

Your job is ONLY to:
- Explain what happened, in plain, clear language a nurse or doctor
  can read in a few seconds.
- Note anything about the patient's context that makes this more or
  less concerning.
- Suggest an urgency level: "routine", "review_soon", or "urgent".

You must NOT:
- Invent details not present in the event or patient context.
- Downgrade or dismiss what the detection layer flagged.
- Provide a diagnosis - only describe the pattern and its likely
  clinical relevance.

Respond ONLY with valid JSON, no other text, in this exact shape:
{{
  "explanation": "<2-3 sentence plain-language explanation>",
  "context_note": "<1 sentence on how patient context affects this, or 'No relevant context.'>",
  "urgency": "routine" | "review_soon" | "urgent"
}}"""

USER_TEMPLATE = """Detected event:
- Type: {kind}
- Severity (from detection rules): {severity}
- Time window: {start_time} to {end_time}
- Signal values: {signals}
- Rule-based description: {description}

Patient context:
{patient_context}
"""


class ReasoningLayer:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.1):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_TEMPLATE),
        ])
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def explain(self, event: DetectedEvent, patient: PatientContext | None) -> dict:
        patient_text = patient.as_prompt_text() if patient else "No patient context available."

        result = self.chain.invoke({
            "kind": event.kind,
            "severity": event.severity,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "signals": json.dumps(event.signals),
            "description": event.description,
            "patient_context": patient_text,
        })
        return result


# ---------------------------------------------------------------------
# Run end-to-end: detector output -> reasoning layer, for each episode
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from vitals_detector import VitalsDetector
    from patient_context import PatientRegistry

    # 1. Detect episodes from the synthetic vitals CSV
    detector = VitalsDetector()
    events = detector.run_on_csv("patient_P001_vitals.csv")

    # 2. Set up patient context
    registry = PatientRegistry()
    registry.add(PatientContext(
        patient_id="P001",
        age=67,
        sex="male",
        admitting_diagnosis="COPD exacerbation",
        known_conditions=["COPD", "hypertension"],
        notes="Baseline SpO2 runs slightly lower than typical due to COPD.",
    ))

    # 3. Explain each episode
    reasoning = ReasoningLayer(model="llama3.2")

    print(f"Reasoning over {len(events)} detected episodes...\n")
    for event in events:
        patient = registry.get(event.patient_id)
        try:
            result = reasoning.explain(event, patient)
        except Exception as exc:
            print(f"--- {event.kind} ({event.start_time}) ---")
            print(f"  [Error calling Ollama: {exc}]\n")
            continue

        print(f"--- {event.kind} ({event.start_time} -> {event.end_time}) ---")
        print(f"  Rule severity: {event.severity}")
        print(f"  LLM urgency:   {result.get('urgency')}")
        print(f"  Explanation:   {result.get('explanation')}")
        print(f"  Context note:  {result.get('context_note')}\n")
