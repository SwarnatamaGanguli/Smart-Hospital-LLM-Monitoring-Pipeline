# Smart Hospital LLM Monitoring Pipeline

An LLM-enhanced patient monitoring system that watches vital signs, flags
concerning patterns using deterministic rules, and uses a locally-run
language model to explain those patterns in plain language for clinical
staff — without letting the AI make the underlying medical judgment calls
itself.

This is a research/learning prototype built on synthetic data. It is not a
medical device and does not replace clinical judgment.

---

## What This Project Does

Hospitals already have monitors that beep when a vital sign crosses a
limit. What they don't do well is explain *why* something matters, or
connect multiple signals into one coherent picture. This project adds that
missing layer:

1. A stream of patient vitals (heart rate, oxygen level, breathing rate)
   is watched continuously.
2. A deterministic, rule-based layer flags anything concerning — a hard
   limit crossed, a value trending in a bad direction, or multiple signals
   moving together in a way that confirms something real is happening.
3. A locally-run language model, given that flagged event plus relevant
   patient context, writes a short plain-language explanation and suggests
   an urgency level.
4. A dashboard shows all of this to a clinician, per patient, with the
   raw vitals chart, the flagged episodes, and the AI's explanation
   available on demand.

The core design decision throughout: **the rules decide what counts as an
event; the AI only explains it.** The AI is never used to decide on its
own that something should or shouldn't have been flagged, and it is
instructed never to downgrade or soften what the rules already caught.
This keeps the system auditable — every alert can be traced back to a
specific, explainable rule — while still getting the real benefit of an
LLM: turning a wall of numbers into something a nurse can read in five
seconds and act on.

---

## Architecture

The pipeline has four layers, kept deliberately separate so each one can
be tested, trusted, and replaced independently:

- **Data layer** — the source of vitals readings, arriving over time.
  Currently a synthetic generator; in a real deployment this would be
  live device data (see the IoT/IoMT section below).
- **Detection layer** — fully deterministic. Applies threshold rules,
  trend detection, and multi-signal correlation checks. Produces a short,
  clean list of "episodes" rather than raw noise.
- **Context layer** — a small patient profile (age, diagnosis, known
  conditions) that gets attached to whatever episode the detector found,
  so the AI's explanation is grounded in the specific patient, not a
  generic answer.
- **Reasoning layer** — a local language model, connected through
  LangChain, that turns a detected episode plus patient context into a
  structured explanation and a suggested urgency level.
- **Presentation layer** — a dashboard where a clinician can browse
  patients, see the vitals chart with detected episodes highlighted on
  it, and request the AI's explanation for any specific episode.

---

## How It Was Built, Step by Step

**1. The synthetic vitals generator.**
Since real hospital data requires credentialed access and a data-use
agreement, the project started with a tool that generates realistic,
naturally-wobbling vital sign data and lets specific problem events be
injected at exact, known times. This meant every later piece of the
system could be tested against a known ground truth, rather than guessing
against unlabeled data from day one.

**2. The rule-based detector.**
A tool that reads the vitals stream and flags concerning patterns using
four kinds of checks: hard thresholds, sustained trends, multi-signal
correlation (which helps distinguish a real event from sensor noise), and
missing-data detection. The very first version flagged the same
underlying problem dozens of times in a row, because the checks ran on
every single reading. A merging step was added afterward to collapse
nearby detections of the same type into one clean episode — much closer
to how a clinician would actually describe what happened.

**3. Patient context and the reasoning layer.**
A small structured patient profile (age, diagnosis, known conditions) was
added, along with the actual connection to a local language model through
LangChain. The model is given a strict instruction set: explain only,
never invent details, never downgrade what the rules flagged, and always
respond in a fixed structured format so the output can be used
programmatically rather than as free text.

**4. The dashboard.**
A Streamlit application ties everything together — loading a patient's
vitals, running detection, showing the chart and the episode list, and
calling the reasoning layer on demand for a selected episode. It was
built to keep working even if the language model isn't available, since
the deterministic monitoring layer has value on its own.

**5. Multi-patient support.**
The project expanded from one patient to three, each representing a
different clinical picture: a COPD patient with a gradual oxygen drop, a
cardiac patient with sudden heart rate swings, and a post-operative
patient with a slow, sustained desaturation pattern consistent with
opioid-related respiratory depression. This turned the dashboard from a
single demo into something closer to a real ward view.

**6. Visual timeline markers.**
The vitals chart was upgraded so that every detected episode is shown as
a colored band directly behind the raw signal — instead of only being
readable from a separate table — making it possible to see exactly where
and how long each event lasted at a glance.

**7. Visual redesign.**
The dashboard's look was reworked around the idea of an actual bedside
monitor rather than a generic web dashboard: a dark, deep-navy theme with
a cyan accent, monospaced digits for the vital readouts (so numbers align
the way a real monitor displays them), and color-coded severity badges
following the same red/amber/green convention used in real clinical
alerting. The single long page was also split into two tabs so the
information isn't all stacked in one long, sparse scroll.

---

## Challenges Along the Way

**Balancing what the AI is allowed to do.**
The hardest design decision wasn't technical — it was deciding that the
AI should only explain, never decide. It would have been easy to let the
language model judge on its own whether a pattern was worth flagging, but
that would make the system's behavior much harder to trust and audit.
Keeping detection fully rule-based, and treating the AI purely as a
communication layer on top of it, was the choice that made the rest of
the system trustworthy.

**Noisy, duplicate alerts.**
Checking vitals against rules on every single reading meant one real
clinical event could generate dozens of near-identical flags. This isn't
just a cosmetic problem — in a real system it would flood staff with
alert fatigue, which is a well-known cause of missed alerts in real
hospitals. Solving it required grouping nearby detections of the same
type into a single episode, which took some care to get right without
accidentally merging two genuinely separate events into one.

**Environment differences between development and the actual machine.**
Paths, file locks, and installed model versions didn't always match
between where the code was written and where it was actually run. A file
open in Excel silently blocked Python from writing to it. A hardcoded
file path from one environment didn't exist on another. The model
initially chosen (llama3.1) wasn't the one actually available locally
(llama3.2 was). None of these were difficult individually, but together
they were a reminder that code has to be verified in the actual target
environment, not just assumed to work because it's logically correct.

**Restructuring without breaking things.**
Reworking the dashboard's layout into tabs and adding custom styling
meant re-indenting a large block of existing, working code by hand — a
process that's easy to introduce small mistakes in, like a stray leftover
character from code that was supposed to be fully replaced. This is why
every change was re-compiled and smoke-tested afterward rather than
assumed correct.

**Verifying a UI without being able to see it rendered.**
Because the dashboard needs to be run on the developer's own machine
(Streamlit, and especially the local AI model through Ollama, don't run
in a sandboxed environment), every visual change had to be verified for
correctness — no syntax errors, no runtime errors, valid chart output —
without being able to see the actual rendered result. Real visual
feedback ("does this actually look good") had to come from the person
running it locally, not from the assistant helping to build it.

---

## Tech Stack

- **Python** — core language for the data simulation, detection logic,
  and dashboard.
- **Ollama** — runs the language model locally, so patient data never
  has to leave the machine it's running on. Important for any real
  healthcare use, given data privacy requirements.
- **LangChain** — organizes how information (a detected episode, patient
  context) flows into and out of the language model in a structured,
  repeatable way.
- **Streamlit** — the dashboard framework, chosen for how quickly a
  usable interface can be built directly from Python without a separate
  frontend codebase.
- **Altair** — used for the vitals chart, specifically because it
  supports layering the colored episode bands behind the vitals lines,
  which a simpler charting tool couldn't do as cleanly.
- **Pandas** — data handling for the vitals CSVs and the detected
  episode records.

---

## From Simulation to Real Hospital IoT / IoMT Data

Everything above runs on synthetic, file-based data. Moving this toward
real hospital deployment means replacing the data layer with something
that ingests live device data — this is a significant architectural
change, not just a swap of one file for another. Here's what that
actually involves.

### The core problem: real vitals don't arrive as a CSV

Real patient monitors, wearable sensors, and bedside devices — often
called **IoMT (Internet of Medical Things)** devices — send data
continuously, in small messages, often using protocols designed
specifically for healthcare or general industrial IoT rather than a flat
file. Three separate concerns come up immediately: what protocol carries
the data, what format the data itself is in, and what happens to the
data once it arrives.

### Communication protocols commonly used

- **MQTT** — a lightweight publish/subscribe messaging protocol that is
  the de facto standard for IoT device telemetry in general, including
  many modern medical devices and hospital sensor networks. Devices
  publish readings to a "topic," and any interested system (like this
  project's detection layer) subscribes to receive them in real time.
- **HL7 v2 and HL7 FHIR** — the two dominant healthcare data
  interoperability standards. HL7 v2 is the older, still widely used
  message format for exchanging clinical data between hospital systems.
  FHIR (Fast Healthcare Interoperability Resources) is the modern,
  web-friendly successor, increasingly used for exchanging structured
  clinical data including vital signs, and is the standard most new
  healthcare integrations are built against today.
- **IEEE 11073** — a standard specifically for personal health device
  communication (the kind of standard a pulse oximeter or blood pressure
  cuff manufacturer might implement), aimed at making medical devices
  from different vendors interoperable.
- **Bluetooth Low Energy (BLE)** — the common protocol for wearable
  health sensors (fitness trackers, wearable pulse oximeters) to
  communicate with a nearby receiving device.

A realistic hospital deployment often has to support more than one of
these at once, since different device vendors and departments use
different standards.

### Libraries and tools this project would add

- **paho-mqtt** — a Python library for publishing and subscribing to
  MQTT topics; this would likely replace the CSV file as the live input
  to the detection layer.
- **bleak** — a cross-platform Python library for communicating with
  Bluetooth Low Energy devices directly, relevant if wearable sensors are
  involved rather than fixed bedside monitors.
- **hl7apy or python-hl7** — libraries for parsing and constructing HL7
  v2 messages, needed if integrating with existing hospital systems that
  speak that format.
- **fhirclient or fhir.resources** — libraries for working with FHIR
  resources, relevant for integrating with more modern hospital data
  systems or exporting processed episodes back into a hospital's records
  system in a standard format.
- **asyncio and websockets** — for building the actual real-time
  ingestion service that listens continuously rather than processing a
  file once and stopping, which is a fundamental shift from how the
  current prototype works.
- **A time-series database** (such as InfluxDB or TimescaleDB, each with
  their own Python client libraries) — real streaming vitals need to be
  stored efficiently for both live use and later review, which a flat
  CSV file is not designed for at scale.
- **Kafka** (via kafka-python or confluent-kafka), for larger
  deployments — if data is coming from many devices across many patients
  and departments at once, a message queue like Kafka becomes necessary
  to reliably handle the volume without dropping readings.

### Architectural changes this requires

The current detection layer already operates on structured rows of data
regardless of where they came from, which means it could largely stay
the same. What changes is everything upstream of it: instead of reading
a finished CSV file once, the system would need a long-running service
that continuously listens for incoming messages (from MQTT, or a
websocket, or a queue), converts them into the same structured format the
detector already expects, and feeds them in as they arrive. The reasoning
layer would similarly shift from being triggered by a button click to
being triggered automatically whenever the detection layer emits a new
episode, since a live system can't wait for someone to manually ask for
an explanation.

### Compliance and security considerations that come with real data

Working with real patient data introduces requirements this
prototype does not currently need to handle: encryption of data in
transit (MQTT and other protocols support TLS for this) and at rest,
strict access controls over who can view patient data, and compliance
with healthcare data regulations such as HIPAA in the US. Running the
language model locally through Ollama — a decision already made in this
project — is directly relevant here, since it avoids sending any patient
data to an external service, which would otherwise be a major compliance
concern.

### What stays the same

Despite all of the above, the actual reasoning about what's clinically
concerning — the threshold rules, the trend detection, the correlation
logic, and the instruction set given to the language model — would not
need to change. This is intentional: the core logic was built to operate
on structured readings, independent of where those readings originally
came from, so the investment in getting the detection and reasoning
layers right carries forward even as the data source underneath it
changes substantially.

---

## Current Limitations

- Runs on synthetic data only; not yet tested against real hospital
  datasets such as MIMIC or PhysioNet, though the pipeline was designed
  with that swap in mind.
- Single-machine prototype; not built for the concurrency or reliability
  requirements of a real hospital IT environment.
- No authentication, audit logging, or access control — required before
  any real patient data could be involved.
- The language model's explanations have not been clinically validated;
  they are a research demonstration of the approach, not a vetted
  clinical tool.

---

## Disclaimer

This project is a research and learning prototype. It uses synthetic
data, has not been clinically validated, and is not a medical device. It
does not replace qualified clinical judgment.
