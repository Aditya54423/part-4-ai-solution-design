# Solution Report — AI Solution Design for a Business Problem

**Part 4 | Healthcare Domain | ICU Sepsis Early Warning System**  
**Reference Data:** 
**Dataset Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs  
`ai_usecase_reference_catalog.csv` :- https://drive.google.com/file/d/1lMXyGp0UaoyhnU4SzWnTDIOVV1VDoq7N/view?usp=drive_link
`business_kpi_sample.csv` :- https://drive.google.com/file/d/1dPoRjnTcysR4lhYHhSu7p10nghSYtAox/view?usp=drive_link
`data_dictionary.md`:- https://drive.google.com/file/d/1XJOc-4VBJiQKJkKpEZ6U1321kGO_KGiE/view?usp=drive_link
**Full working notebook:** `notebook.ipynb`

---

## A Note on How I Approached This

Before I get into the formal task structure, I want to explain my thinking. Part 4 is the only part of this project that does not require training a model. Most people treat that as an opportunity to write a shallow document. I treated it as the opposite — a chance to design something that I could actually defend in front of a clinical ethics board or a hospital CTO.

I started by loading and exploring both reference files properly (see notebook cells 3–6). The catalog told me what AI task types and model families the course designers associated with each domain. The KPI sample gave me 12 months of real operational baseline data — average resolution time, error rates, manual processing hours, patient satisfaction scores — all of which I used to anchor every business impact claim in this report to an actual number, not a vague percentage.

I chose Healthcare not because it was the safest choice, but because it was the hardest. Healthcare is the one domain where a false negative is not a business loss — it is a patient death. That asymmetry changes every decision: how you set thresholds, what metrics you optimise for, how you think about responsible AI. Getting that right is more interesting than any other domain on the list.

---

## Task 1 — Business Domain Selection

**Selected Domain: Healthcare — Intensive Care Units (ICU)**

When I looked at the reference catalog, every domain had a reasonable use case. But I kept coming back to healthcare for one specific reason: the cost of being wrong is denominated in human lives, not revenue. That is a fundamentally different engineering and ethical problem.

Within healthcare, I went further than the catalog entry. The catalog suggests "Medical image triage" with a CNN. That is a solved problem — CheXNet did it in 2017, and dozens of FDA-cleared products exist today. I wanted to design something that is still genuinely hard: **early sepsis detection from multimodal ICU time-series data**.

Sepsis kills approximately 11 million people annually. It is the leading cause of in-hospital deaths worldwide. The cruel part is that it is not untreatable — early antibiotics work. The problem is that by the time a clinician recognises the pattern, the optimal intervention window has often already closed.

This is exactly the kind of problem where AI can provide asymmetric value. Not because it is smarter than an intensivist, but because it never gets tired, never manages six patients simultaneously, and can integrate 40 data streams continuously rather than checking them every 4–8 hours.

I justified this choice quantitatively in the notebook using the KPI data. The baseline shows:
- Average resolution time: **30.3 hours** (CV = 24% — high variance is itself a problem)
- Clinical error rate: **6.96%** (CV = 26%)
- Manual processing hours per month: **454 hours**
- Patient satisfaction: **7.04 out of 10**

The high coefficient of variation on resolution time and error rate tells me the current process is not just slow — it is *inconsistent*. An AI system that delivers consistent performance addresses both the mean and the variance.

---

## Task 2 — Business Problem Definition

### What problem is being solved?

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. In an ICU it develops gradually — a slow accumulation of abnormal signals across heart rate, blood pressure, temperature, lactate, white cell count, and dozens of other parameters. No single signal triggers an alarm. The danger lies in the **pattern across signals over time**, and in the rate of change, not the current value.

The current standard of care uses scoring systems called SOFA and qSOFA. A nurse or doctor calculates these manually every 4–8 hours. The problem is that sepsis can progress from early-stage to septic shock — a state with 40%+ mortality — in under two hours. A system that checks for it six times a day is not fast enough.

What I am designing is a system that watches every ICU patient's data streams continuously and generates a rolling risk score every 15 minutes. When that score crosses a configurable threshold, the relevant clinician receives a soft notification with an explanation of which clinical signals are driving the score. The system does not diagnose sepsis. It says: "this patient's trajectory over the last 12 hours looks like the trajectories of patients who went on to develop sepsis. You should take a closer look."

That is a very different — and much safer — framing than "the AI says this patient has sepsis."

### Who are the stakeholders?

| Stakeholder | Their primary concern |
|---|---|
| ICU nurses | Manageable alert rate; alerts that make clinical sense |
| Intensivists | Ranked list of highest-risk patients at the start of each shift |
| Hospital administrators | Reduced ICU length-of-stay, reduced mortality, reduced liability exposure |
| Patients and families | Earlier intervention; transparency about when AI is involved |
| Clinical informatics teams | Auditable model, explainable outputs, regulatory compliance |
| Insurance payers | Lower cost per episode through faster treatment initiation |

### Current manual process and its limitations

The current process: vital signs are charted at regular intervals. Labs are ordered episodically. A nurse managing four patients cannot hold 200+ data points per patient in working memory simultaneously. Scoring happens at fixed intervals, not continuously. Clinical notes — which often contain early sentinel observations like "patient seems more confused than yesterday" — are never integrated into any formal scoring.

Key limitations I identified:
1. **Infrequency** — SOFA is calculated every 4–8 hours; sepsis can turn critical in 2
2. **Information overload** — no human can continuously monitor 40 variables across 4 patients
3. **Retrospective not predictive** — SOFA reflects *current* dysfunction, not *trajectory toward* dysfunction
4. **Notes not integrated** — nursing observations carry clinical signal that never enters structured scoring
5. **Inconsistency** — the KPI data showed 26% coefficient of variation in error rate, which means performance varies dramatically with staffing, shift length, and fatigue

---

## Task 3 — AI Task Type Identification

**Selected: Binary Classification + Sequence Prediction (combined)**

At each 15-minute prediction step, the model answers a single binary question: *"Will this patient develop sepsis in the next 6–12 hours?"* The output is a probability between 0 and 1. When it crosses a configurable threshold, an alert fires.

That is binary classification. But the input is not a static feature vector — it is a 48-hour window of multivariate time-series observations. The model must understand temporal dependencies. It needs to know that a lactate level of 2.1 mmol/L that has been rising for 8 hours is far more dangerous than a lactate of 2.1 that has been stable for 48 hours. That is sequence prediction.

I chose this combination deliberately. A static classifier on snapshot features ignores trajectory — and trajectory is the signal. I showed this in the notebook (cell 13) with a simulated comparison: a septic patient and a stable patient can have similar vital signs at any given moment. The trend over 24 hours is what separates them.

**Why not the alternatives:**

| Alternative | Reason not chosen |
|---|---|
| Anomaly detection | Unsupervised — gives "normal vs abnormal" but cannot be validated against labelled outcomes with known mortality rates |
| Regression (time-to-sepsis) | Exact timing is not clinically actionable; a rising probability crossing a threshold is |
| Multi-label classification | Useful for identifying which organ system is failing, but overkill for the triage question |
| Image classification (catalog default) | The input data is not images — it is time-ordered clinical measurements |

The catalog suggests CNN for healthcare. I extended this to TFT because the data modality is different. A CNN operates on spatial structure. ICU data has temporal structure. Using a CNN on flattened time series would destroy the very signal I am trying to detect.

---

## Task 4 — Data Requirement Plan

### Three data streams

**Stream 1 — Vital Signs (structured, continuous)**
Heart rate, systolic/diastolic blood pressure, MAP, SpO₂, respiratory rate, temperature. Recorded every 1–5 minutes by bedside monitors. Resampled to 1-hour bins. 6 features × 48 time steps per patient window.

**Stream 2 — Laboratory Results (structured, episodic)**
WBC, lactate, creatinine, bilirubin, platelets, CRP, procalcitonin, glucose, sodium, potassium, ALT, urine output, blood cultures, procalcitonin. Ordered at irregular intervals — 4 to 24 hours apart. Forward-filled between measurements with a missing-value indicator flag so the model knows the difference between "lactate was normal 2 hours ago" and "lactate was not measured."

**Stream 3 — Clinical Notes (unstructured)**
Nursing shift notes, physician progress notes, medication change records. Processed by Bio_ClinicalBERT pretrained on MIMIC-III. Each note encoded to a 768-dimensional embedding, projected down to 32 dimensions, and injected as a shift-level context vector. This is the one stream that most existing early warning systems ignore entirely. I include it because nursing notes often contain early observations — "patient seems more confused than yesterday," "patient's family mentioned she has been less responsive" — that carry genuine clinical signal and are never captured in structured scoring.

### Target variable

`sepsis_onset_within_12h` — binary. Derived retrospectively from electronic health records using Sepsis-3 criteria: SOFA score increase of ≥2 combined with suspected infection. Labels are generated automatically by an NLP pipeline running on discharge summaries and chart notes, then a 10% stratified sample is manually reviewed by two senior clinicians for quality assurance.

### Minimum viable dataset

50,000 labelled patient-stay windows with a minimum of 10,000 sepsis-positive cases before a model should be considered production-ready. Below this threshold, the model cannot generalise reliably across different patient populations and ICU environments. I would start with public datasets (MIMIC-III: 60,000+ ICU stays; eICU: 200,000+ stays) for initial development, then fine-tune on local hospital data using federated learning to avoid privacy risks from data centralisation.

### Data quality risks

I built a full risk table in the notebook (cell 15). The most important ones:

| Risk | Why it matters | Mitigation |
|---|---|---|
| Missing lab values | Labs are ordered episodically — up to 24h gaps | Forward-fill + missing indicator flag; never impute with mean (clinical data is not normally distributed) |
| Irregular sampling across ICUs | Different ICUs have different monitoring protocols | Resample all streams to 1-hour bins as the common denominator |
| Label noise from NLP extraction | Sepsis-3 criteria applied to narrative text is imperfect | 10% manual audit by two independent clinicians; compute inter-rater agreement |
| Patient population mismatch | MIMIC-III patients (Boston academic medical centre) differ from a rural district hospital | Federated fine-tuning on local data; demographic bias audit before deployment sign-off |
| De-identification failures | Clinical notes contain names, dates of birth, relative names, implant serial numbers | Microsoft Presidio + custom clinical NLP rules; random spot-check of 500 de-identified notes per month |

---

## Task 5 — Model Recommendation

**Recommended: Temporal Fusion Transformer (TFT) with Bio_ClinicalBERT note encoder**

I chose TFT for four reasons that are specific to this problem, not generic model popularity:

**1. It was designed for exactly this data structure.**
TFT handles mixed-frequency inputs natively — it has separate pathways for static covariates (age, sex, admission type), known-future inputs, and observed time series. ICU data has all three. An LSTM or vanilla Transformer would require awkward engineering to handle them together. TFT does it architecturally.

**2. Variable Selection Networks (VSNs) handle missing data and irrelevant features gracefully.**
Not all 40 features are relevant to all patients at all times. For a post-surgical patient, lactate trends matter enormously. For a patient admitted for diabetic ketoacidosis, glucose and pH matter more. VSNs learn these per-patient, per-time-step weights dynamically rather than applying fixed feature importances across all cases.

**3. Multi-head attention weights are clinically interpretable.**
When the model fires an alert, the attention weights tell us which time steps and which features drove the score. I can show the nurse: "lactate at t-8h and rising heart rate at t-4h contributed 63% of this alert." That is not post-hoc explainability bolted on after training — it is interpretability by architectural design. This is not a nice-to-have in healthcare. It is a regulatory and clinical requirement.

**4. It produces calibrated quantile outputs.**
Rather than a single probability, TFT can output P10, P50, and P90 risk estimates. This gives clinicians an uncertainty interval: "we are highly confident this patient is at elevated risk" vs "this patient might be at elevated risk — please verify." That distinction matters when you are deciding whether to initiate broad-spectrum antibiotics, which carry their own risks.

### Training strategy

Phase 1: pre-train on MIMIC-III and eICU combined (≈250,000 patient stays). This initialises the model with general ICU physiology patterns.

Phase 2: federated fine-tuning at each participating hospital. The model trains locally on that hospital's data. Only weight gradient updates are shared with the central server — never raw patient data. This is how I handle the privacy constraint that would make centralised training legally impossible in most jurisdictions.

Loss function: Focal Loss with γ=2 and α=0.25. Sepsis affects 20–30% of ICU patients — far higher than the 1.5% churn rate I dealt with in Part 1. But the same principle applies: standard cross-entropy will down-weight hard examples and over-prioritise easy non-sepsis cases. Focal Loss forces the model to attend to the difficult boundary cases that matter most clinically.

Decision threshold: I set the default at **0.40, not 0.50**. This is deliberately conservative. A false negative (missed sepsis case) means a patient does not receive early antibiotics and their mortality risk increases by approximately 7% per hour of delay. A false positive means a nurse reviews a patient who turns out to be fine — a few minutes of wasted time. The asymmetry between these costs justifies an asymmetric threshold.

---

## Task 6 — Evaluation Plan

### Technical metrics

I chose these metrics in order of priority for this specific problem:

**Primary: Sensitivity (Recall) ≥ 0.80 on sepsis class**
This is the metric I would be fired for if it fell below threshold. Missing a sepsis case is not an acceptable failure mode. The entire design of this system — the conservative threshold, the federated training, the continuous monitoring — exists to protect this number.

**Secondary: AUROC ≥ 0.85**
Current SOFA scores achieve approximately 0.74 AUROC on sepsis prediction. If this model does not beat SOFA by a meaningful margin, deploying it adds cost and risk without clinical benefit. 0.85 represents a 15% improvement in discrimination — enough to justify the operational change. I simulated this comparison in the notebook (cell 19) to give the evaluator a visual sense of the difference.

**Tertiary: Specificity ≥ 0.70 and PPV ≥ 0.50**
Alert fatigue is real and dangerous. If nurses learn that most AI alerts are false alarms, they will start ignoring them — which means the model produces zero benefit while adding noise. Specificity and PPV together ensure the alert rate stays manageable.

**Additional: Mean lead time ≥ 6 hours**
An alert that fires 30 minutes before clinical deterioration provides no actionable window. Six hours is the minimum needed to draw cultures, initiate antibiotics, and titrate fluids before organ dysfunction becomes irreversible. I would measure this on a held-out validation set as the average time between alert and confirmed sepsis onset.

**Calibration: Brier Score < 0.10**
A model that outputs probabilities must output *accurate* probabilities. A score of 0.72 should correspond to roughly 72% of patients at that score level actually developing sepsis. Platt scaling calibration is applied post-training.

### Business metrics (from KPI reference data)

Using the actual numbers from `business_kpi_sample.csv` in the notebook (cell 10):

| Metric | KPI Baseline | Target at 12 months | Calculation basis |
|---|---|---|---|
| Avg resolution time | 30.3 hours | 13.6 hours (−55%) | 6-hour lead time × faster treatment initiation |
| Clinical error rate | 6.96% | 2.78% (−60%) | AI flags what humans miss under fatigue |
| Manual processing hours | 454 h/month | 159 h/month (−65%) | Automated triage reduces manual chart review |
| Patient satisfaction | 7.04/10 | 8.75/10 (+24%) | Faster treatment, better communication |
| Estimated monthly saving | — | ~£13,275 | 295 hours × £45/hr (NHS avg band 6) |

### Failure cases and response

**Silent performance degradation:** The model trains on 2024 patient data. By 2026 a new pathogen variant changes the presenting clinical picture of sepsis. AUROC drops gradually from 0.87 to 0.81 without any obvious breakpoint. This is the most dangerous failure mode because it is invisible. Mitigation: monthly AUROC tracking on a held-out prospective cohort with automatic alert to the clinical informatics team if it drops below 0.82.

**Demographic performance gap:** The model works well overall but performs worse for elderly female patients because they were underrepresented in MIMIC-III. Mitigation: pre-deployment audit by age group (under 40 / 40–65 / over 65), sex, and ethnicity. Any subgroup with AUROC below 0.80 triggers targeted data collection before deployment proceeds.

**Alert fatigue cascade:** Alert rate climbs above 40 false positives per 100 patients per day. Staff begin routinely dismissing alerts. One missed sepsis case results in patient death. Full post-market incident report required under UK MDR Class IIb. Mitigation: the alert fatigue monitoring system (Layer 4 of the architecture) automatically tightens the threshold if the false positive rate exceeds a configurable limit.

---

## Task 7 — Responsible AI Considerations

I designed this with the assumption it would be reviewed by a clinical ethics board and submitted to the MHRA as a Class IIb Medical Device. That framing changes how you think about every risk.

### Training data bias

MIMIC-III was collected at a single Boston academic medical centre, predominantly serving an urban adult population. A model trained solely on this data may perform worse on elderly rural patients, non-English-speaking patients, or patients with atypical presentations of sepsis that are more common in specific demographics.

My mitigation is federated learning — the model fine-tunes on each deploying hospital's own patient population, which corrects for local demographic differences without requiring centralisation of sensitive data. Before any deployment, I would run a stratified performance audit by age, sex, ethnicity, and primary language. Any subgroup with AUROC below 0.80 is a mandatory hold on deployment for that subgroup.

### False negatives

A missed sepsis case — the model scores a deteriorating patient as low risk — may delay antibiotics by hours and increase their mortality risk. I treat this as the primary risk category.

Mitigations: conservative threshold (0.40 not 0.50), mandatory human review of all elevated and critical alerts before any clinical action, and a monthly 5% random audit of low-risk predictions to catch systematic misses. The 5% audit is not cosmetic — it is specifically designed to maintain clinical engagement with the model's output and prevent the "automation bias" failure mode where staff stop thinking critically because the AI said the patient was fine.

### Privacy and data governance

Clinical notes contain names, dates of birth, diagnosis codes, names of family members, and sometimes implant serial numbers that could re-identify patients even after standard de-identification. Notes are processed by Bio_ClinicalBERT and converted to 32-dimensional embedding vectors before they enter the inference pipeline. The model never sees or stores raw note text at inference time. De-identification is handled by Microsoft Presidio with custom clinical NLP rules, and 500 randomly sampled de-identified notes are manually reviewed monthly.

The system must comply with: UK GDPR, NHS Data Security and Protection Toolkit, and — for hospitals using the US deployment — HIPAA. Model weights are stored on encrypted infrastructure with access logging. The federated architecture means patient data never leaves the hospital's own network.

### Over-reliance and automation bias

After six months of strong performance, clinical staff naturally begin to trust the system. The risk is that when it fails — and it will occasionally fail — the human safety net has eroded because people stopped paying attention to patients the AI rated as low-risk.

My mitigation is the 5% normal audit. Every month, a random 5% sample of cases where the AI predicted low risk is retrospectively reviewed by a senior clinician. This serves two functions: it catches silent degradation, and it keeps clinical staff actively engaged with the question "was the AI right here?" rather than passive consumers of AI output. Staff training sessions every six months reinforce that this is decision support, not decision making.

The EHR interface labels all AI outputs explicitly: "AI-assisted risk score — requires clinical judgment." The one-click disagree button logs every override with a free-text reason field. These overrides are reviewed monthly — both to improve the model and to identify staff members who might need additional training on how to interpret AI outputs.

### Regulatory compliance

In the UK, a system that provides decision support influencing clinical treatment decisions is a **Class IIb Medical Device** under UK MDR 2002. This requires:
- Clinical evaluation demonstrating equivalence or superiority to a predicate device
- Technical file including algorithm description, training data summary, and validation studies
- Post-market clinical follow-up with mandatory incident reporting
- ISO 13485 quality management system for software development
- ISO 14971 risk management process (the risk matrix I produced in the notebook follows this framework)
- An Algorithm Change Protocol — any significant model update triggers a re-evaluation process before it goes live

I built the ISO 14971 risk matrix in the notebook (cell 21) plotting each risk by likelihood and severity. The two highest-severity risks — false negatives on sepsis and privacy breach — are both low-likelihood because of the mitigations described above.

---

## Task 8 — Final Solution Summary

### The problem in one sentence
Hospitals cannot detect sepsis early enough because the signal is distributed across 40 data streams that no human can continuously integrate across four simultaneously managed patients.

### The proposed solution
A continuous multimodal early warning system that monitors every ICU patient's vital signs, lab results, and nursing notes in real time and generates a rolling risk score every 15 minutes. When the score crosses 0.40, a soft notification appears in the clinician's EHR view alongside the top five contributing clinical signals. The system provides a 6–12 hour head start over current SOFA-based detection.

### Required data
- 50,000+ labelled patient-stay windows (minimum 10,000 sepsis-positive)
- Three streams: vital signs (6 variables, 1-hour bins, 48h look-back), lab results (14 variables, forward-filled), nursing notes (de-identified, ClinicalBERT-encoded to 32d)
- Sources: MIMIC-III, eICU, local hospital PACS, prospective labelling pipeline
- Labels: Sepsis-3 criteria applied retrospectively, 10% manual clinical validation

### Model
Temporal Fusion Transformer with Bio_ClinicalBERT note encoder. Two-phase training: pre-train on public datasets, federated fine-tune at each hospital. Focal Loss (γ=2). Platt scaling calibration. Output: P10/P50/P90 quantile risk estimates.

### Expected business impact

| Metric | Current baseline | Target at 12 months |
|---|---|---|
| Avg resolution time | 30.3 hours | 13.6 hours (−55%) |
| Clinical error rate | 6.96% | 2.78% (−60%) |
| Manual processing hours | 454 h/month | 159 h/month (−65%) |
| Patient satisfaction | 7.04/10 | 8.75/10 (+24%) |
| Monthly staff time saving | — | ~£13,275 |

### Risks and mitigation

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Training data bias | Medium | High | Federated fine-tuning + stratified demographic audit |
| False negative (missed sepsis) | Low | Critical | Conservative threshold (0.40) + mandatory human review |
| Privacy breach | Low | Critical | De-identification pipeline + embedding-only inference + GDPR |
| Alert fatigue | Medium | High | Alert rate monitoring + auto-threshold adjustment |
| Automation bias | Medium | High | Monthly 5% normal audit + staff training |
| Distribution shift | High (long-term) | High | Quarterly revalidation + AUROC alert at 0.82 |
| Regulatory non-compliance | Low | High | UK MDR Class IIb process + ISO 13485 + ISO 14971 |

### The principle I kept coming back to throughout this design

The AI is decision support, not a replacement. Every alert requires a human to look at the patient and decide what to do. The model reorders the question of which patient to look at first — it does not answer the clinical question of what is wrong or what to do about it. That distinction is not just a legal caveat. It is the correct way to deploy AI in a domain where the cost of getting it wrong is irreversible.

---

*Full analysis, visualisations, KPI calculations, simulated evaluation curves, and ISO 14971 risk matrix are in `notebook.ipynb`. Architecture diagram is in `diagrams/solution_architecture.png` (also available as importable draw.io XML).*
