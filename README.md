# Part 4 — AI Solution Design for a Business Problem

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Domain](https://img.shields.io/badge/Domain-Healthcare-red)
![Model](https://img.shields.io/badge/Model-Temporal%20Fusion%20Transformer-blueviolet)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-orange)


---

## Problem Statement

This part is a business analyst exercise — designing an AI solution from the ground up, justifying every decision, and thinking through what could go wrong before it does.

**Domain:** Healthcare — Intensive Care Units (ICU)  
**Problem:** Early Sepsis Detection using Multimodal Deep Learning

Sepsis kills approximately 11 million people annually and is the leading cause of in-hospital deaths worldwide. Current detection methods (SOFA, qSOFA) are periodic, threshold-based, and check for sepsis only 3–6 times per day. By the time a clinician recognises the pattern, the optimal intervention window has often already closed. The proposed AI system watches patient data continuously and flags deterioration **6–12 hours before clinical presentation**, giving the care team time to act.

**Reference Data Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs
`ai_usecase_reference_catalog.csv`:-https://drive.google.com/file/d/1lMXyGp0UaoyhnU4SzWnTDIOVV1VDoq7N/view?usp=share_link
`business_kpi_sample.csv`:-https://drive.google.com/file/d/1dPoRjnTcysR4lhYHhSu7p10nghSYtAox/view?usp=share_link

`data_dictionary.md`:- https://drive.google.com/file/d/1XJOc-4VBJiQKJkKpEZ6U1321kGO_KGiE/view?usp=share_link

> Reference files (`ai_usecase_reference_catalog.csv` and `business_kpi_sample.csv`) are **not stored in this repository** per project instructions. The notebook loads them directly from Google Drive into memory at runtime.

---



---

## Reference Data Used

Two CSV files were provided as design support material and loaded directly from Google Drive:

| File | Purpose | Key insight extracted |
|---|---|---|
| `ai_usecase_reference_catalog.csv` | 10 AI use cases across domains with recommended models, metrics, risks | Healthcare row suggests CNN for image triage — extended to TFT for ICU time-series |
| `business_kpi_sample.csv` | 12 months of operational KPI data — baseline performance before AI | Used to anchor every business impact claim to a real number |

**KPI baseline (from `business_kpi_sample.csv`):**

| Metric | Current Baseline | CV | Note |
|---|---|---|---|
| Avg resolution time | 30.3 hours | 24% | High variance = inconsistency is also a problem |
| Clinical error rate | 6.96% | 26% | High variance = unpredictable quality |
| Manual processing hours | 454 h/month | 16% | |
| Patient satisfaction | 7.04 / 10 | — | |

The high coefficient of variation on resolution time and error rate was an important finding — the AI system needs to reduce variance, not just the mean.

---

## Solution Overview

| Component | Decision | Reason |
|---|---|---|
| **Domain** | Healthcare — ICU | Highest-stakes, clearest asymmetric cost of being wrong |
| **AI Task** | Binary Classification + Sequence Prediction | Each 15-min step answers: will this patient develop sepsis in the next 6–12 hours? |
| **Model** | Temporal Fusion Transformer (TFT) + Bio_ClinicalBERT | Designed for mixed-frequency multimodal inputs; interpretable attention weights |
| **Input** | Vitals (6 features × 48 steps) + Labs (14 features) + Clinical notes (32d embed) + Static context | Three concurrent data streams — no existing EWS integrates all three |
| **Output** | P(sepsis onset < 12h) — updated every 15 minutes | Rolling risk score with configurable threshold |
| **Loss** | Focal Loss (γ=2) | 20–30% sepsis prevalence — same principle as class weights in Part 1 |
| **Threshold** | 0.40 not 0.50 | False negative (missed sepsis) costs a life; false positive costs a few minutes of nurse time — asymmetric |

---

## Notebook Structure

| Cell | Task | Content |
|---|---|---|
| 0–1 | Setup | Imports, reasoning for approach |
| 2–5 | Reference Data | Load both CSVs from Drive, explore catalog and KPI data |
| 6 | Baseline KPIs | 4-panel chart with CV annotations — saves `baseline_kpis.png` |
| 7–8 | Task 1 | Domain selection, comparison scatter (impact vs difficulty) |
| 9–11 | Task 2 | Problem definition, before/after KPI chart, sepsis detection timeline |
| 12–13 | Task 3 | AI task type, ICU time-series visualisation (septic vs stable) |
| 14–15 | Task 4 | Data requirement table, multimodal input schema diagram |
| 16–17 | Task 5 | Model recommendation, TFT architecture diagram |
| 18–19 | Task 6 | Evaluation plan, ROC + precision-recall curves (TFT vs SOFA baseline) |
| 20–21 | Task 7 | Responsible AI, ISO 14971 risk matrix |
| 22–23 | Task 8 | Final summary, 3-panel dashboard |

---

## Task 1 — Domain Selection: Healthcare

I chose healthcare over the other 9 domains in the catalog for one specific reason: healthcare is the only domain where the AI model's false negative rate is measured in human lives, not revenue. That asymmetry changes every decision — how the threshold is set, which metrics are optimised for, and how responsible AI is framed.

Within healthcare, I extended the catalog's default suggestion (CNN for medical image triage) to something harder and more impactful: **ICU early sepsis detection from multimodal time-series data**. Image triage is a largely solved problem. Continuous multimodal sepsis prediction is not.

---

## Task 2 — Business Problem

**Current process limitations:**
1. SOFA scored every 4–8 hours — sepsis can turn critical in under 2 hours
2. Information overload — a nurse managing 4 patients cannot continuously monitor 40 variables each
3. Retrospective not predictive — SOFA reflects current dysfunction, not trajectory toward it
4. Nursing notes never integrated into structured scoring despite carrying early sentinel observations
5. 26% CV on error rate means performance varies dramatically with staffing and fatigue

**Proposed solution:** Continuous monitoring system that reads vital signs, lab results, and nursing notes every 15 minutes and generates a rolling risk score. When the score crosses 0.40, a soft alert appears in the clinician's EHR view alongside the top 5 contributing clinical signals.

---

## Task 3 — AI Task Type

**Selected: Binary Classification + Sequence Prediction**

At each 15-minute step the model answers: *"Will this patient develop sepsis in the next 6–12 hours?"*

The input is a 48-hour window of multivariate time-series observations. The model must understand temporal dependencies — a lactate of 2.1 mmol/L rising for 8 hours is far more dangerous than a stable 2.1. This is why a static classifier on snapshot features is insufficient, and why the ICU time-series visualisation (cell 13) shows two patients with similar values at hour 24 but completely different trajectories from hours 0–24.

---

## Task 4 — Data Requirements

**Three input streams:**

| Stream | Features | Format | Frequency |
|---|---|---|---|
| Vital signs | HR · BP · MAP · SpO₂ · Resp rate · Temp | Time series (6 × 48) | Every 1–5 min |
| Lab values | WBC · Lactate · Creatinine · Bilirubin · Platelets · CRP + 8 more | Episodic, forward-filled (14 × 48) | Every 4–24 hours |
| Clinical notes | Nursing shift notes, medication changes | Bio_ClinicalBERT → 32-dim | 1–4 per shift |

**Target:** `sepsis_onset_within_12h` — binary, derived retrospectively using Sepsis-3 criteria with 10% manual clinical validation.

**Minimum viable dataset:** 50,000 labelled patient-stay windows with at least 10,000 sepsis-positive cases.

---

## Task 5 — Model Recommendation

**Temporal Fusion Transformer (TFT) + Bio_ClinicalBERT**

Four reasons specific to this problem:

1. **Designed for this data structure** — TFT handles mixed-frequency inputs natively with separate pathways for static covariates, temporal observations, and known-future inputs. ICU data has all three.
2. **Variable Selection Networks (VSNs)** — learn feature importance dynamically per patient per time step. Not fixed weights across all cases.
3. **Attention weights are clinically interpretable** — when the model fires an alert, the nurse sees: *"lactate at t-8h and rising HR at t-4h contributed 63% of this score."* This is interpretability by architectural design, not post-hoc explanation. It is a regulatory requirement, not a nice-to-have.
4. **Quantile outputs (P10/P50/P90)** — gives clinicians an uncertainty interval, not just a point estimate.

**Training:** Phase 1 pre-train on MIMIC-III + eICU (~250,000 stays). Phase 2 federated fine-tuning at each hospital — only weight updates shared, never raw patient data.

---

## Task 6 — Evaluation Plan

**Technical metrics (priority order):**

| Metric | Target | Why |
|---|---|---|
| **Sensitivity (Recall)** | ≥ 0.80 | Missing a sepsis case costs a life — primary metric |
| **AUROC** | ≥ 0.85 | SOFA achieves ~0.74 — must beat it meaningfully |
| **Specificity** | ≥ 0.70 | Too many false alarms → alert fatigue → staff stop trusting it |
| **PPV (Precision)** | ≥ 0.50 | At least 1 in 2 alerts must be real |
| **Lead time** | ≥ 6 hours | Alerts with < 6h lead time provide no actionable window |

**Business impact targets (from KPI reference data):**

| Metric | Baseline | Target | Change |
|---|---|---|---|
| Avg resolution time | 30.3 hours | 13.6 hours | −55% |
| Clinical error rate | 6.96% | 2.78% | −60% |
| Manual processing hours | 454 h/month | 159 h/month | −65% |
| Patient satisfaction | 7.04/10 | 8.75/10 | +24% |
| Monthly staff saving | — | ~$13,275 | 295 hrs × $45/hr |

---

## Task 7 — Responsible AI

Designed assuming review by a clinical ethics board and MHRA/FDA submission as a Class IIb Medical Device.

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Training data bias | Medium | High | Federated multi-site training + stratified demographic audit |
| False negative (missed sepsis) | Low | Critical | Conservative threshold 0.40 + mandatory human review |
| Privacy breach | Low | Critical | Full DICOM de-identification, GDPR/HIPAA, embedding-only inference |
| Alert fatigue | Medium | High | Weekly alert rate monitoring + auto-threshold tightening |
| Automation bias | Medium | High | Monthly 5% normal audit + quarterly staff training |
| Distribution shift | High (long-term) | High | Quarterly revalidation + AUROC alert at 0.82 |
| Regulatory non-compliance | Low | High | UK MDR Class IIb + ISO 13485 + ISO 14971 + post-market surveillance |

---

## Task 8 — Final Solution Summary

| Component | Detail |
|---|---|
| **Problem** | Sepsis kills 11M/year. Current SOFA detection is periodic and retrospective — alert arrives too late. |
| **Solution** | Continuous TFT-based risk scoring every 15 minutes across 3 data streams. 6–12h early detection. |
| **Data** | 50k+ labelled ICU stays · 3 streams · Sepsis-3 labels · Federated privacy architecture |
| **Model** | TFT + Bio_ClinicalBERT · Focal Loss · Threshold 0.40 · Quantile uncertainty outputs |
| **AUROC target** | 0.87 vs SOFA baseline 0.74 (+18% discrimination) |
| **Core principle** | The AI reorders the worklist — it never diagnoses, never acts, never replaces clinical judgment |

---

## Key Design Principle

> The AI is decision support, not a replacement. Every alert requires a human to review the patient and decide what to do. The model answers which patient to look at first — it does not answer what is wrong or what to do about it. That distinction is not a legal caveat. It is the correct way to deploy AI in a domain where the cost of getting it wrong is irreversible.

---
## Note

The notebook loads both reference CSVs from Google Drive on the first data cell. If Drive is unavailable, place `ai_usecase_reference_catalog.csv` and `business_kpi_sample.csv` in the same folder as the notebook and rerun.

---



## References

- Rajpurkar et al. (2022). *AI in Health and Medicine*. Nature Medicine.
- Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting.
- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings* (Bio_ClinicalBERT). NAACL.
- Singer et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)*. JAMA.
- Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database*. Scientific Data.
- ISO 14971:2019 — Application of risk management to medical devices.


