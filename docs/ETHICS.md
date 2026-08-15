# Ethics Statement

**Project:** 30-day hospital readmission risk for diabetic inpatients
**Course:** MCA 521-4 Machine Learning, CIA 3
**Scope of this document:** what the model may be used for, what it must never
be used for, and the harms it can cause if either boundary is ignored.

---

## 1. Intended use

The model produces a ranked worklist. Given a fixed number of follow-up calls a
discharge team can make in a day, it orders diabetic inpatients by predicted
probability of readmission within 30 days.

**In scope**
- Prioritising nurse-led follow-up calls and medication reconciliation
- Flagging candidates for a pharmacist review of a churning drug regimen
- Prompting a clinician to consider ordering HbA1c before discharge

**Out of scope, permanently**
- Denying, delaying or shortening care for anyone
- Insurance pricing, underwriting, or coverage decisions
- Any patient-facing risk score shown without a clinician present
- Staff performance evaluation or ward-level punishment
- Discharge decisions themselves

The model estimates *who is likely to return*. It does not estimate who
*deserves* care, and the two must never be conflated. A patient with a low score
who asks for help gets help.

---

## 2. Privacy

- The dataset is publicly released by the UCI ML Repository under CC BY 4.0 and
  was de-identified before release under HIPAA Safe Harbor.
- `patient_nbr` is a surrogate key with no link to any identity. It is used only
  to enforce that one patient cannot appear in two data splits, then dropped
  before modelling.
- No re-identification was attempted, and no external data was joined that could
  enable it.
- No personally identifiable or confidential data was added at any stage.
- The live demo generates **synthetic** patient records. No real record is ever
  displayed.

---

## 3. Fairness and bias

### 3.1 What was measured
Subgroup performance (TPR, FPR, PPV, selection rate, AUC) was computed across
race, gender, age band and payer group, together with demographic parity
difference, equalised-odds difference, and the disparate impact ratio against
the four-fifths rule. Results are in `results/metrics/fairness_per_group.csv`
and `fairness_summary.csv`.

### 3.2 Label bias is the deepest problem here
The dataset records readmissions **to the same hospital network**. A patient
readmitted to a different hospital is labelled as *not readmitted*.

That error is not random. Patients who move between providers are
disproportionately uninsured, transient, or reliant on whichever emergency
department is nearest. The label therefore undercounts readmissions precisely
among the most vulnerable, and a model trained on it will systematically
*under-flag* the people who most need follow-up. No amount of threshold tuning
fixes a biased label; only better data collection does.

This is a limitation of the evidence, not a defect that the fairness metrics can
detect — the metrics are computed against the same biased labels.

### 3.3 Missingness as a proxy for disadvantage
`payer_code` is missing for roughly 40% of encounters, and that missingness is
informative: it skews toward self-pay and uninsured patients. It is retained as
an explicit `Unknown` category rather than imputed away, so that the fairness
audit can see it. Using it as a *predictor* is defensible; using it as a reason
to deprioritise anyone is not.

### 3.4 Race is never imputed
Race is missing for ~2.2% of encounters. Imputing it would fabricate the exact
evidence the fairness audit exists to test, so it is kept as `Unknown`.

### 3.5 On per-group thresholds
`src/fairness.py` includes a per-group threshold experiment showing that recall
disparities can be equalised by construction. It is reported as a diagnostic,
**not adopted**. Explicitly conditioning a clinical threshold on race raises
legal and ethical objections in most jurisdictions that a metric improvement
does not settle. The defensible route to equity here is better data and better
features, not a race-indexed cut-off applied silently.

---

## 4. Costs of being wrong

| | Consequence | Who bears it |
|---|---|---|
| **False negative** | A patient who will return gets no follow-up call. Preventable deterioration, an avoidable admission, ~$15,000 of avoidable cost. | The patient, primarily |
| **False positive** | A patient who would not have returned receives a follow-up call, ~$75 and a few minutes of their time. | The clinic, mildly |

The asymmetry is roughly 200:1, which has a counter-intuitive consequence
documented in `results/metrics/threshold_regimes.csv`: unconstrained cost
minimisation says *call every patient*. That is the correct answer for a clinic
with unlimited staff and a useless one for the under-resourced clinics this
project targets.

The deployed threshold therefore minimises expected cost **subject to the
clinic's actual follow-up capacity**. This is an explicit rationing decision and
it should be made by the clinic, in the open, not buried in a default of 0.5.

---

## 5. Uncertainty and human oversight

- **Calibration** is reported (Brier score, reliability curves overall and by
  race). The decision rule compares a probability to a threshold, so a
  miscalibrated probability is a mis-made decision, not just an inelegant one.
- **Confidence intervals**: the improvement over the baseline is reported with a
  paired bootstrap 95% CI and a McNemar test, not as a bare point estimate.
- **Abstention band**: predictions between 0.40 and 0.60 are routed to human
  review rather than auto-actioned.
- **Clinician override is mandatory and unlogged against the clinician.** If
  overrides are used to evaluate staff, clinicians stop overriding, and the
  safeguard becomes decorative.

---

## 6. Limits on deployment

1. **Temporal validity.** The data covers 1999–2008 US hospitals. Diabetes care,
   coding practice and discharge planning have all changed. The model is stale
   for present-day deployment without refitting.
2. **Geographic validity.** It must not be deployed in Indian clinics on the
   strength of this evaluation. Case mix, coding, insurance structure and
   readmission drivers all differ. Prospective local validation is required.
3. **Performance ceiling.** Test ROC-AUC lands near 0.66. This is consistent
   with the published literature on this dataset and reflects a real limit:
   readmission is driven substantially by housing, food security, transport and
   social support, none of which appear in a discharge record. A model cannot
   recover a variable that was never collected.
4. **Monitoring.** Any deployment needs quarterly drift checks and a repeat
   subgroup audit, with a pre-agreed threshold at which the model is withdrawn.
5. **Not a substitute for clinical judgement.** A ranked list is a starting
   point for attention, not a verdict.

---

## 7. Reproducibility and integrity

- All randomness is seeded (`SEED = 42` in `src/config.py`).
- `python -m src.train` regenerates every number reported in the notebook, the
  dashboard and the deck.
- The dataset is downloaded from UCI and its SHA-256 is verified on download.
- Borrowed methodology is cited: the expired/hospice exclusion list and the
  first-encounter-per-patient rule follow Strack et al. (2014). The ICD-9
  chapter grouping follows the same paper.
- Libraries used are listed with pinned versions in `requirements.txt`.

---

## 8. Dataset citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J.,
& Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission
rates: Analysis of 70,000 clinical database patient records. *BioMed Research
International*, 2014, 781670. https://doi.org/10.1155/2014/781670

UCI Machine Learning Repository, dataset 296. Licensed CC BY 4.0.
https://archive.ics.uci.edu/dataset/296/
