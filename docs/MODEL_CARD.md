# Model Card — 30-Day Readmission Risk Triage

## Model details

| Field | Value |
|---|---|
| Task | Binary classification: readmission within 30 days of discharge |
| Architecture | Heterogeneous stacking ensemble |
| Base learners | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Meta-learner | Logistic Regression (`class_weight='balanced'`) |
| Meta-leakage control | 5-fold stratified internal CV (`StackingClassifier(cv=StratifiedKFold(5))`) |
| Imbalance handling | Class weights + `scale_pos_weight`; **no** synthetic resampling |
| Preprocessing | Median imputation → winsorisation (p0.5/p99.5) → standardisation; one-hot with `min_frequency=0.01` |
| Deployed threshold | **0.655**, chosen on validation by cost minimisation subject to a 10% follow-up capacity constraint |
| Abstention band | 0.40–0.60 → escalate to human review |
| Seed | 42 |
| Artefact | `models/best_pipeline.joblib` |
| Reproduce | `python -m src.train` |

## Training data

- 69,987 patients (one first encounter each) from the UCI Diabetes 130-US Hospitals dataset.
- Split 60/20/20 stratified: 41,991 train / 13,998 validation / 13,998 test.
- No patient appears in more than one split (asserted in `src/pipeline.py`).
- Positive rate 8.98% in every split.

## Performance on the untouched test set

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Lift@10% |
|---|---|---|---|---|---|---|
| **Stacking (deployed)** | **0.6574** | 0.1772 | 0.2315 | 0.2241 | 0.2395 | 2.44× |
| Random Forest | 0.6559 | 0.1717 | 0.2207 | 0.2149 | 0.2267 | 2.38× |
| XGBoost | 0.6553 | 0.1769 | 0.2363 | 0.2254 | 0.2482 | 2.50× |
| LightGBM | 0.6545 | 0.1788 | 0.2358 | 0.2316 | 0.2403 | 2.51× |
| Soft Voting | 0.6528 | 0.1741 | 0.2259 | 0.2165 | 0.2363 | 2.38× |
| Logistic Regression (baseline) | 0.6467 | 0.1663 | 0.2292 | 0.2219 | 0.2371 | 2.45× |
| Bagged Trees | 0.6391 | 0.1718 | 0.2221 | 0.2133 | 0.2315 | 2.34× |
| Decision Tree (baseline) | 0.6254 | 0.1523 | 0.2144 | 0.2370 | 0.1957 | 2.44× |

**Ensemble vs baseline:** +0.0107 ROC-AUC, paired bootstrap 95% CI [+0.0043, +0.0171] over 2,000
resamples. The interval excludes zero, so the gain is statistically real — and small.

**Confusion matrix at threshold 0.655:** TP 301 · FP 1,042 · FN 956 · TN 11,699.

**Operational meaning.** At 10% follow-up capacity the model reaches 21.9% precision against a
9.0% base rate: about **2.4× more genuinely at-risk patients per nurse-hour** than calling at random.

## Where the model fails

### Chronic high-utilisers
On patients with prior encounters in the training data, ROC-AUC falls to **0.566** — barely better
than chance — versus 0.687 on first-admission patients. The model's competence is concentrated on
patients it has never seen before. Repeat admissions are driven by instability that a discharge
record does not capture.

This matters because chronic high-utilisers are arguably the group with greatest need.

### Fairness

| Attribute | Disparate impact ratio | Passes 4/5ths rule | AUC gap |
|---|---|---|---|
| Gender | 0.945 | Yes | 0.015 |
| Race | 0.541 | **No** | 0.101 |
| Payer group | 0.265 | **No** | 0.074 |
| Age band | 0.164 | **No** | 0.183 |

**The finding that blocks deployment:** African American patients have a readmission rate
statistically indistinguishable from Caucasian patients (**9.23% vs 9.07%**) but are flagged for
follow-up substantially less often (**selection rate 8.0% vs 10.2%**, recall **19.9% vs 25.0%**).

Equal clinical need, unequal service. This is not explained by differing risk, and it is not
acceptable in a resource-allocation tool.

Payer and age disparities are partly attributable to genuine risk differences (Commercial-payer
patients really do have a 6.1% readmission rate versus 10.0% for the unrecorded-payer group), so
those ratios overstate the injustice. Race has no such explanation.

### Label bias the metrics cannot see
Only readmissions to the same hospital network are recorded. Patients who present elsewhere —
disproportionately uninsured and mobile — are labelled negative. Every fairness number above is
computed against these biased labels, so the true disparity is likely worse than measured.

## Intended use

**In scope:** ranked worklist for nurse-led follow-up calls; prompting HbA1c ordering; flagging
medication reconciliation candidates.

**Out of scope, permanently:** denying/delaying/shortening care; insurance pricing or underwriting;
discharge decisions; staff performance evaluation; any patient-facing score without a clinician present.

## Deployment status

**Not deployment-ready.** The race-based allocative disparity in the fairness section must be
addressed before any clinical use. The audit was not a formality — it produced a blocking result.

Additional prerequisites:
- Refit on contemporary local data (this model is trained on 1999–2008 US records)
- Prospective validation in the target setting
- Quarterly drift monitoring and subgroup re-audit with a pre-agreed withdrawal threshold
- Mandatory, non-punitive clinician override

## Ethical considerations

See [`docs/ETHICS.md`](ETHICS.md) for the full statement covering privacy, false-positive and
false-negative costs, uncertainty, human oversight, and prohibited uses.

## Citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N.
(2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical
database patient records. *BioMed Research International*, 2014, 781670.
https://doi.org/10.1155/2014/781670
