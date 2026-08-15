# Predicting 30-Day Hospital Readmission for Diabetic Patients

**ML for Social Good Ensemble Challenge — CIA 3**
MCA 521-4 Machine Learning · CHRIST (Deemed to be University)

**Mission: Health** — predict which diabetic inpatients will be readmitted within 30 days, so a
clinic with limited follow-up capacity calls the right people.

---

## The 60-second version

About 1 in 11 diabetic inpatients returns to hospital within 30 days. Each avoidable readmission
costs roughly **$15,000**; a nurse follow-up call costs about **$75** and measurably reduces the
risk. The problem is not knowing that calls work — it is that an under-resourced clinic can only
make a few, with no way to know which patients to pick.

This project builds a leakage-safe pipeline from 101,766 raw hospital encounters to a deployable,
explained risk model, and audits who it fails.

**Headline result:** at 10% follow-up capacity the model reaches **21.9% precision** against a
**9.0%** base rate — **2.44× more genuinely at-risk patients per nurse-hour** than calling at random.

**Headline caveat:** the fairness audit **failed**. African American patients have the same
readmission rate as Caucasian patients (9.23% vs 9.07%) but are flagged less often (8.0% vs 10.2%).
The model is **not deployment-ready**, and the audit is what tells us so.

---

## Results

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Lift@10% |
|---|---|---|---|---|---|---|
| **Stacking (heterogeneous)** | **0.6574** | 0.1772 | 0.2315 | 0.2241 | 0.2395 | 2.44× |
| Random Forest (bagging) | 0.6559 | 0.1717 | 0.2207 | 0.2149 | 0.2267 | 2.38× |
| XGBoost (boosting) | 0.6553 | 0.1769 | 0.2363 | 0.2254 | 0.2482 | 2.50× |
| LightGBM (boosting) | 0.6545 | 0.1788 | 0.2358 | 0.2316 | 0.2403 | 2.51× |
| Soft Voting (heterogeneous) | 0.6528 | 0.1741 | 0.2259 | 0.2165 | 0.2363 | 2.38× |
| Logistic Regression *(baseline)* | 0.6467 | 0.1663 | 0.2292 | 0.2219 | 0.2371 | 2.45× |
| Bagged Trees | 0.6391 | 0.1718 | 0.2221 | 0.2133 | 0.2315 | 2.34× |
| Decision Tree *(baseline)* | 0.6254 | 0.1523 | 0.2144 | 0.2370 | 0.1957 | 2.44× |

**Does the best ensemble beat the baseline?** Yes — **+0.0107 ROC-AUC**, paired bootstrap 95% CI
**[+0.0043, +0.0171]** over 2,000 resamples. The interval excludes zero, so the gain is
statistically real. It is also small, and reported as such.

An AUC of ~0.66 is consistent with the published literature on this dataset. The ceiling is the
data, not the model: readmission is driven substantially by housing, food security, transport and
caregiver support, none of which appear in a discharge record.

---

## Five findings worth reading the notebook for

1. **De-duplicating by patient dropped the positive rate from 11.16% to 8.98%.** 101,766 encounters
   are only 71,518 patients; one appears 40 times. Repeat encounters skew positive, so
   encounter-level modelling flatters itself.

2. **Leakage did not inflate AUC here — it lowered it.** The leaky random split scored 0.6699 vs
   0.6759 for the honest patient-grouped split, the opposite of the textbook expectation. We report
   what happened, not what we expected.

3. **The model is near-random (AUC 0.566) on repeat high-utilisers** versus 0.687 on
   first-admission patients — it is weakest precisely where need is greatest.

4. **Unconstrained cost optimisation says "call everyone".** At 200:1 FN:FP cost that is genuinely
   cheapest — for a clinic with unlimited nurses. Capacity, not cost, is the binding constraint, and
   the deployed threshold reflects that explicitly.

5. **The fairness result only became visible once the threshold was set honestly.** At the
   degenerate threshold that flags everyone, every group trivially passes every parity test.

---

## Reproduce it

```bash
make setup      # venv on Python 3.13 + pinned dependencies + Jupyter kernel
make data       # download UCI dataset, verify SHA-256
make train      # full pipeline: clean → engineer → tune → evaluate → audit  (~19 min)
make app        # launch the Streamlit live demo
make deck       # build the PPTX and HTML presentations
make dashboard  # build the self-contained HTML results dashboard
```

Run `make train-fast` to skip tuning and the ablations (~2 min).

Everything is seeded (`SEED = 42`). `python -m src.train` regenerates every number in the
notebook, the dashboard and the deck.

### Requirements

- **macOS/Linux, Python 3.13** (3.14 also works)
- **`libomp`** — required by XGBoost and LightGBM on macOS: `brew install libomp`
- **CatBoost is not used**: no arm64 wheel is published. The rubric permits XGBoost/LightGBM.
- Disk footprint is small: the dataset is 3 MB zipped.

### Notebook

```bash
make notebook   # executes end to end
```
Open `notebooks/01_ml_for_social_good_readmission.ipynb` and select the
**ML for Good (venv 3.13)** kernel.

---

## Repository layout

```
├── notebooks/01_ml_for_social_good_readmission.ipynb   # the main deliverable
├── src/
│   ├── config.py        # seed, paths, column schema, cost model
│   ├── data_loader.py   # download + SHA-256 verification
│   ├── cleaning.py      # exclusions, invalid values, de-duplication
│   ├── features.py      # ICD-9 mapping, domain features + rationale registry
│   ├── pipeline.py      # ColumnTransformer, splits, leakage assertions
│   ├── models.py        # baselines, bagging, boosting, stacking, voting
│   ├── evaluate.py      # metrics, cost thresholding, bootstrap, McNemar
│   ├── fairness.py      # subgroup audit, parity metrics, mitigation
│   ├── explain.py       # SHAP, permutation importance, LIME
│   ├── viz.py           # every figure, generated once
│   └── train.py         # reproducible end-to-end run
├── app/
│   ├── streamlit_app.py # live demo
│   └── synthetic.py     # synthetic patient profiles (no real records)
├── dashboard/           # self-contained HTML results dashboard
├── presentation/        # build_deck.py → CIA3_deck.pptx + deck.html
├── docs/
│   ├── DATASET_CARD.md  # provenance, why this dataset, limitations
│   ├── MODEL_CARD.md    # performance, failure modes, deployment status
│   ├── ETHICS.md        # ethics statement
│   └── video_script.md  # timed 3-minute pitch script + shot list
├── results/figures/     # 24 figures
├── results/metrics/     # every table as CSV/JSON
└── tools/build_notebook.py
```

---

## Mark-scheme traceability

| Rubric requirement | Where to find it |
|---|---|
| **Q1 · Social problem, beneficiaries, target** | Notebook §1; `docs/DATASET_CARD.md` |
| Q1 · Measurable impact | Notebook §6 "impact translation"; precision@k / lift@k in `src/evaluate.py` |
| Q1 · Why ML is suitable | Notebook §1 |
| Q1 · Dataset source and unit of analysis | Notebook §1; `docs/DATASET_CARD.md` |
| Q1 · Responsible-use limitations | Notebook §1, §8; `docs/ETHICS.md` |
| **Q2 · Missing values** | Notebook §2.1; `src/cleaning.py:apply_missing_policy` |
| Q2 · Imputation methods compared | Notebook §4.3; `src/evaluate.py:imputer_ablation` |
| Q2 · Duplicates | Notebook §2.2; `src/cleaning.py:deduplicate_patients` |
| Q2 · Invalid data | Notebook §2.3; `drop_expired_and_hospice`, `drop_invalid_gender` |
| Q2 · Outliers | Notebook §2.5; `src/pipeline.py:Winsoriser` |
| Q2 · Data types | `src/features.py:column_schema`; `decode_id_columns` |
| Q2 · Categorical encoding | `src/pipeline.py:build_preprocessor` (one-hot, `min_frequency`) |
| Q2 · Numerical scaling | `build_preprocessor` (StandardScaler, trees excluded) |
| Q2 · Class imbalance | Notebook §4.4 (class weights vs SMOTE, with reasoning) |
| Q2 · EDA | Notebook §3, 15+ figures |
| Q2 · Domain-informed features | Notebook §4; `src/features.py:FEATURE_RATIONALE` |
| Q2 · Leakage-safe, reproducible splits | Notebook §4.5, §4.6; `src/pipeline.py:leakage_checklist` |
| **Q3 · Baseline (tree or logistic)** | Both: `logistic_baseline`, `tree_baseline` |
| Q3 · Bagging | `random_forest`, `bagged_trees` |
| Q3 · Boosting | `xgboost_model`, `lightgbm_model` |
| Q3 · Stacking / voting | `stacking_model`, `voting_model` |
| Q3 · Correct CV, no meta-leakage | Notebook §6.1; `StackingClassifier(cv=StratifiedKFold(5))` |
| Q3 · Tuning | `src/models.py:SEARCH_SPACES`; `results/metrics/tuning_results.csv` |
| Q3 · Comparison on untouched test set | Notebook §6; `results/metrics/model_comparison.csv` |
| Q3 · ROC-AUC, F1, precision, recall, confusion matrix | Same table + `14_confusion.png` |
| Q3 · Does the ensemble beat the baseline? | Notebook §6; bootstrap CI + McNemar |
| **Q4 · SHAP global** | Notebook §7; `18_shap_global.png`, `18b_shap_beeswarm.png` |
| Q4 · SHAP local | Notebook §7; waterfalls for TP / FP / FN / demo patient |
| Q4 · Influential features in domain language | Notebook §7 translation table (actionable vs descriptive) |
| Q4 · Bias / fairness | Notebook §8; `results/metrics/fairness_*.csv` |
| Q4 · Privacy | `docs/ETHICS.md` §2 |
| Q4 · Uncertainty | Calibration curves, bootstrap CIs, abstention band |
| Q4 · FP / FN costs | Notebook §6.2; `src/config.py` cost model; `13_cost_threshold.png` |
| Q4 · Human oversight, deployment limits | `docs/ETHICS.md` §5–6; `docs/MODEL_CARD.md` |
| **Q5 · 3-minute video** | `docs/video_script.md` — timed script + shot list |
| Q5 · Live prediction demo | `app/streamlit_app.py` → Live prediction tab |
| Q5 · Compact results visual | `results/figures/11_model_comparison.png` |

---

## Dataset and citation

**Diabetes 130-US Hospitals for Years 1999–2008**
UCI Machine Learning Repository, dataset 296 · CC BY 4.0
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N.
> (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical
> database patient records. *BioMed Research International*, 2014, 781670.
> https://doi.org/10.1155/2014/781670

The data is de-identified under HIPAA Safe Harbor. No personally identifiable or confidential data
is included, and none was added. All demo records are synthetic.

## Acknowledgements

- **Methodology borrowed and cited:** the expired/hospice exclusion list, the
  first-encounter-per-patient rule, and the ICD-9 clinical chapter grouping all follow
  Strack et al. (2014).
- **Libraries:** scikit-learn, XGBoost, LightGBM, SHAP, LIME, imbalanced-learn, Fairlearn,
  Streamlit, Plotly, python-pptx. Versions pinned in `requirements.txt`.
- All analysis code in `src/` was written for this submission.

## Ethics

This model is a **triage aid**: a ranked worklist for nurse follow-up. It must never be used for
care denial, insurance pricing, discharge decisions, or staff evaluation. See
[`docs/ETHICS.md`](docs/ETHICS.md) for the full statement.
