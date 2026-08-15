# Dataset Card

## Identification

| Field | Value |
|---|---|
| Name | Diabetes 130-US Hospitals for Years 1999–2008 |
| Repository | UCI Machine Learning Repository, dataset 296 |
| URL | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| Licence | CC BY 4.0 |
| Download size | 3,170,254 bytes (SHA-256 `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a`) |
| Raw shape | 101,766 encounters × 50 columns |
| Unique patients | 71,518 |
| Collection period | 1999–2008 |
| Coverage | 130 hospitals and integrated delivery networks across the United States |

## Source publication

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J.,
& Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission
rates: Analysis of 70,000 clinical database patient records. *BioMed Research
International*, 2014, 781670. https://doi.org/10.1155/2014/781670

The dataset was extracted from the Cerner Health Facts database and released
alongside the paper above. It has since become a standard benchmark for
readmission modelling and for fairness-in-healthcare research, including
inclusion in Fairlearn's dataset collection.

## Why this dataset was selected

The assessment requires demonstrable handling of missing values, duplicates,
invalid data, outliers, data types, categorical encoding and class imbalance,
plus a defensible fairness analysis. Most public health datasets are pre-cleaned
and force you to argue why half that checklist is not applicable. This one is
not:

| Requirement | What this dataset actually contains |
|---|---|
| Missing values | Genuine `?` sentinels at four very different rates: `weight` 96.9%, `medical_specialty` 49.1%, `payer_code` 39.6%, `race` 2.2% — each needing a *different* policy |
| Duplicates | 101,766 encounters from 71,518 patients; one patient appears 40 times. A real leakage trap, not a synthetic one |
| Invalid data | `gender = 'Unknown/Invalid'`; discharge codes for died/hospice that make the outcome impossible |
| Outliers | Heavy right tails in prior visit counts and lab procedures, belonging to real high-utilising patients |
| Data types | Three nominal fields encoded as integers, which every model would misread as ordered |
| Categorical encoding | 30+ nominal columns, one with 71 levels, plus ICD-9 codes needing clinical grouping |
| Class imbalance | 11.2% positive before de-duplication, 9.0% after |
| Fairness | Four sensitive attributes: race, gender, age band, payer group |

Alternatives considered and rejected: the UCI Maternal Health Risk set (1,014
rows, no missingness, no categoricals), the Beijing Multi-Site Air Quality set
(strong on missingness and temporal splitting but has no demographic attributes
for the fairness requirement), and the UCI Credit Card Default set (excellent
for fairness but only disguised missingness and a weaker fit to a health or
environment mission).

## Composition

**Unit of analysis.** One row is one hospital *encounter*. After cleaning it
becomes one row per *patient*, using each patient's first encounter. This
follows Strack et al. and is what makes the train/test split honest.

**Inclusion criteria applied by the original authors:** the encounter was an
inpatient admission, the patient was diabetic, the stay lasted 1–14 days, lab
tests were performed, and medications were administered.

**Target.** `readmitted` takes three values: `<30`, `>30`, `NO`. This project
binarises to `readmit_lt30 = (readmitted == '<30')`, because the 30-day window
is the one that carries clinical urgency and CMS financial penalty. A
readmission at day 200 is a different phenomenon with different drivers.

| Split | Rows | Positives | Positive rate |
|---|---|---|---|
| Raw (all encounters) | 101,766 | 11,357 | 11.16% |
| After cleaning (one row per patient) | 69,987 | 6,285 | 8.98% |

The drop from 11.16% to 8.98% is itself a finding: repeat encounters are
disproportionately readmissions, so treating encounters as independent inflates
both the positive rate and any model score computed on a random split.

## Feature groups

- **Demographics:** race, gender, age (10-year bands)
- **Admission context:** admission type, admission source, discharge disposition
  (all supplied as integer codes with a separate `IDS_mapping.csv` lookup)
- **Utilisation history:** prior outpatient, emergency and inpatient visits
- **This stay:** length of stay, lab procedures, procedures, medications,
  number of diagnoses
- **Diagnoses:** `diag_1`, `diag_2`, `diag_3` as ICD-9 codes
- **Glycaemic testing:** `A1Cresult`, `max_glu_serum` (where `None` means the
  test was not ordered, a clinical decision rather than a missing value)
- **Medications:** 23 drug columns valued `No` / `Steady` / `Up` / `Down`
- **Payer:** `payer_code`

## Known limitations

1. **Out-of-network readmissions are invisible.** A patient readmitted to a
   different hospital appears as a negative. This biases the label against
   mobile and uninsured patients. See `docs/ETHICS.md` §3.2.
2. **Age is banded, not exact**, limiting fine-grained age modelling.
3. **`weight` is 96.9% missing** and unusable as a value.
4. **No social determinants.** Housing, food security, transport and caregiver
   support drive readmission and none are recorded. This is the main reason
   published AUCs on this dataset plateau in the 0.63–0.72 range.
5. **Vintage.** 1999–2008. Clinical practice and coding have moved on.
6. **US-specific.** Payer structure and discharge pathways do not transfer to
   Indian clinics without local revalidation.

## Ethical and legal notes

De-identified under HIPAA Safe Harbor prior to public release. No PII. No
re-identification attempted. Redistribution permitted under CC BY 4.0 with
attribution, which this project provides in the README, the notebook, and here.
