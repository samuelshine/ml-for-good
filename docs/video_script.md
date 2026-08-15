# Three-Minute Pitch and Demo — Script and Shot List

**Total runtime: 3:00.** Word counts are calibrated to ~150 words per minute, which is a
comfortable, unhurried pace. Do not rush to fit more in — the timings below already work.

**Before recording**
- [ ] `make train` has been run (artefacts exist in `models/` and `results/`)
- [ ] `streamlit run app/streamlit_app.py` is already open on the **Live prediction** tab
- [ ] Notebook open in a second tab, scrolled to Section 6 (model comparison table)
- [ ] Repo open in a third tab (VS Code or GitHub)
- [ ] Screen resolution 1920×1080, browser zoom 110%, notifications silenced
- [ ] Profile selector on the Streamlit app pre-set to **High risk**

---

## 0:00 – 0:35 · Problem and beneficiaries (87 words)

> **Narration**
>
> "One in eleven diabetic patients is back in hospital within thirty days of going home.
> Each of those readmissions costs about fifteen thousand dollars, and many are preventable
> with a single follow-up phone call.
>
> The problem isn't knowing that calls work. It's that an under-resourced clinic can only
> make a handful of them a day, and no way to know which patients to pick.
>
> So I built a model that ranks patients at discharge — so the calls a clinic *can* make go
> to the people most likely to come back."

**On screen**
| Time | Show |
|---|---|
| 0:00–0:12 | Streamlit **Data & problem** tab — the problem statement and cost figures |
| 0:12–0:22 | Scroll to the dataset citation block (Strack et al., UCI #296) |
| 0:22–0:35 | Cut to the repo tree in VS Code, briefly showing `src/`, `notebooks/`, `app/` |

**Beneficiaries to make visible:** diabetic inpatients, discharge nurses, safety-net hospitals.

---

## 0:35 – 1:15 · Data, cleaning, EDA, feature engineering (103 words)

> **Narration**
>
> "The data is a hundred and one thousand hospital encounters from a hundred and thirty US
> hospitals, published with a peer-reviewed paper.
>
> Cleaning mattered more than modelling here. Weight is missing for ninety-seven percent of
> records, so I dropped it — but kept *whether* a patient was weighed, because that predicts
> care quality. I never imputed race, because imputing a protected attribute fabricates the
> evidence a fairness audit is meant to test.
>
> The big one: those hundred and one thousand rows are only seventy-one thousand patients.
> One person appears forty times. I collapsed to one row per patient — and the positive rate
> dropped from eleven percent to nine."

**On screen**
| Time | Show |
|---|---|
| 0:35–0:45 | Notebook Section 2 — missingness bar chart (`01_missingness.png`) |
| 0:45–0:55 | The missing-value policy table (four columns, four different policies) |
| 0:55–1:05 | `03_encounters_per_patient.png` — the 101,766 vs 71,518 bar pair |
| 1:05–1:15 | `06_prior_inpatient.png` — the 8.1% → 29.3% readmission gradient |

---

## 1:15 – 2:10 · Baseline vs bagging, boosting, stacking (137 words)

> **Narration**
>
> "Baselines first: logistic regression and a single decision tree. Then bagging with a random
> forest, boosting with XGBoost and LightGBM, and a stacking ensemble that combines all four.
>
> Everything is tuned with five-fold cross-validation, and every transform lives inside the
> pipeline so it refits on each fold. Stacking uses internal cross-validation too — that's what
> stops the meta-learner from just rewarding whichever base model memorised hardest.
>
> Stacking wins, at point six five seven versus point six four seven for the baseline. That's a
> real improvement — the bootstrap confidence interval excludes zero — but it's a small one, and
> I'm not going to oversell it.
>
> The number that actually matters is this: at ten percent follow-up capacity, the model finds
> two point four times more at-risk patients than calling at random. Same nurse hours, better
> targeting."

**On screen**
| Time | Show |
|---|---|
| 1:15–1:28 | Notebook Section 5 — the model list and tuning results table |
| 1:28–1:38 | Section 6 — the stacking `cv=5` explanation cell |
| 1:38–1:52 | `11_model_comparison.png` — the full eight-model bar chart |
| 1:52–2:00 | `17_bootstrap.png` — the CI excluding zero |
| 2:00–2:10 | Streamlit **Model comparison** tab — the threshold regimes table |

**Do not skip:** say the word "bootstrap confidence interval" out loud. It is the difference
between claiming an improvement and demonstrating one.

---

## 2:10 – 3:00 · Live demo, explanation, ethics, limits (124 words)

> **Narration**
>
> "Here's a synthetic patient — seventies, three prior admissions, HbA1c above eight, no
> payer code recorded.
>
> [*click Predict*]
>
> Eighty-three percent risk. Above the threshold, so: flag for a follow-up call. And the model
> says *why* — prior inpatient stays and emergency visits are driving it.
>
> Now the part I won't hide. The fairness audit failed. African American patients have the same
> readmission rate as Caucasian patients — nine point two versus nine point one percent — but the
> model flags them less often. Equal need, unequal service.
>
> So this isn't deployment-ready, and the audit is what tells us that. It's a triage aid for a
> nurse, never a reason to deny anyone care."

**On screen**
| Time | Show |
|---|---|
| 2:10–2:18 | Streamlit **Live prediction** — the synthetic patient's sliders |
| 2:18–2:26 | Risk gauge reads **83.1%** → "FLAG for follow-up call" |
| 2:26–2:36 | The live SHAP contribution bars appearing beside it |
| 2:36–2:50 | Switch to **Fairness** tab; select `race`, metric `selection_rate` |
| 2:50–3:00 | The label-bias warning box, then cut to `docs/ETHICS.md` |

**Closing frame:** repo URL on screen for two seconds.

---

## Delivery notes

- **Do not read this script verbatim on camera.** Learn the beats; speak it. Slight
  imperfection sounds more credible than a polished read.
- **Pause after "The fairness audit failed."** One full beat. It is the most memorable sentence
  in the pitch and it needs air.
- Say numbers as words — "point six five seven", not "zero point six five seven".
- If a take runs long, cut from the 0:35–1:15 block first. The cleaning detail is the most
  compressible part; the fairness finding is not.
- Record the demo click **live**. If the app misbehaves, re-record rather than splicing — a
  visibly real demo is worth more than a perfect one.

## Rubric coverage check

| Required element | Where it appears |
|---|---|
| Problem and beneficiaries | 0:00–0:35 |
| Data cleaning, EDA, feature engineering | 0:35–1:15 |
| Baseline vs bagging vs boosting vs stacking | 1:15–2:10 |
| Live prediction on a synthetic record | 2:10–2:26 |
| Local explanation | 2:26–2:36 |
| Ethics and limitations | 2:36–3:00 |
| Code/repository shown | 0:22–0:35, closing frame |
| Compact results visual | 1:38–1:52 |
| Live model output | 2:18–2:26 |
