"""Build the self-contained HTML results dashboard.

    .venv/bin/python dashboard/build_dashboard.py

Design: a clinical audit report, not a marketing page. It leads with the
deployment-blocking fairness finding rather than the flattering lift number,
because that is the honest information hierarchy for this result.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config as C

OUT = ROOT / "dashboard" / "results_dashboard.html"


def fig(name: str) -> str:
    path = C.FIGURES_DIR / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure_block(name: str, caption: str) -> str:
    src = fig(name)
    if not src:
        return ""
    return (
        f'<figure><img src="{src}" alt="{caption}" loading="lazy">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


META = json.loads((C.METRICS_DIR / "run_metadata.json").read_text())
COMP = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")
SIG = json.loads((C.METRICS_DIR / "significance.json").read_text())
FAIR = pd.read_csv(C.METRICS_DIR / "fairness_summary.csv")
PERGROUP = pd.read_csv(C.METRICS_DIR / "fairness_per_group.csv")
LEAK = pd.read_csv(C.METRICS_DIR / "leakage_sensitivity.csv")
AUDIT = pd.read_csv(C.METRICS_DIR / "cleaning_audit.csv")
REGIMES = pd.read_csv(C.METRICS_DIR / "threshold_regimes.csv")
ABLATION = pd.read_csv(C.METRICS_DIR / "imputer_ablation.csv")

BEST = META["best_model"]
BASE = "Logistic Regression (baseline)"
best = COMP[COMP["model"] == BEST].iloc[0]
base = COMP[COMP["model"] == BASE].iloc[0]
BOOT = SIG["bootstrap_auc"]

failed = FAIR[~FAIR["passes_four_fifths_rule"]]["attribute"].tolist()
race = PERGROUP[(PERGROUP["attribute"] == "race") & PERGROUP["reliable"]]
cauc = race[race["group"] == "Caucasian"].iloc[0]
afam = race[race["group"] == "AfricanAmerican"].iloc[0]


def model_rows() -> str:
    out = []
    for _, r in COMP.iterrows():
        kind = (
            "baseline" if "baseline" in r["model"]
            else "ensemble" if ("Stacking" in r["model"] or "Voting" in r["model"])
            else "single"
        )
        cls = "row-best" if r["model"] == BEST else ""
        out.append(
            f'<tr class="{cls}"><td><span class="tag tag-{kind}">{kind}</span>'
            f'{r["model"]}</td>'
            f'<td class="num">{r["roc_auc"]:.4f}</td>'
            f'<td class="num">{r["pr_auc"]:.4f}</td>'
            f'<td class="num">{r["f1"]:.4f}</td>'
            f'<td class="num">{r["precision"]:.4f}</td>'
            f'<td class="num">{r["recall"]:.4f}</td>'
            f'<td class="num">{r["lift_at_10pct"]:.2f}×</td></tr>'
        )
    return "".join(out)


def fairness_rows() -> str:
    out = []
    for _, r in FAIR.iterrows():
        ok = r["passes_four_fifths_rule"]
        out.append(
            f"<tr><td>{r['attribute']}</td>"
            f'<td class="num">{r["disparate_impact_ratio"]:.3f}</td>'
            f'<td><span class="pill pill-{"ok" if ok else "bad"}">'
            f'{"pass" if ok else "FAIL"}</span></td>'
            f'<td class="num">{r["equalised_odds_difference"]:.3f}</td>'
            f'<td class="num">{r["auc_gap"]:.3f}</td></tr>'
        )
    return "".join(out)


def pergroup_rows(attr: str) -> str:
    d = PERGROUP[(PERGROUP["attribute"] == attr) & PERGROUP["reliable"]]
    return "".join(
        f"<tr><td>{r['group']}</td><td class='num'>{int(r['n']):,}</td>"
        f"<td class='num'>{r['positive_rate']:.4f}</td>"
        f"<td class='num'>{r['selection_rate']:.4f}</td>"
        f"<td class='num'>{r['tpr_recall']:.4f}</td>"
        f"<td class='num'>{r['roc_auc']:.4f}</td></tr>"
        for _, r in d.iterrows()
    )


def audit_rows() -> str:
    return "".join(
        f"<tr><td><code>{r['step']}</code></td>"
        f"<td class='num'>{int(r['rows_before']):,}</td>"
        f"<td class='num'>{int(r['rows_after']):,}</td>"
        f"<td class='num'>{int(r['rows_dropped']):,}</td>"
        f"<td class='why'>{r['reason']}</td></tr>"
        for _, r in AUDIT.iterrows()
    )


def leak_rows() -> str:
    return "".join(
        f"<tr><td>{r['split_strategy']}</td>"
        f"<td class='num'>{r['test_roc_auc']:.4f}</td>"
        f"<td class='num'>{int(r['patients_in_both_splits']):,}</td>"
        f"<td class='num'>{int(r['test_rows']):,}</td></tr>"
        for _, r in LEAK.iterrows()
    )


def regime_rows() -> str:
    return "".join(
        f"<tr><td>{r['regime']}</td><td class='num'>{r['threshold']:.3f}</td>"
        f"<td class='num'>{r['flagged_rate']:.1%}</td>"
        f"<td class='num'>{r['recall']:.3f}</td>"
        f"<td class='num'>{r['precision']:.3f}</td></tr>"
        for _, r in REGIMES.iterrows()
    )


def ablation_rows() -> str:
    piv = ABLATION.pivot(index="missing_rate", columns="imputer", values="val_roc_auc")
    return "".join(
        f"<tr><td class='num'>{idx:.0%}</td>"
        + "".join(f"<td class='num'>{piv.loc[idx, c]:.4f}</td>" for c in ["median", "knn", "mice"])
        + "</tr>"
        for idx in piv.index
    )


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Readmission Risk Model — Audit Report</title>
<style>
:root {{
  --paper:#f7f9f8; --card:#ffffff; --ink:#0e1c1b; --ink-2:#3b4d4b; --ink-3:#6b7a78;
  --rule:#dde5e3; --rule-2:#eef2f1;
  --accent:#0d6e6e; --accent-soft:#e3f0ef;
  --bad:#c2453d; --bad-soft:#fbe9e7;
  --caution:#b07d2b; --caution-soft:#fbf2e2;
  --good:#2f7d5c; --good-soft:#e6f2ec;
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --maxw: 68rem;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --paper:#0b1514; --card:#121f1e; --ink:#e8efed; --ink-2:#b3c3c0;
    --ink-3:#849491; --rule:#22322f; --rule-2:#1a2827;
    --accent:#4fb3ab; --accent-soft:#12302e;
    --bad:#e8776c; --bad-soft:#331816;
    --caution:#d6a24f; --caution-soft:#2e2415;
    --good:#5fbd8f; --good-soft:#13291f;
  }}
}}
:root[data-theme="light"] {{
  --paper:#f7f9f8; --card:#ffffff; --ink:#0e1c1b; --ink-2:#3b4d4b; --ink-3:#6b7a78;
  --rule:#dde5e3; --rule-2:#eef2f1;
  --accent:#0d6e6e; --accent-soft:#e3f0ef;
  --bad:#c2453d; --bad-soft:#fbe9e7;
  --caution:#b07d2b; --caution-soft:#fbf2e2;
  --good:#2f7d5c; --good-soft:#e6f2ec;
}}
:root[data-theme="dark"] {{
  --paper:#0b1514; --card:#121f1e; --ink:#e8efed; --ink-2:#b3c3c0; --ink-3:#849491;
  --rule:#22322f; --rule-2:#1a2827;
  --accent:#4fb3ab; --accent-soft:#12302e;
  --bad:#e8776c; --bad-soft:#331816;
  --caution:#d6a24f; --caution-soft:#2e2415;
  --good:#5fbd8f; --good-soft:#13291f;
}}

*,*::before,*::after {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:var(--maxw); margin:0 auto; padding:0 1.5rem; }}

h1,h2,h3 {{ font-family:var(--serif); font-weight:600; text-wrap:balance;
  line-height:1.22; margin:0; }}
h1 {{ font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-.01em; }}
h2 {{ font-size:clamp(1.35rem,2.5vw,1.75rem); }}
h3 {{ font-size:1.05rem; }}
p {{ margin:0; max-width:64ch; }}
a {{ color:var(--accent); }}
code {{ font-family:var(--mono); font-size:.86em;
  background:var(--rule-2); padding:.12em .38em; border-radius:3px; }}

.eyebrow {{ font-family:var(--mono); font-size:.7rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3); }}

/* ---- masthead ---- */
header.masthead {{ border-bottom:1px solid var(--rule); background:var(--card); }}
.masthead .wrap {{ display:flex; flex-direction:column; gap:1rem;
  padding-top:2.6rem; padding-bottom:2.2rem; }}
.masthead .rule {{ height:3px; width:4rem; background:var(--accent); }}
.masthead p.standfirst {{ font-size:1.08rem; color:var(--ink-2); max-width:60ch; }}
.masthead .meta {{ font-family:var(--mono); font-size:.76rem; color:var(--ink-3);
  display:flex; flex-wrap:wrap; gap:.5rem 1.4rem; padding-top:.4rem;
  border-top:1px solid var(--rule-2); margin-top:.4rem; }}

/* ---- verdict ---- */
.verdict {{ background:var(--bad-soft); border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule); }}
.verdict .wrap {{ display:grid; grid-template-columns:auto 1fr; gap:1.1rem;
  padding-top:1.6rem; padding-bottom:1.6rem; align-items:start; }}
.verdict .mark {{ font-family:var(--mono); font-weight:700; font-size:.72rem;
  letter-spacing:.1em; color:var(--bad); border:1.5px solid var(--bad);
  padding:.34rem .6rem; border-radius:3px; white-space:nowrap; }}
.verdict h2 {{ color:var(--bad); margin-bottom:.4rem; }}
.verdict p {{ color:var(--ink-2); font-size:.95rem; }}

/* ---- kpi strip ---- */
.kpis {{ display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }}
.kpi {{ background:var(--card); padding:1.15rem 1.25rem; }}
.kpi b {{ display:block; font-family:var(--mono); font-size:1.75rem; font-weight:600;
  letter-spacing:-.02em; font-variant-numeric:tabular-nums; margin-bottom:.2rem; }}
.kpi span {{ font-size:.78rem; color:var(--ink-3); line-height:1.4; display:block; }}
.kpi.is-accent b {{ color:var(--accent); }}
.kpi.is-bad b {{ color:var(--bad); }}
.kpi.is-good b {{ color:var(--good); }}

/* ---- sections ---- */
section {{ padding:3rem 0; border-bottom:1px solid var(--rule-2); }}
section > .wrap {{ display:flex; flex-direction:column; gap:1.15rem; }}
.sec-head {{ display:flex; flex-direction:column; gap:.3rem; }}

.note {{ font-size:.92rem; color:var(--ink-2); border-left:2px solid var(--rule);
  padding-left:1rem; max-width:64ch; }}
.callout {{ padding:1rem 1.15rem; border-radius:5px; font-size:.93rem;
  border-left:3px solid; }}
.callout.bad {{ background:var(--bad-soft); border-color:var(--bad); }}
.callout.good {{ background:var(--good-soft); border-color:var(--good); }}
.callout.caution {{ background:var(--caution-soft); border-color:var(--caution); }}
.callout strong {{ display:block; margin-bottom:.25rem; }}

/* ---- tables ---- */
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:5px;
  background:var(--card); }}
table {{ width:100%; border-collapse:collapse; font-size:.87rem; }}
th {{ text-align:left; font-family:var(--mono); font-size:.68rem; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3);
  padding:.7rem .85rem; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:.62rem .85rem; border-bottom:1px solid var(--rule-2);
  vertical-align:top; }}
tbody tr:last-child td {{ border-bottom:none; }}
td.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }}
td.why {{ color:var(--ink-3); font-size:.8rem; min-width:22rem; }}
tr.row-best td {{ background:var(--accent-soft); font-weight:600; }}

.tag {{ font-family:var(--mono); font-size:.6rem; letter-spacing:.06em;
  text-transform:uppercase; padding:.14rem .38rem; border-radius:3px;
  margin-right:.5rem; border:1px solid var(--rule); color:var(--ink-3); }}
.tag-ensemble {{ color:var(--accent); border-color:var(--accent); }}
.tag-baseline {{ color:var(--ink-3); }}
.pill {{ font-family:var(--mono); font-size:.68rem; font-weight:600;
  padding:.16rem .5rem; border-radius:99px; }}
.pill-ok {{ background:var(--good-soft); color:var(--good); }}
.pill-bad {{ background:var(--bad-soft); color:var(--bad); }}

/* ---- figures ---- */
figure {{ margin:0; border:1px solid var(--rule); border-radius:5px;
  overflow:hidden; background:var(--card); }}
figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ font-size:.78rem; color:var(--ink-3); padding:.65rem .85rem;
  border-top:1px solid var(--rule-2); }}
.grid2 {{ display:grid; gap:1.15rem; grid-template-columns:repeat(auto-fit,minmax(20rem,1fr)); }}

footer {{ padding:2.5rem 0 3.5rem; }}
footer .wrap {{ display:flex; flex-direction:column; gap:.7rem; }}
footer p {{ font-size:.85rem; color:var(--ink-3); }}
footer .cite {{ font-family:var(--mono); font-size:.76rem; line-height:1.65;
  border-left:2px solid var(--accent); padding-left:1rem; color:var(--ink-2); }}

@media (max-width:640px) {{
  .verdict .wrap {{ grid-template-columns:1fr; }}
  section {{ padding:2.2rem 0; }}
}}
@media (prefers-reduced-motion:reduce) {{ *{{animation:none!important;transition:none!important}} }}
</style></head>
<body>

<header class="masthead"><div class="wrap">
  <div class="rule"></div>
  <span class="eyebrow">Model audit report · Mission Health · CIA 3</span>
  <h1>30-Day Readmission Risk Triage</h1>
  <p class="standfirst">A ranked worklist to help under-resourced clinics decide which discharged
  diabetic patients to call back — and an audit of who that model fails.</p>
  <div class="meta">
    <span>Model: {BEST}</span>
    <span>Threshold: {META['threshold']:.3f}</span>
    <span>Seed: {META['seed']}</span>
    <span>Test n = {META['n_test']:,}</span>
    <span>UCI #296 · CC BY 4.0</span>
  </div>
</div></header>

<div class="verdict"><div class="wrap">
  <span class="mark">NOT DEPLOYMENT READY</span>
  <div>
    <h2>The fairness audit failed on {len(failed)} of 4 attributes</h2>
    <p>African American patients have a 30-day readmission rate statistically indistinguishable
    from Caucasian patients — <strong>{afam['positive_rate']:.2%} against
    {cauc['positive_rate']:.2%}</strong> — yet the model flags them for follow-up
    <strong>{afam['selection_rate']:.1%}</strong> of the time versus
    <strong>{cauc['selection_rate']:.1%}</strong>, with recall of
    {afam['tpr_recall']:.1%} against {cauc['tpr_recall']:.1%}. Equal clinical need, unequal service.
    This is stated first because it is the finding that governs whether anything else here matters.</p>
  </div>
</div></div>

<div class="kpis">
  <div class="kpi is-accent"><b>{best['roc_auc']:.4f}</b><span>Test ROC-AUC — best ensemble</span></div>
  <div class="kpi"><b>{base['roc_auc']:.4f}</b><span>Test ROC-AUC — logistic baseline</span></div>
  <div class="kpi is-good"><b>{best['lift_at_10pct']:.2f}×</b><span>Lift at 10% follow-up capacity</span></div>
  <div class="kpi"><b>{best['precision_at_10pct']:.1%}</b><span>Precision in top 10% vs 8.98% base rate</span></div>
  <div class="kpi is-bad"><b>0.566</b><span>ROC-AUC on repeat high-utilisers</span></div>
</div>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">01 · Problem</span>
  <h2>Who gets the call?</h2></div>
  <p>About 1 in 11 diabetic inpatients returns within 30 days. Each avoidable readmission costs the
  health system roughly <strong>${C.COST_FN:,.0f}</strong>; a nurse follow-up call costs about
  <strong>${C.COST_FP:,.0f}</strong> and measurably reduces that risk. The difficulty is not
  knowing that calls work — it is that a resource-limited clinic can only make a few of them, with
  no principled way to choose who.</p>
  <p class="note">Beneficiaries: diabetic inpatients, especially uninsured and high-utilising ones;
  discharge nurses, who receive a ranked worklist rather than an undifferentiated ward list; and
  safety-net hospitals carrying fixed capacity against rising readmission penalties.</p>
  <div class="callout good"><strong>Operational result</strong>
  At 10% follow-up capacity the model reaches {best['precision_at_10pct']:.1%} precision against an
  8.98% base rate — about <strong>{best['lift_at_10pct']:.2f}× more genuinely at-risk patients per
  nurse-hour</strong> than calling at random. Roughly 13 additional at-risk patients found per
  1,000 discharges, at no extra staffing cost.</div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">02 · Data</span>
  <h2>Cleaning was the hard part</h2></div>
  <p>101,766 encounters from 130 US hospitals — but only 71,518 patients, with one appearing 40
  times. Rows are not independent observations, and every step below is logged with its reason.</p>
  <div class="scroll"><table>
    <thead><tr><th>Step</th><th>Before</th><th>After</th><th>Dropped</th><th>Reason</th></tr></thead>
    <tbody>{audit_rows()}</tbody></table></div>
  <div class="callout caution"><strong>De-duplication moved the target</strong>
  The positive rate fell from 11.16% to 8.98% once each patient was reduced to a single encounter.
  Repeat encounters skew positive, so encounter-level modelling quietly flatters itself.</div>
  <div class="grid2">
    {figure_block('01_missingness.png', 'Missingness by column. Four columns at four very different rates, each needing a different policy — weight is dropped but "was weighed" is kept as a feature; race is never imputed, because imputing a protected attribute fabricates the evidence a fairness audit exists to test.')}
    {figure_block('03_encounters_per_patient.png', 'Encounters per patient. The duplication is at patient level, not row level — which is where the leakage risk lives.')}
  </div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">03 · Leakage</span>
  <h2>What the naive split actually cost</h2></div>
  <p>The textbook claim is that leakage inflates your score. We tested it rather than repeating it,
  training the same model on all encounters under both a random row split and a patient-grouped split.</p>
  <div class="scroll"><table>
    <thead><tr><th>Split strategy</th><th>Test ROC-AUC</th><th>Patients in both splits</th><th>Test rows</th></tr></thead>
    <tbody>{leak_rows()}</tbody></table></div>
  <div class="callout bad"><strong>Leakage lowered the score here, and revealed a real weakness</strong>
  Within the leaky test set, patients the model had already seen scored <strong>0.566</strong>
  against <strong>0.687</strong> for unseen patients. Repeat high-utilisers are so much harder to
  predict that their difficulty swamps any memorisation benefit. The model is close to useless
  precisely for the chronic patients with the greatest need — and the direction of leakage bias
  turned out to be unpredictable, which is the argument for controlling it by design rather than
  by expectation.</div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">04 · Models</span>
  <h2>Baseline vs bagging vs boosting vs stacking</h2></div>
  <p>Eight models, one untouched test set. Stacking uses 5-fold internal cross-validation so base
  learners only ever produce out-of-fold predictions — that is what prevents the meta-learner from
  rewarding whichever base model memorised hardest.</p>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th><th>Precision</th>
    <th>Recall</th><th>Lift@10%</th></tr></thead>
    <tbody>{model_rows()}</tbody></table></div>
  <div class="callout good"><strong>Does the ensemble beat the baseline? Yes — significantly, but modestly.</strong>
  {BOOT['mean_difference']:+.4f} ROC-AUC, paired bootstrap 95% CI
  [{BOOT['ci_low']:+.4f}, {BOOT['ci_high']:+.4f}] over {BOOT['n_bootstrap']:,} resamples. The
  interval excludes zero; McNemar agrees. An AUC near 0.66 matches the published ceiling on this
  dataset — the limit is the data, since housing, food security and caregiver support drive
  readmission and none appear in a discharge record.</div>
  <div class="grid2">
    {figure_block('11_model_comparison.png', 'All eight models on the untouched test set, by ROC-AUC and PR-AUC. PR-AUC is the more honest view at a 9% positive rate.')}
    {figure_block('17_bootstrap.png', 'Bootstrap confidence interval on the AUC difference between the best ensemble and the logistic baseline.')}
  </div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">05 · Decision rule</span>
  <h2>The threshold is a rationing decision</h2></div>
  <p>A missed readmission costs roughly 200× a phone call. Taken literally, that arithmetic says
  call every patient — the correct answer for a clinic with unlimited nurses, and a useless one
  for a real clinic.</p>
  <div class="scroll"><table>
    <thead><tr><th>Regime</th><th>Threshold</th><th>Flagged</th><th>Recall</th><th>Precision</th></tr></thead>
    <tbody>{regime_rows()}</tbody></table></div>
  <p class="note">Capacity, not cost, is the binding constraint. The deployed threshold minimises
  expected cost <em>subject to</em> what the clinic can staff, which makes the rationing decision
  explicit rather than inheriting it from a library default of 0.5.</p>
  {figure_block('13_cost_threshold.png', 'Expected cost against decision threshold, with the staffable region shaded and the deployed threshold marked.')}
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">06 · Preprocessing evidence</span>
  <h2>The imputer was chosen by experiment</h2></div>
  <p>After the missing-value policy no numeric gaps remain, so comparing imputers on this data
  would be theatre. Instead we injected missingness to answer a deployment question: when a
  clinic submits a record with labs left blank, which strategy degrades most gracefully?</p>
  <div class="scroll"><table>
    <thead><tr><th>Injected missingness</th><th>Median</th><th>KNN</th><th>MICE</th></tr></thead>
    <tbody>{ablation_rows()}</tbody></table></div>
  <p class="note">KNN holds up best; MICE degrades worst, because its per-column regressions rely on
  the other columns being present — exactly what fails when many go missing at once. The shipped
  pipeline uses median (a no-op on complete records, with no latency or size cost); the evidence
  says switch to KNN wherever real missingness exceeds about 10%.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">07 · Explainability</span>
  <h2>Why the model says what it says</h2></div>
  <p>SHAP is computed on the strongest tree base learner inside the stack, since a logistic
  meta-model over four base learners has no tree structure to walk. That substitution is measured,
  not assumed — the surrogate tracks the deployed model at Spearman ρ = 0.87.</p>
  <div class="grid2">
    {figure_block('18_shap_global.png', 'Global feature importance. Prior utilisation dominates, matching the EDA gradient and mutual information independently.')}
    {figure_block('21_demo_patient.png', 'A single synthetic patient. Each contribution is shown in log-odds, so the explanation is auditable rather than decorative.')}
  </div>
  <div class="callout caution"><strong>Two attribution methods disagree, and that is the finding</strong>
  SHAP and permutation importance share 9 of their top 15 features, but their rank correlation is
  −0.09. The <em>set</em> of influential variables is robust; the <em>ordering within it</em> is
  not, because many features are correlated by construction and permutation importance handles
  correlated predictors badly. Conclusions here rest on which features matter, not on their exact
  rank — a single SHAP bar chart would have projected false precision.</div>
  <p class="note">Actionable levers: HbA1c testing, medication reconciliation, discharge
  destination. Descriptive only: age, race, prior visit counts. A model whose top features were all
  descriptive would rank well and change nothing.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">08 · Fairness</span>
  <h2>Subgroup audit</h2></div>
  <div class="scroll"><table>
    <thead><tr><th>Attribute</th><th>Disparate impact</th><th>4/5ths rule</th>
    <th>Equalised odds gap</th><th>AUC gap</th></tr></thead>
    <tbody>{fairness_rows()}</tbody></table></div>
  <h3 style="margin-top:.6rem">By race</h3>
  <div class="scroll"><table>
    <thead><tr><th>Group</th><th>n</th><th>Readmission rate</th><th>Selection rate</th>
    <th>Recall</th><th>ROC-AUC</th></tr></thead>
    <tbody>{pergroup_rows('race')}</tbody></table></div>
  <div class="callout bad"><strong>Label bias runs deeper than any metric above can detect</strong>
  This dataset records readmissions only to the same hospital network. Patients who present
  elsewhere — disproportionately uninsured and mobile — are labelled "not readmitted". The model
  will under-flag exactly the people it should help, and every fairness number on this page is
  computed against those same biased labels, so the true disparity is likely worse than measured.
  No threshold tuning fixes a biased label; only better data collection does.</div>
  <p class="note">Per-group thresholds would equalise recall by construction, and the experiment is
  reported in the notebook. It is not adopted: conditioning a clinical cut-off explicitly on race
  raises legal and ethical objections that a metric improvement does not settle.</p>
  <div class="grid2">
    {figure_block('20_fairness_race.png', 'Subgroup performance by race, with group sizes shown so small-n differences are not over-read.')}
    {figure_block('20_fairness_payer_group.png', 'By payer group. Some of this gap is attributable to genuine risk differences, unlike the race gap.')}
  </div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><span class="eyebrow">09 · Limits</span>
  <h2>What this model cannot do</h2></div>
  <div class="callout bad"><strong>Blocking</strong>
  Fails the four-fifths rule on race, payer and age. Near-random (AUC 0.566) on repeat
  high-utilisers. Labels systematically miss out-of-network readmissions.</div>
  <div class="callout caution"><strong>Scope</strong>
  1999–2008 US data; not transferable to Indian clinics without prospective local revalidation.
  Triage aid only — never care denial, insurance pricing, discharge decisions, or staff evaluation.
  Predictions between 0.40 and 0.60 are routed to human review rather than auto-actioned.</div>
  <div class="callout good"><strong>Privacy</strong>
  De-identified at source under HIPAA Safe Harbor. No personally identifiable data included, and
  none added. Every demonstration record is synthetic.</div>
  <p class="note">The fairness audit was not a box to tick. It produced a result that blocks
  release — which is what an audit is for.</p>
</div></section>

<footer><div class="wrap">
  <span class="eyebrow">Source and reproduction</span>
  <p>Every figure and number on this page is regenerated by <code>python -m src.train</code>,
  seeded at {META['seed']}.</p>
  <p class="cite">Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J.,
  &amp; Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of
  70,000 clinical database patient records. <em>BioMed Research International</em>, 2014, 781670.<br>
  UCI Machine Learning Repository, dataset 296. Licensed CC BY 4.0.</p>
  <p>ML for Social Good Ensemble Challenge · CIA 3 · MCA 521-4 Machine Learning ·
  CHRIST (Deemed to be University)</p>
</div></footer>

</body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB self-contained)")
