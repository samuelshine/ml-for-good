"""Live demo: 30-day readmission risk triage.

    streamlit run app/streamlit_app.py

Loads the artefacts written by `python -m src.train`. It does not retrain
anything, so what you see here is exactly the model evaluated in the notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.synthetic import PROFILES, make_synthetic_patient, profile_label
from src import config as C

st.set_page_config(
    page_title="Readmission Risk Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = C.PALETTE["accent"]
PRIMARY = C.PALETTE["primary"]
POSITIVE = C.PALETTE["positive"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained pipeline…")
def load_model():
    pipe = joblib.load(C.MODELS_DIR / "best_pipeline.joblib")
    meta = json.loads((C.METRICS_DIR / "run_metadata.json").read_text())
    return pipe, meta


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = C.METRICS_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_json(name: str):
    path = C.METRICS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


try:
    model, META = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` first.")
    st.stop()

THRESHOLD = META["threshold"]
FEATURE_COLS = list(getattr(model, "feature_names_in_", []))


@st.cache_resource(show_spinner=False)
def load_explainer():
    """A TreeExplainer over an explainable stand-in for the deployed model.

    The deployed model is a stacking ensemble — a logistic meta-model over four
    base learners — so there is no single tree for TreeExplainer to walk. We
    explain the strongest tree base learner instead and say so on screen.
    """
    import shap

    from src import explain as X

    surrogate, label, _ = X.explainable_surrogate(model)
    return surrogate, shap.TreeExplainer(surrogate.named_steps["clf"]), label


PRETTY = {
    "number_inpatient": "Prior inpatient visits",
    "number_emergency": "Prior emergency visits",
    "number_outpatient": "Prior outpatient visits",
    "total_prior_visits": "Total prior visits",
    "prior_inpatient_flag": "Any prior inpatient stay",
    "high_utilizer": "High utiliser (2+ prior stays)",
    "service_intensity": "Service intensity this stay",
    "num_medications": "Medications this stay",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "number_diagnoses": "Number of diagnoses",
    "comorbidity_burden": "Comorbidity burden",
    "time_in_hospital": "Length of stay",
    "meds_per_day": "Medications per day",
    "labs_per_day": "Labs per day",
    "procedures_per_day": "Procedures per day",
    "num_med_changes": "Medication changes",
    "num_meds_prescribed": "Active medications",
    "a1c_tested": "HbA1c was measured",
    "a1c_severity_ord": "HbA1c severity",
    "discharged_home": "Discharged straight home",
    "age_midpoint": "Age",
    "payer_unknown": "No payer code recorded",
    "weight_recorded": "Weight was recorded",
    "emergency_admission": "Emergency admission",
    "diabetes_is_primary": "Diabetes is primary diagnosis",
    "has_circulatory": "Circulatory diagnosis present",
    "long_stay": "Stay longer than a week",
}


def humanise(name: str) -> str:
    """Turn encoder output names into something a clinician can read.

    One-hot columns arrive as `column_category`, and rare levels collapsed by
    `min_frequency` arrive as `column_infrequent_sklearn`.
    """
    if name in PRETTY:
        return PRETTY[name]
    if name.endswith("_infrequent_sklearn"):
        return f"{PRETTY.get(name[:-19], name[:-19].replace('_', ' ').capitalize())} (rare value)"
    for raw, label in PRETTY.items():
        if name.startswith(raw + "_"):
            return f"{label}: {name[len(raw) + 1:]}"
    if "_" in name:
        head, _, tail = name.rpartition("_")
        return f"{head.replace('_', ' ').capitalize()}: {tail}"
    return name.replace("_", " ").capitalize()


def explain_one(X_one: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from src import explain as X

    surrogate, explainer, label = load_explainer()
    enc = X.transform_frame(surrogate, X_one)

    sv = np.asarray(explainer.shap_values(enc))
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    contrib = (
        pd.DataFrame({"feature": enc.columns, "shap": sv[0],
                      "value": enc.iloc[0].to_numpy()})
        .assign(mag=lambda d: d["shap"].abs())
        .nlargest(10, "mag")
        .sort_values("shap")
        .assign(feature=lambda d: d["feature"].map(humanise))
    )
    return contrib, f"Computed on {label}."


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏥 30-Day Readmission Risk Triage")
st.caption(
    "ML for Social Good · CIA 3 · Mission Health — "
    "helping under-resourced clinics choose who to call after discharge"
)

comparison = load_csv("model_comparison")
best_row = (
    comparison[comparison["model"] == META["best_model"]].iloc[0]
    if comparison is not None else None
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", META["best_model"].split(" (")[0])
c2.metric("Test ROC-AUC", f"{best_row['roc_auc']:.4f}" if best_row is not None else "—")
c3.metric("Lift @ 10% capacity", f"{best_row['lift_at_10pct']:.2f}×" if best_row is not None else "—")
c4.metric("Deployed threshold", f"{THRESHOLD:.3f}")

tabs = st.tabs([
    "🎯 Live prediction",
    "📊 Model comparison",
    "🔍 Explainability",
    "⚖️ Fairness",
    "📁 Data & problem",
])


# ---------------------------------------------------------------------------
# Tab 1 — Live prediction
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Score a patient at discharge")
    st.caption(
        "All records here are **synthetic**. No real patient data is displayed. "
        "Blank numeric fields are imputed by the pipeline."
    )

    left, right = st.columns([1, 1.35])

    with left:
        profile = st.selectbox(
            "Start from a profile",
            list(PROFILES),
            format_func=lambda p: f"{p.replace('_', ' ').title()} — {profile_label(p)}",
        )
        base = make_synthetic_patient(profile)

        st.markdown("**Adjust the record**")
        age_band = st.select_slider("Age band", C.AGE_ORDER,
                                    value=str(base["age"].iloc[0]))
        los = st.slider("Length of stay (days)", 1, 14,
                        int(base["time_in_hospital"].iloc[0]))
        n_inpatient = st.slider("Prior inpatient visits (past year)", 0, 10,
                                int(base["number_inpatient"].iloc[0]))
        n_emergency = st.slider("Prior emergency visits (past year)", 0, 10,
                                int(base["number_emergency"].iloc[0]))
        n_outpatient = st.slider("Prior outpatient visits (past year)", 0, 15,
                                 int(base["number_outpatient"].iloc[0]))
        n_meds = st.slider("Medications this stay", 1, 60,
                           int(base["num_medications"].iloc[0]))
        n_diag = st.slider("Number of diagnoses", 1, 16,
                           int(base["number_diagnoses"].iloc[0]))
        a1c = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"],
                           index=["None", "Norm", ">7", ">8"].index(
                               str(base["A1Cresult"].iloc[0])
                               if "A1Cresult" in base else "None")
                           if "A1Cresult" in base else 0)
        discharge = st.selectbox(
            "Discharge destination",
            ["Discharged to home",
             "Discharged/transferred to SNF",
             "Discharged/transferred to home with home health service"],
            index=0,
        )

    patient = make_synthetic_patient(
        profile,
        age=age_band,
        time_in_hospital=los,
        number_inpatient=n_inpatient,
        number_emergency=n_emergency,
        number_outpatient=n_outpatient,
        num_medications=n_meds,
        number_diagnoses=n_diag,
        A1Cresult=a1c,
    )
    patient["discharge_disposition_id_desc"] = discharge
    patient["discharged_home"] = int(discharge == "Discharged to home")

    X_one = patient.reindex(columns=FEATURE_COLS) if FEATURE_COLS else patient
    prob = float(model.predict_proba(X_one)[:, 1][0])

    lo, hi = C.ABSTENTION_BAND
    if lo <= prob <= hi:
        decision, colour = "ESCALATE — human review", C.PALETTE["warning"]
    elif prob >= THRESHOLD:
        decision, colour = "FLAG for follow-up call", ACCENT
    else:
        decision, colour = "Routine discharge", POSITIVE

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Predicted 30-day readmission risk"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, THRESHOLD * 100], "color": "#e8f5e9"},
                    {"range": [THRESHOLD * 100, 60], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": THRESHOLD * 100,
                },
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:{colour}22;"
            f"border-left:5px solid {colour}'>"
            f"<b style='font-size:1.15rem'>{decision}</b><br>"
            f"<span style='opacity:.8'>Threshold {THRESHOLD:.3f} · "
            f"abstention band {lo:.2f}–{hi:.2f}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Why the model said this")
        try:
            contrib, note = explain_one(X_one)
            fig = px.bar(
                contrib, x="shap", y="feature", orientation="h",
                color=contrib["shap"] > 0,
                color_discrete_map={True: ACCENT, False: POSITIVE},
                labels={"shap": "Contribution to risk (log-odds)", "feature": ""},
                hover_data={"value": True},
            )
            fig.update_layout(showlegend=False, height=380,
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Red pushes risk up, green pushes it down. {note}")
        except Exception as exc:
            st.info(
                f"Per-prediction SHAP unavailable ({exc.__class__.__name__}: {exc}). "
                "Global importances are in the Explainability tab."
            )

    st.divider()
    st.markdown("**Clinical safeguard.** This is a ranked worklist, not a decision. "
                "A clinician retains the final call, and a low score never justifies "
                "refusing a patient who asks for help.")


# ---------------------------------------------------------------------------
# Tab 2 — Model comparison
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Baseline vs bagging vs boosting vs stacking")
    if comparison is None:
        st.warning("Run `make train` to generate results.")
    else:
        show = comparison[[
            "model", "roc_auc", "pr_auc", "f1", "precision", "recall",
            "specificity", "brier", "precision_at_10pct", "lift_at_10pct",
        ]].round(4)
        st.dataframe(
            show.style.background_gradient(subset=["roc_auc", "pr_auc"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )

        fig = px.bar(
            comparison.sort_values("roc_auc"), x="roc_auc", y="model",
            orientation="h", labels={"roc_auc": "Test ROC-AUC", "model": ""},
            color="roc_auc", color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, coloraxis_showscale=False,
                          xaxis_range=[comparison["roc_auc"].min() - 0.015,
                                       comparison["roc_auc"].max() + 0.01])
        st.plotly_chart(fig, use_container_width=True)

        sig = load_json("significance")
        if sig:
            b = sig["bootstrap_auc"]
            verdict = "significant" if b["significant"] else "not significant"
            st.markdown(
                f"**Best ensemble vs logistic baseline:** "
                f"{b['mean_difference']:+.4f} AUC, "
                f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] — **{verdict}**. "
                f"McNemar p = {sig['mcnemar']['p_value']:.3g}."
            )

        regimes = load_csv("threshold_regimes")
        if regimes is not None:
            st.markdown("#### Why the threshold is not 0.5")
            st.table(regimes.round(4).set_index("regime"))
            st.caption(
                "Unconstrained cost minimisation says call everyone — correct for a clinic "
                "with unlimited nurses, useless for a real one. Capacity is the binding constraint."
            )


# ---------------------------------------------------------------------------
# Tab 3 — Explainability
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("What drives the predictions")
    for label, fname in [
        ("Global importance (mean |SHAP|)", "18_shap_global.png"),
        ("SHAP beeswarm", "18b_shap_beeswarm.png"),
    ]:
        path = C.FIGURES_DIR / fname
        if path.exists():
            st.markdown(f"**{label}**")
            st.image(str(path), use_container_width=True)

    st.markdown(
        "**Actionable vs descriptive.** Prior visit counts and age rank patients but "
        "cannot be changed. HbA1c testing, medication reconciliation and discharge "
        "destination can — those are where a clinic can actually intervene."
    )


# ---------------------------------------------------------------------------
# Tab 4 — Fairness
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Subgroup performance audit")
    per_group = load_csv("fairness_per_group")
    summary = load_csv("fairness_summary")

    if summary is not None:
        n_failed = int((~summary["passes_four_fifths_rule"]).sum())
        if n_failed:
            st.error(
                f"**The model fails the four-fifths rule on {n_failed} of "
                f"{len(summary)} attributes.** African American patients have a readmission "
                "rate of 9.23% against 9.07% for Caucasian patients, yet are flagged 8.0% of "
                "the time versus 10.2%. Equal clinical need, unequal service — this is why "
                "the model is marked not deployment-ready."
            )

        tidy = pd.DataFrame({
            "Attribute": summary["attribute"],
            "Disparate impact": summary["disparate_impact_ratio"].round(3),
            "4/5ths rule": summary["passes_four_fifths_rule"].map(
                {True: "pass", False: "FAIL"}),
            "Equalised odds gap": summary["equalised_odds_difference"].round(3),
            "AUC gap": summary["auc_gap"].round(3),
        })
        st.table(tidy.set_index("Attribute"))

    if per_group is not None:
        attrs = sorted(per_group["attribute"].unique())
        attr = st.selectbox(
            "Attribute", attrs, index=attrs.index("race") if "race" in attrs else 0
        )
        d = per_group[(per_group["attribute"] == attr) & per_group["reliable"]]
        if not d.empty:
            labels = {
                "roc_auc": "Ranking quality (ROC-AUC)",
                "tpr_recall": "Recall — share of at-risk patients caught",
                "selection_rate": "Selection rate — share flagged for a call",
                "positive_rate": "Actual readmission rate",
            }
            metric = st.radio(
                "Metric", list(labels), horizontal=True,
                format_func=lambda m: labels[m],
            )
            fig = px.bar(
                d.sort_values("n", ascending=False), x="group", y=metric, text="n",
                labels={metric: labels[metric], "group": ""},
                color=metric, color_continuous_scale="Teal",
            )
            fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
            fig.update_layout(height=420, coloraxis_showscale=False,
                              margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Group sizes are shown so a gap measured on a few hundred patients is not "
                "read like one measured on ten thousand. Groups under n=100 are excluded."
            )

    st.warning(
        "**Label bias.** This dataset only records readmissions to the same hospital "
        "network. Patients who go elsewhere — disproportionately uninsured and mobile — "
        "are labelled 'not readmitted'. The model will under-flag exactly the people it "
        "should help, and no fairness metric above can detect it, because they are all "
        "computed against the same biased labels."
    )


# ---------------------------------------------------------------------------
# Tab 5 — Data and problem
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("The problem and the data")
    st.markdown(f"""
**Problem.** About 1 in 11 diabetic inpatients returns within 30 days. Each avoidable
readmission costs roughly \\${C.COST_FN:,.0f}. A follow-up call costs about \\${C.COST_FP:,.0f}
and measurably reduces that risk — but an under-resourced clinic can only call a fraction
of the patients it discharges.

**What the model does.** Ranks patients at discharge so the calls that *can* be made go to
the people most likely to return.

**Dataset.** {C.DATASET_NAME}
UCI ML Repository #296 · {C.DATASET_LICENCE} · 101,766 encounters from 71,518 patients
across 130 US hospitals, 1999–2008.

**Citation.** {C.CITATION}
    """)

    audit = load_csv("cleaning_audit")
    if audit is not None:
        st.markdown("#### Cleaning audit — every row we dropped and why")
        st.table(audit.set_index("step"))

    leak = load_csv("leakage_checklist")
    if leak is not None:
        st.markdown("#### Leakage checklist")
        st.table(leak.set_index("check"))

    st.info(
        f"**Limits.** US data from 1999–2008. Triage aid only — never care denial, "
        f"insurance pricing, or a discharge decision. Not deployable in Indian clinics "
        f"without prospective local validation. Full statement in `docs/ETHICS.md`."
    )
