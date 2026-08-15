"""End-to-end reproducible training run.

    python -m src.train              # full run
    python -m src.train --fast       # skip tuning and the slow ablations

Writes every table the notebook, dashboard and deck consume into
results/metrics/, and the deployable artefacts into models/.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src import config as C
from src import evaluate as E
from src import models as M
from src.cleaning import clean
from src.features import column_schema, engineer, rationale_table
from src.pipeline import leakage_checklist, make_splits, split_summary

warnings.filterwarnings("ignore")


def _save(df: pd.DataFrame, name: str) -> None:
    path = C.METRICS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(C.ROOT)}  ({len(df)} rows)")


def _save_json(obj, name: str) -> None:
    path = C.METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  wrote {path.relative_to(C.ROOT)}")


def main(fast: bool = False) -> None:
    t0 = time.time()
    print("=" * 72)
    print("ML for Social Good — 30-day readmission risk")
    print("=" * 72)

    # ---------------------------------------------------------------- data
    print("\n[1/9] Cleaning")
    cleaned, audit = clean()
    _save(audit.to_frame(), "cleaning_audit")
    print(f"  {len(cleaned):,} patients, positive rate {cleaned[C.TARGET].mean():.4f}")

    print("\n[2/9] Feature engineering")
    feat = engineer(cleaned)
    schema = column_schema(feat)
    _save(rationale_table(), "feature_rationale")
    print(f"  {feat.shape[1]} columns; {len(schema['numeric'])} numeric, "
          f"{len(schema['nominal'])} nominal, {len(schema['binary'])} binary")

    print("\n[3/9] Splitting")
    splits = make_splits(feat)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    _save(split_summary(splits), "split_summary")
    _save(leakage_checklist(feat, splits), "leakage_checklist")

    feat.loc[X_test.index, ["patient_nbr"] + C.SENSITIVE_COLS].to_csv(
        C.PROCESSED_DIR / "test_sensitive.csv", index=False
    )

    # ------------------------------------------------------- diagnostics
    if not fast:
        print("\n[4/9] Leakage sensitivity (patient vs row split)")
        all_enc = engineer(_all_encounters())
        _save(E.leakage_sensitivity(all_enc), "leakage_sensitivity")

        print("\n[5/9] Imputer ablation under simulated missingness")
        _save(
            E.imputer_ablation(X_train, y_train, X_val, y_val, schema),
            "imputer_ablation",
        )
    else:
        print("\n[4/9] Leakage sensitivity  — skipped (--fast)")
        print("[5/9] Imputer ablation     — skipped (--fast)")

    # -------------------------------------------------------------- tuning
    spw = M.scale_pos_weight(y_train)
    tuned_params: dict[str, dict] = {}

    if not fast:
        print("\n[6/9] Hyperparameter tuning")
        to_tune = {
            "random_forest": M.random_forest(schema),
            "xgboost": M.xgboost_model(schema, spw),
            "lightgbm": M.lightgbm_model(schema, spw),
        }
        rows = []
        for key, est in to_tune.items():
            t = time.time()
            search = RandomizedSearchCV(
                est, M.SEARCH_SPACES[key], n_iter=20, scoring="roc_auc",
                cv=M.CV, random_state=C.SEED, n_jobs=-1, refit=True, error_score="raise",
            )
            search.fit(X_train, y_train)
            tuned_params[key] = search.best_params_
            rows.append(
                {
                    "model": key,
                    "best_cv_roc_auc": round(float(search.best_score_), 5),
                    "cv_std": round(float(
                        search.cv_results_["std_test_score"][search.best_index_]
                    ), 5),
                    "seconds": round(time.time() - t, 1),
                    "best_params": json.dumps(search.best_params_),
                }
            )
            print(f"  {key:<14} cv_auc={search.best_score_:.4f} "
                  f"({time.time() - t:.0f}s)")
        _save(pd.DataFrame(rows), "tuning_results")
        _save_json(tuned_params, "tuned_params")
    else:
        print("\n[6/9] Tuning — skipped (--fast)")

    # -------------------------------------------------------------- models
    print("\n[7/9] Fitting all models")
    zoo = M.build_all(schema, y_train)

    for key, params in tuned_params.items():
        name = {
            "random_forest": "Random Forest (bagging)",
            "xgboost": "XGBoost (boosting)",
            "lightgbm": "LightGBM (boosting)",
        }[key]
        zoo[name].set_params(**params)

    fitted, val_probs, test_probs = {}, {}, {}
    fit_rows = []
    for name, model in zoo.items():
        t = time.time()
        model.fit(X_train, y_train)
        fitted[name] = model
        val_probs[name] = model.predict_proba(X_val)[:, 1]
        test_probs[name] = model.predict_proba(X_test)[:, 1]
        fit_rows.append({"model": name, "fit_seconds": round(time.time() - t, 1)})
        print(f"  {name:<32} val_auc={E.roc_auc_score(y_val, val_probs[name]):.4f}  "
              f"({time.time() - t:.0f}s)")
    _save(pd.DataFrame(fit_rows), "fit_times")

    # ------------------------------------------------- threshold on VAL only
    print("\n[8/9] Threshold selection and test evaluation")
    val_auc = {n: E.roc_auc_score(y_val, p) for n, p in val_probs.items()}
    best_name = max(val_auc, key=val_auc.get)
    print(f"  best on validation: {best_name} ({val_auc[best_name]:.4f})")

    thresholds, curves = {}, {}
    for name in zoo:
        thr, curve = E.cost_optimal_threshold(y_val, val_probs[name])
        thresholds[name] = thr
        curves[name] = curve
    curves[best_name].to_csv(C.METRICS_DIR / "cost_threshold_curve.csv", index=False)

    # Record why the capacity constraint is needed: without it the cost model
    # degenerates to flagging everyone, which no under-resourced clinic can do.
    thr_free, _ = E.cost_optimal_threshold(y_val, val_probs[best_name], capacity=None)
    curve_b = curves[best_name]
    row_free = curve_b.iloc[(curve_b["threshold"] - thr_free).abs().idxmin()]
    row_cap = curve_b.iloc[(curve_b["threshold"] - thresholds[best_name]).abs().idxmin()]
    _save(
        pd.DataFrame(
            [
                {"regime": "Unconstrained cost minimisation", "threshold": thr_free,
                 **row_free[["flagged_rate", "recall", "precision", "expected_cost"]].to_dict()},
                {"regime": f"Capacity-constrained (<={C.FOLLOWUP_CAPACITY:.0%})",
                 "threshold": thresholds[best_name],
                 **row_cap[["flagged_rate", "recall", "precision", "expected_cost"]].to_dict()},
                {"regime": "Default 0.5 threshold", "threshold": 0.5,
                 **curve_b.iloc[(curve_b["threshold"] - 0.5).abs().idxmin()][
                     ["flagged_rate", "recall", "precision", "expected_cost"]].to_dict()},
            ]
        ),
        "threshold_regimes",
    )
    print(f"  cost-optimal threshold: {thresholds[best_name]:.3f} "
          f"(capacity-constrained); unconstrained would be {thr_free:.3f} "
          f"flagging {row_free['flagged_rate']:.0%} of patients")

    comparison = pd.DataFrame(
        [
            E.classification_metrics(y_test, test_probs[n], thresholds[n], n)
            for n in zoo
        ]
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    _save(comparison, "model_comparison")

    at_half = pd.DataFrame(
        [E.classification_metrics(y_test, test_probs[n], 0.5, n) for n in zoo]
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    _save(at_half, "model_comparison_threshold_0.5")

    # ---------------------------------------------------------- significance
    base = M.BASELINE_KEY
    boot = E.bootstrap_auc_difference(y_test, test_probs[best_name], test_probs[base])
    boot["model_a"], boot["model_b"] = best_name, base

    mc = E.mcnemar_test(
        y_test,
        (test_probs[best_name] >= thresholds[best_name]).astype(int),
        (test_probs[base] >= thresholds[base]).astype(int),
    )
    _save_json({"bootstrap_auc": boot, "mcnemar": mc}, "significance")
    print(f"  AUC gain over baseline: {boot['mean_difference']:+.4f} "
          f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
          f"{'SIGNIFICANT' if boot['significant'] else 'not significant'}")

    # base-learner correlation
    corr = pd.DataFrame({n: test_probs[n] for n in zoo}).corr(method="spearman")
    corr.to_csv(C.METRICS_DIR / "prediction_correlation.csv")

    # ------------------------------------------------------------- fairness
    print("\n[9/9] Fairness audit")
    from src import fairness as F

    sens = feat.loc[X_test.index, C.SENSITIVE_COLS].reset_index(drop=True)
    sens = sens.rename(columns={"age": "age_band"})
    per_group, summary = F.full_audit(
        y_test, test_probs[best_name], sens, thresholds[best_name]
    )
    _save(per_group, "fairness_per_group")
    _save(summary, "fairness_summary")
    _save(
        F.group_threshold_mitigation(y_test, test_probs[best_name], sens["race"]),
        "fairness_group_thresholds",
    )
    _save(
        F.calibration_by_group(y_test, test_probs[best_name], sens["race"]),
        "calibration_by_race",
    )
    if not summary.empty:
        print(summary.round(4).to_string(index=False))

    # ---------------------------------------------------------- persistence
    # compress=3 takes the stacking artefact from ~143 MB to ~50 MB with no
    # measurable load-time penalty, which matters for the live demo.
    joblib.dump(fitted[best_name], C.MODELS_DIR / "best_pipeline.joblib", compress=3)
    joblib.dump(fitted[base], C.MODELS_DIR / "baseline_pipeline.joblib", compress=3)
    np.save(C.MODELS_DIR / "test_probs_best.npy", test_probs[best_name])

    meta = {
        "best_model": best_name,
        "threshold": thresholds[best_name],
        "threshold_rule": "minimises expected cost on the validation split",
        "cost_fn": C.COST_FN,
        "cost_fp": C.COST_FP,
        "abstention_band": C.ABSTENTION_BAND,
        "test_metrics": comparison[comparison["model"] == best_name].iloc[0].to_dict(),
        "baseline_metrics": comparison[comparison["model"] == base].iloc[0].to_dict(),
        "auc_gain_vs_baseline": boot,
        "seed": C.SEED,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1),
        "fast_mode": fast,
    }
    _save_json(meta, "run_metadata")
    (C.MODELS_DIR / "threshold.json").write_text(
        json.dumps({"threshold": thresholds[best_name], "model": best_name}, indent=2)
    )

    print("\n" + "=" * 72)
    print(comparison[
        ["model", "roc_auc", "pr_auc", "f1", "recall", "precision", "lift_at_10pct"]
    ].round(4).to_string(index=False))
    print("=" * 72)
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


def _all_encounters() -> pd.DataFrame:
    """Cleaned data WITHOUT patient de-duplication, for the leakage experiment."""
    from src.cleaning import (
        CleaningAudit, apply_missing_policy, binarise_target, decode_id_columns,
        drop_expired_and_hospice, drop_invalid_gender, drop_zero_variance,
    )
    from src.data_loader import load_raw_typed

    a = CleaningAudit()
    df = load_raw_typed()
    df = binarise_target(df)
    df = drop_expired_and_hospice(df, a)
    df = drop_invalid_gender(df, a)
    df = decode_id_columns(df)
    df = apply_missing_policy(df, a)
    return drop_zero_variance(df, a).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip tuning and ablations")
    main(**vars(ap.parse_args()))
