#!/usr/bin/env python3
"""
generate_viz_data.py — Generate ROC / forest / confusion-matrix / model-detail
data for the ANS predictor visualizations on kineticaai.com.

For each of the 5 targets (severity, pem, fatiga, niebla_mental,
disfuncion_autonomica) we run LOO-CV with three models:

  1. LogisticRegression(C=0.5, class_weight=balanced) + StandardScaler
  2. RandomForestClassifier(n_estimators=200, class_weight=balanced)
  3. GradientBoostingClassifier(n_estimators=100, lr=0.05, max_depth=3)

Feature selection: forward selection on logistic regression (max 5 features,
ΔAUC≥0.01). The selected feature set is applied to all three models.

Per (target, model) we serialise:
  - LOO-CV y_true / y_prob
  - ROC curve points (fpr, tpr, thr) — Inf threshold → null
  - AUC (LOO-CV) + 95 % bootstrap CI (1000 iter, seed=42)
  - Confusion matrix at Youden-J optimal threshold
  - Operating point (threshold, sensitivity, specificity)
  - Hyperparameters dict
  - For LR: feature coefficients (signed). For RF/GBM: feature importances.

Deterministic: seed=42. Writes to ~/viz_data_v2.json.
"""

import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────────────────
REPO       = Path.home() / "polar-lyme-predictor"
DIARY_CSV  = REPO / "data" / "diary_live.csv"
LIVE_JSON  = REPO / "public" / "data" / "polar_live.json"
OUT_JSON   = Path.home() / "viz_data_v2.json"

# ── Config ─────────────────────────────────────────────────────────────────────────────────
CANDIDATE_FEATURES = [
    ("ans_status",          [0, 1, 2, 3]),
    ("hrv_rmssd_night",     [0, 1, 2, 3]),
    ("recovery_sublevel",   [0, 1, 2, 3]),
    ("sleep_wake_min",      [0, 1, 2]),
    ("sleep_interruptions", [0, 1, 2]),
    ("hrv_rri_mean_ms",     [2]),
    ("hrv_sdnn",            [0, 1, 2]),
    ("hrv_pnn50",           [0, 1, 2]),
    ("hrv_lf_hf_ratio",     [0, 1, 2]),
    ("hrv_hf_power",        [0, 1, 2]),
    ("hrv_sd1",             [0, 1, 2]),
    ("hrv_sd2",             [0, 1, 2]),
    ("hrv_dfa_alpha1",      [0, 1, 2]),
]

TARGETS = [
    {"name": "severity",              "diary_key": "sev",     "threshold": 6},
    {"name": "pem",                   "diary_key": "pem",     "threshold": 5},
    {"name": "fatiga",                "diary_key": "fatiga",  "threshold": 6},
    {"name": "niebla_mental",         "diary_key": "niebla",  "threshold": 5},
    {"name": "disfuncion_autonomica", "diary_key": "auton",   "threshold": 5},
]

MAX_FEATURES = 5
MIN_AUC_IMPROVEMENT = 0.01
N_BOOTSTRAP = 1000
SEED = 42

LR_HYPER  = {"C": 0.5, "max_iter": 1000, "class_weight": "balanced",
             "random_state": SEED, "scaler": "StandardScaler"}
RF_HYPER  = {"n_estimators": 200, "class_weight": "balanced",
             "random_state": SEED}
GBM_HYPER = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
             "random_state": SEED}


def _f(v):
    try:
        return float(v) if v and str(v).strip() not in ("", "None") else None
    except (ValueError, TypeError):
        return None


def load_diary():
    rows = []
    with open(DIARY_CSV, newline="") as f:
        for r in csv.DictReader(f):
            sev = _f(r.get("severidad_global"))
            if sev is not None:
                rows.append({
                    "date":    r["date"],
                    "sev":     sev,
                    "pem":     _f(r.get("pem")),
                    "fatiga":  _f(r.get("fatiga")),
                    "niebla":  _f(r.get("niebla_mental")),
                    "auton":   _f(r.get("disfuncion_autonomica")),
                })
    return sorted(rows, key=lambda r: r["date"])


def load_polar():
    return {r["date"]: r for r in json.loads(LIVE_JSON.read_text()).get("series", [])}


def polar_at(polar, date_str, lag):
    dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=lag)
    return polar.get(dt.strftime("%Y-%m-%d"), {})


def expand_feature_names():
    out = []
    for feat, lags in CANDIDATE_FEATURES:
        for lag in lags:
            out.append((f"{feat}_t{lag}", feat, lag))
    return out


def build_dataset(diary, polar, target_key, threshold):
    all_features = expand_feature_names()
    raw_rows = []
    for row in diary:
        target_val = row.get(target_key)
        if target_val is None:
            continue
        feat_vals = {}
        has_any = False
        for fname, polar_key, lag in all_features:
            p = polar_at(polar, row["date"], lag)
            v = _f(p.get(polar_key))
            feat_vals[fname] = v
            if v is not None:
                has_any = True
        if has_any:
            raw_rows.append({
                "date": row["date"],
                "target": 1 if target_val >= threshold else 0,
                **feat_vals,
            })

    if not raw_rows:
        return None, None, [], []

    feature_names = [f[0] for f in all_features]

    medians = {}
    for fname in feature_names:
        vals = [r[fname] for r in raw_rows if r[fname] is not None]
        medians[fname] = sorted(vals)[len(vals)//2] if vals else 0.0

    X = np.zeros((len(raw_rows), len(feature_names)))
    y = np.array([r["target"] for r in raw_rows])
    dates = [r["date"] for r in raw_rows]
    for j, fname in enumerate(feature_names):
        for i, r in enumerate(raw_rows):
            X[i, j] = r[fname] if r[fname] is not None else medians[fname]
    return X, y, feature_names, dates


# ── Model factories ─────────────────────────────────────────────────────────────────────────────────
def lr_factory():
    return LogisticRegression(C=LR_HYPER["C"], max_iter=LR_HYPER["max_iter"],
                              class_weight=LR_HYPER["class_weight"],
                              random_state=SEED)


def rf_factory():
    return RandomForestClassifier(n_estimators=RF_HYPER["n_estimators"],
                                  class_weight=RF_HYPER["class_weight"],
                                  random_state=SEED)


def gbm_factory():
    return GradientBoostingClassifier(n_estimators=GBM_HYPER["n_estimators"],
                                      learning_rate=GBM_HYPER["learning_rate"],
                                      max_depth=GBM_HYPER["max_depth"],
                                      random_state=SEED)


def loo_cv_lr(X, y, feature_indices):
    """LOO-CV with logistic regression + StandardScaler."""
    Xs = X[:, feature_indices]
    y_true, y_prob = [], []
    for tr, te in LeaveOneOut().split(Xs):
        sc = StandardScaler().fit(Xs[tr])
        clf = lr_factory()
        try:
            clf.fit(sc.transform(Xs[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(Xs[te]))[0, 1]
        except Exception:
            prob = float(y[tr].mean())
        y_true.append(int(y[te][0]))
        y_prob.append(float(prob))
    yt = np.array(y_true)
    yp = np.array(y_prob)
    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = 0.5
    return auc, yt, yp


def loo_cv_tree(X, y, feature_indices, factory):
    """LOO-CV with tree-based model (no scaling needed)."""
    Xs = X[:, feature_indices]
    y_true, y_prob = [], []
    for tr, te in LeaveOneOut().split(Xs):
        clf = factory()
        try:
            clf.fit(Xs[tr], y[tr])
            prob = clf.predict_proba(Xs[te])[0, 1]
        except Exception:
            prob = float(y[tr].mean())
        y_true.append(int(y[te][0]))
        y_prob.append(float(prob))
    yt = np.array(y_true)
    yp = np.array(y_prob)
    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = 0.5
    return auc, yt, yp


def forward_select(X, y, feature_names):
    """Forward selection on LR LOO-CV."""
    n = X.shape[1]
    available = list(range(n))
    selected = []
    best_auc = 0.0
    for _ in range(MAX_FEATURES):
        best_candidate = None
        best_candidate_auc = best_auc
        for idx in available:
            trial = selected + [idx]
            auc, _, _ = loo_cv_lr(X, y, trial)
            if auc > best_candidate_auc:
                best_candidate_auc = auc
                best_candidate = idx
        if best_candidate is None or (best_candidate_auc - best_auc) < MIN_AUC_IMPROVEMENT:
            break
        selected.append(best_candidate)
        available.remove(best_candidate)
        best_auc = best_candidate_auc
    return selected, best_auc


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt_b, yp_b)))
        except Exception:
            continue
    if len(aucs) < 100:
        return None, None, None
    aucs_sorted = sorted(aucs)
    lo = aucs_sorted[int(0.025 * len(aucs_sorted))]
    hi = aucs_sorted[int(0.975 * len(aucs_sorted))]
    median = aucs_sorted[len(aucs_sorted)//2]
    return lo, hi, median


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best]), float(tpr[best]), float(1 - fpr[best])


def roc_points(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    out = []
    for f, t, h in zip(fpr, tpr, thr):
        hv = None if not math.isfinite(float(h)) else round(float(h), 6)
        out.append({"fpr": round(float(f), 6),
                    "tpr": round(float(t), 6),
                    "thr": hv})
    return out


def fit_full_lr(X, y, feature_indices):
    """Fit a single LR on the entire dataset to extract coefficients."""
    Xs = X[:, feature_indices]
    sc = StandardScaler().fit(Xs)
    clf = lr_factory()
    clf.fit(sc.transform(Xs), y)
    coefs = clf.coef_[0].tolist()
    return coefs


def fit_full_tree(X, y, feature_indices, factory):
    """Fit a single tree-based model on the entire dataset for importances."""
    Xs = X[:, feature_indices]
    clf = factory()
    clf.fit(Xs, y)
    return clf.feature_importances_.tolist()


def model_block(yt, yp, hyper, extra):
    """Build the per-model output block."""
    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = 0.5
    ci_lo, ci_hi, ci_med = bootstrap_auc_ci(yt, yp)
    thr, sens, spec = youden_threshold(yt, yp)
    ypred = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
    block = {
        "hyperparams":     hyper,
        "auc_loo":         round(float(auc), 4),
        "auc_ci95_lower":  round(float(ci_lo), 4) if ci_lo is not None else None,
        "auc_ci95_upper":  round(float(ci_hi), 4) if ci_hi is not None else None,
        "auc_ci95_median": round(float(ci_med), 4) if ci_med is not None else None,
        "y_true":          [int(v) for v in yt.tolist()],
        "y_prob":          [round(float(v), 6) for v in yp.tolist()],
        "roc":             roc_points(yt, yp),
        "operating_point": {
            "threshold":   round(float(thr), 6),
            "sensitivity": round(float(sens), 6),
            "specificity": round(float(spec), 6),
            "fpr":         round(1.0 - float(spec), 6),
            "tpr":         round(float(sens), 6),
        },
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        },
    }
    block.update(extra)
    return block


def main():
    print("generate_viz_data.py — multi-model viz data")
    print("=" * 60)
    diary = load_diary()
    polar = load_polar()
    print(f"diary entries: {len(diary)}   polar days: {len(polar)}")

    out = {
        "metadata": {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "data_paths": {
                "diary": "polar-lyme-predictor/data/diary_live.csv",
                "polar": "polar-lyme-predictor/public/data/polar_live.json",
            },
            "validation":       "LOO-CV (3 models: LR + RF + GBM)",
            "feature_selection": "Forward selection on LR LOO-CV (max 5, ΔAUC≥0.01); same feature set applied to RF and GBM.",
            "bootstrap":        f"{N_BOOTSTRAP} iterations, seed={SEED}",
            "threshold":        "Youden's J (max TPR - FPR on ROC curve)",
            "random_seed":      SEED,
            "n_diary":          len(diary),
            "n_polar_days":     len(polar),
            "models": {
                "logistic_regression": LR_HYPER,
                "random_forest":       RF_HYPER,
                "gradient_boosting":   GBM_HYPER,
            },
        },
        "targets": {},
    }

    for tcfg in TARGETS:
        name = tcfg["name"]
        print(f"\n=== {name} (threshold ≥ {tcfg['threshold']}) ===")
        X, y, fnames, dates = build_dataset(diary, polar, tcfg["diary_key"], tcfg["threshold"])
        if X is None or len(y) < 10:
            print("  insufficient data, skipping"); continue
        n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
        if n_pos < 3 or n_neg < 3:
            print(f"  too few of one class (pos={n_pos}, neg={n_neg}), skipping"); continue

        selected_idx, _ = forward_select(X, y, fnames)
        if not selected_idx:
            print("  no features selected, skipping"); continue
        selected_names = [fnames[i] for i in selected_idx]
        print(f"  selected features: {selected_names}")

        # LR
        _, yt_lr, yp_lr = loo_cv_lr(X, y, selected_idx)
        lr_coefs = fit_full_lr(X, y, selected_idx)
        lr_extra = {"feature_weights": {selected_names[i]: round(float(c), 6)
                                        for i, c in enumerate(lr_coefs)}}
        lr_block = model_block(yt_lr, yp_lr, LR_HYPER, lr_extra)

        # RF
        _, yt_rf, yp_rf = loo_cv_tree(X, y, selected_idx, rf_factory)
        rf_imp = fit_full_tree(X, y, selected_idx, rf_factory)
        rf_extra = {"feature_importances": {selected_names[i]: round(float(c), 6)
                                            for i, c in enumerate(rf_imp)}}
        rf_block = model_block(yt_rf, yp_rf, RF_HYPER, rf_extra)

        # GBM
        _, yt_gb, yp_gb = loo_cv_tree(X, y, selected_idx, gbm_factory)
        gb_imp = fit_full_tree(X, y, selected_idx, gbm_factory)
        gb_extra = {"feature_importances": {selected_names[i]: round(float(c), 6)
                                            for i, c in enumerate(gb_imp)}}
        gb_block = model_block(yt_gb, yp_gb, GBM_HYPER, gb_extra)

        print(f"  AUC  LR={lr_block['auc_loo']}  RF={rf_block['auc_loo']}  GBM={gb_block['auc_loo']}")

        out["targets"][name] = {
            "label":             name,
            "threshold_value":   tcfg["threshold"],
            "selected_features": selected_names,
            "n_training":        int(len(y)),
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "models": {
                "logistic_regression": lr_block,
                "random_forest":       rf_block,
                "gradient_boosting":   gb_block,
            },
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}  ({OUT_JSON.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
