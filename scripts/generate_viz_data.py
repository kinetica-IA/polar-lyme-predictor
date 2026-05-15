#!/usr/bin/env python3
"""
generate_viz_data.py — Generate ROC / forest / confusion-matrix data for the
ANS predictor visualizations on kineticaai.com.

Standalone adaptation of retrain_predictor.py:
  - Reads polar_live.json + diary_live.csv
  - Runs LOO-CV (logistic regression) for each of the 5 targets
  - Serialises ROC points, bootstrap AUC CI, confusion matrix per target
  - Writes ~/viz_data.json

Deterministic: seed=42. Outputs reproducible from published data.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

REPO       = Path.home() / "polar-lyme-predictor"
DIARY_CSV  = REPO / "data" / "diary_live.csv"
LIVE_JSON  = REPO / "public" / "data" / "polar_live.json"
OUT_JSON   = Path.home() / "viz_data.json"

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
    {"name": "severity",              "diary_key": "sev",    "threshold": 6},
    {"name": "pem",                   "diary_key": "pem",    "threshold": 5},
    {"name": "fatiga",                "diary_key": "fatiga", "threshold": 6},
    {"name": "niebla_mental",         "diary_key": "niebla", "threshold": 5},
    {"name": "disfuncion_autonomica", "diary_key": "auton",  "threshold": 5},
]

MAX_FEATURES = 5
MIN_AUC_IMPROVEMENT = 0.01
N_BOOTSTRAP = 1000
SEED = 42


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
                    "date":   r["date"],
                    "sev":    sev,
                    "pem":    _f(r.get("pem")),
                    "fatiga": _f(r.get("fatiga")),
                    "niebla": _f(r.get("niebla_mental")),
                    "auton":  _f(r.get("disfuncion_autonomica")),
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
            raw_rows.append({"date": row["date"], "target": 1 if target_val >= threshold else 0, **feat_vals})

    if not raw_rows:
        return None, None, [], []

    feature_names = [f[0] for f in all_features]
    medians = {}
    for fname in feature_names:
        vals = [r[fname] for r in raw_rows if r[fname] is not None]
        medians[fname] = sorted(vals)[len(vals)//2] if vals else 0.0

    X = np.zeros((len(raw_rows), len(feature_names)))
    y = np.array([r["target"] for r in raw_rows])
    for j, fname in enumerate(feature_names):
        for i, r in enumerate(raw_rows):
            X[i, j] = r[fname] if r[fname] is not None else medians[fname]
    return X, y, feature_names, [r["date"] for r in raw_rows]


def lr_factory():
    return LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=SEED)


def loo_cv(X, y, feature_indices):
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
    yt, yp = np.array(y_true), np.array(y_prob)
    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = 0.5
    return auc, yt, yp


def forward_select(X, y, feature_names):
    available = list(range(X.shape[1]))
    selected = []
    best_auc = 0.0
    for _ in range(MAX_FEATURES):
        best_candidate, best_candidate_auc = None, best_auc
        for idx in available:
            auc, _, _ = loo_cv(X, y, selected + [idx])
            if auc > best_candidate_auc:
                best_candidate_auc, best_candidate = auc, idx
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
        yt_b, yp_b = y_true[idx], y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt_b, yp_b)))
        except Exception:
            continue
    if len(aucs) < 100:
        return None, None, None
    aucs_sorted = sorted(aucs)
    return (aucs_sorted[int(0.025*len(aucs_sorted))],
            aucs_sorted[int(0.975*len(aucs_sorted))],
            aucs_sorted[len(aucs_sorted)//2])


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(thr[best]), float(tpr[best]), float(1 - fpr[best])


def roc_points(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return [{"fpr": round(float(f), 6), "tpr": round(float(t), 6),
             "thr": round(float(h), 6) if not (h != h or abs(h) == float('inf')) else None}
            for f, t, h in zip(fpr, tpr, thr)]


def main():
    print("generate_viz_data.py")
    diary = load_diary()
    polar = load_polar()
    print(f"diary={len(diary)} polar={len(polar)}")

    out = {
        "metadata": {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "validation": "LOO-CV (logistic regression, L2 C=0.5, class_weight=balanced)",
            "bootstrap": f"{N_BOOTSTRAP} iterations, seed={SEED}",
            "threshold": "Youden's J",
            "random_seed": SEED,
            "n_diary": len(diary),
            "n_polar_days": len(polar),
        },
        "targets": {},
    }

    for tcfg in TARGETS:
        name = tcfg["name"]
        X, y, fnames, _ = build_dataset(diary, polar, tcfg["diary_key"], tcfg["threshold"])
        if X is None or len(y) < 10:
            continue
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos < 3 or n_neg < 3:
            continue
        selected_idx, _ = forward_select(X, y, fnames)
        if not selected_idx:
            continue
        auc, yt, yp = loo_cv(X, y, selected_idx)
        ci_lo, ci_hi, ci_med = bootstrap_auc_ci(yt, yp)
        thr, sens, spec = youden_threshold(yt, yp)
        ypred = (yp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
        print(f"{name}: n={len(y)} AUC={auc:.4f} CI=[{ci_lo},{ci_hi}]")
        out["targets"][name] = {
            "selected_features": [fnames[i] for i in selected_idx],
            "n_training": int(len(y)), "n_positive": n_pos, "n_negative": n_neg,
            "auc_loo": round(float(auc), 4),
            "auc_ci95_lower": round(float(ci_lo), 4) if ci_lo is not None else None,
            "auc_ci95_upper": round(float(ci_hi), 4) if ci_hi is not None else None,
            "auc_ci95_median": round(float(ci_med), 4) if ci_med is not None else None,
            "y_true": [int(v) for v in yt.tolist()],
            "y_prob": [round(float(v), 6) for v in yp.tolist()],
            "roc": roc_points(yt, yp),
            "operating_point": {
                "threshold": round(float(thr), 6), "sensitivity": round(float(sens), 6),
                "specificity": round(float(spec), 6),
                "fpr": round(1.0 - float(spec), 6), "tpr": round(float(sens), 6),
            },
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
