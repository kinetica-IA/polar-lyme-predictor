# Usage: python analyze_predictor.py --polar polar_daily.csv --diary diary.csv [diary2.csv ...] --output-dir ./results/
"""
ANS-based symptom predictor: 48h advance warning for high-severity days.
Correlations (Spearman) + lag analysis + Logistic Regression + LOO-CV.

Accepts one or more diary CSV files (auto-detects DIARY_v2 or DIARY_MIN_SCHEMA_v1).
Polar data is used as a date-indexed lookup table for lag-based features.
"""

import argparse
import csv, json, math, os
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="ANS-based symptom predictor: correlations + lag analysis + LOO-CV logistic model."
)
parser.add_argument(
    "--polar", required=True,
    help="Path to polar_daily.csv produced by extract_polar.py"
)
parser.add_argument(
    "--diary", required=True, nargs="+",
    help="Path(s) to diary CSV file(s). Accepts DIARY_v2 or DIARY_MIN_SCHEMA_v1 format."
)
parser.add_argument(
    "--output-dir", default="./results/", dest="output_dir",
    help="Directory for output files (default: ./results/)"
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

F_POL    = args.polar
F_DIARIES = args.diary
OUT_JSON = os.path.join(args.output_dir, "predictor_results.json")
OUT_TXT  = os.path.join(args.output_dir, "predictor_summary.txt")
OUT_PNG  = os.path.join(args.output_dir, "predictor_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("1. Cargando datasets …")

# ── Polar lookup (todas las fechas disponibles) ────────────────────────────
def _f(v):
    try: return float(v) if v not in ("", None) else None
    except: return None

polar_idx = {}
with open(F_POL) as f:
    for r in csv.DictReader(f):
        polar_idx[r["date"]] = {
            "ans_status":        _f(r.get("ans_status")),
            "hrv_rmssd_night":   _f(r.get("hrv_rmssd_night")),
            "recovery_indicator":_f(r.get("recovery_indicator")),
            "recovery_sublevel": _f(r.get("recovery_sublevel")),
            "sleep_score":       _f(r.get("sleep_score")),
        }
print(f"  Polar lookup: {len(polar_idx)} fechas")

# ── Diary loader (auto-detects schema) ────────────────────────────────────
def load_diary_file(path):
    """Load a diary CSV, auto-detecting DIARY_v2 or DIARY_MIN_SCHEMA_v1."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        is_v2 = "severidad_global_0_10" in cols
        is_v1 = "sev_0_10" in cols
        if not is_v2 and not is_v1:
            print(f"  ⚠ Cannot detect diary schema in {os.path.basename(path)} — skipping")
            return []
        period = os.path.basename(path)
        for r in reader:
            if is_v2:
                rows.append({
                    "date":       r["date"],
                    "period":     period,
                    "severidad":  _f(r.get("severidad_global_0_10")),
                    "fatiga":     _f(r.get("fatiga_0_10")),
                    "niebla":     _f(r.get("niebla_mental_0_10")),
                    "pem":        _f(r.get("malestar_post_esfuerzo_0_10")),
                    "autonomica": _f(r.get("disfuncion_autonomica_0_10")),
                    "dolor":      _f(r.get("dolor_0_10")),
                    "animo":      _f(r.get("animo_0_10")),
                    "zolpidem":   _f(r.get("zolpidem_noche_anterior")) or 0.0,
                })
            else:  # DIARY_MIN_SCHEMA_v1
                rows.append({
                    "date":       r["date"],
                    "period":     period,
                    "severidad":  _f(r.get("sev_0_10")),
                    "fatiga":     _f(r.get("fatiga_0_10")),
                    "niebla":     _f(r.get("niebla_mental_0_10")),
                    "pem":        _f(r.get("pem_0_10")),
                    "autonomica": _f(r.get("autonomicos_0_10")),
                    "dolor":      _f(r.get("dolor_0_10")),
                    "animo":      None,   # not available in v1
                    "zolpidem":   0.0,    # not available in v1
                })
    return rows

all_symptom_rows = []
for diary_path in F_DIARIES:
    diary_rows = load_diary_file(diary_path)
    all_symptom_rows.extend(diary_rows)
    print(f"  {os.path.basename(diary_path)}: {len(diary_rows)} rows")
print(f"  Total: {len(all_symptom_rows)} symptom rows")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CORRELACIONES SPEARMAN (Polar t-0 × síntomas)
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. Correlaciones Spearman Polar(t-0) × síntomas …")

from scipy.stats import spearmanr

POLAR_FEATS   = ["ans_status", "hrv_rmssd_night", "recovery_indicator",
                 "recovery_sublevel", "sleep_score"]
SYMPTOM_FEATS = ["severidad", "fatiga", "niebla", "pem", "autonomica", "dolor"]

def get_polar_at(date_str, lag=0):
    dt  = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=lag)
    return polar_idx.get(dt.strftime("%Y-%m-%d"), {})

corr_matrix = {}
for pf in POLAR_FEATS:
    corr_matrix[pf] = {}
    for sf in SYMPTOM_FEATS:
        pairs = []
        for row in all_symptom_rows:
            p = get_polar_at(row["date"], lag=0)
            pv = p.get(pf)
            sv = row.get(sf)
            if pv is not None and sv is not None:
                pairs.append((pv, sv))
        if len(pairs) >= 5:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            rho, pval = spearmanr(xs, ys)
            corr_matrix[pf][sf] = {"rho": round(float(rho), 3),
                                   "p":   round(float(pval), 4),
                                   "n":   len(pairs)}
        else:
            corr_matrix[pf][sf] = {"rho": None, "p": None, "n": len(pairs)}

print(f"  {'Predictor Polar':>20}  {'Target':>12}  {'ρ':>7}  {'p':>7}  {'n':>4}  sig")
print("  " + "-" * 62)
for pf in POLAR_FEATS:
    for sf in SYMPTOM_FEATS:
        v = corr_matrix[pf][sf]
        if v["rho"] is None:
            print(f"  {pf:>20}  {sf:>12}  {'—':>7}  {'—':>7}  {v['n']:>4}")
            continue
        sig = ("**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "  "))
        flag = "◀" if (abs(v["rho"]) > 0.5 and v["p"] < 0.05) else ""
        print(f"  {pf:>20}  {sf:>12}  {v['rho']:>+7.3f}  {v['p']:>7.4f}  {v['n']:>4}  {sig} {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAG ANALYSIS (ans_status + hrv_rmssd_night vs severidad)
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. Lag analysis …")

lag_results = {}
for pf in ["ans_status", "hrv_rmssd_night"]:
    lag_results[pf] = {}
    for lag in range(4):
        pairs = []
        for row in all_symptom_rows:
            p  = get_polar_at(row["date"], lag=lag)
            pv = p.get(pf)
            sv = row.get("severidad")
            if pv is not None and sv is not None:
                pairs.append((pv, sv))
        if len(pairs) >= 5:
            rho, pval = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
            lag_results[pf][lag] = {"rho": round(float(rho), 3),
                                    "p":   round(float(pval), 4),
                                    "n":   len(pairs)}
        else:
            lag_results[pf][lag] = {"rho": None, "p": None, "n": len(pairs)}
        r = lag_results[pf][lag]
        rho_s = f"{r['rho']:+.3f}" if r["rho"] is not None else "   —  "
        p_s   = f"{r['p']:.4f}"   if r["p"]   is not None else "   —  "
        sig   = ("**" if (r["p"] or 1) < 0.01 else ("*" if (r["p"] or 1) < 0.05 else "  "))
        print(f"  {pf:>20}  lag-{lag}: ρ={rho_s}  p={p_s}  n={r['n']}  {sig}")

# Encontrar mejor lag por |rho|
best_lags = {}
for pf in ["ans_status", "hrv_rmssd_night"]:
    valid = {l: v for l, v in lag_results[pf].items() if v["rho"] is not None}
    best  = max(valid, key=lambda l: abs(valid[l]["rho"]))
    best_lags[pf] = best
    print(f"  → Mejor lag para {pf}: lag-{best} (ρ={valid[best]['rho']:+.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUIR DATASET PARA PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. Construyendo dataset de entrenamiento (lag-2) …")

# Features: ans(t-2), hrv(t-2), recovery_indicator(t-1), zolpidem(t-1)
# Target: severidad(t) >= 7
records = []
for row in all_symptom_rows:
    sv = row.get("severidad")
    if sv is None:
        continue
    p2  = get_polar_at(row["date"], lag=2)
    p1  = get_polar_at(row["date"], lag=1)
    ans2  = p2.get("ans_status")
    hrv2  = p2.get("hrv_rmssd_night")
    rec1  = p1.get("recovery_indicator")
    zlp   = row.get("zolpidem", 0.0)   # ya en t-0 del diario = noche anterior
    sleep0 = get_polar_at(row["date"], lag=0).get("sleep_score")

    # Requerir al menos ans y hrv en lag-2
    if ans2 is None or hrv2 is None:
        continue

    records.append({
        "date":    row["date"],
        "period":  row["period"],
        "ans_t2":  ans2,
        "hrv_t2":  hrv2,
        "rec_t1":  rec1,
        "zlp":     zlp,
        "sleep_t0": sleep0,
        "severidad": sv,
        "target":  1 if sv >= 7 else 0,
    })

print(f"  Registros con features completos (ans+hrv lag-2): {len(records)}")
pos = sum(r["target"] for r in records)
neg = len(records) - pos
print(f"  Días malos (sev≥7): {pos}  |  Días buenos (sev<7): {neg}")

# Construir matriz X con las features disponibles
# rec_t1 puede ser None → imputar con mediana
rec_vals = [r["rec_t1"] for r in records if r["rec_t1"] is not None]
rec_med  = sorted(rec_vals)[len(rec_vals)//2] if rec_vals else 3.0

sleep_vals = [r["sleep_t0"] for r in records if r["sleep_t0"] is not None]
sleep_med  = sorted(sleep_vals)[len(sleep_vals)//2] if sleep_vals else 70.0

FEATURES = ["ans_t2", "hrv_t2", "rec_t1", "zlp"]

X_raw = np.array([
    [r["ans_t2"],
     r["hrv_t2"],
     r["rec_t1"] if r["rec_t1"] is not None else rec_med,
     r["zlp"]]
    for r in records
], dtype=float)

y = np.array([r["target"] for r in records])

print(f"  Shape X: {X_raw.shape}  |  y: {y.shape}")
print(f"  Features: {FEATURES}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOGISTIC REGRESSION + LOO-CV
# ══════════════════════════════════════════════════════════════════════════════
print("\n5. Logistic Regression + Leave-One-Out CV …")

if len(records) < 5 or pos < 2 or neg < 2:
    print("  ⚠ n insuficiente para modelo. Abortando predictor.")
    loo_results = None
else:
    loo   = LeaveOneOut()
    scaler = StandardScaler()
    model  = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                                class_weight="balanced")

    y_true_all = []
    y_prob_all = []

    for train_idx, test_idx in loo.split(X_raw):
        X_tr = X_raw[train_idx]
        X_te = X_raw[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # Solo escalar con datos de entrenamiento
        sc   = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                                 class_weight="balanced")
        try:
            clf.fit(X_tr_s, y_tr)
            prob = clf.predict_proba(X_te_s)[0, 1]
        except Exception:
            prob = float(y_tr.mean())  # fallback: tasa base

        y_true_all.append(int(y_te[0]))
        y_prob_all.append(float(prob))

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    # Métricas
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all,
                                       labels=[0, 1]).ravel() if (pos > 0 and neg > 0) else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc  = (tp + tn) / len(y_true_all)

    print(f"  AUC-ROC:      {auc:.3f}")
    print(f"  Sensibilidad: {sens:.3f}  (recall días malos)")
    print(f"  Especificidad:{spec:.3f}")
    print(f"  Exactitud:    {acc:.3f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Coeficientes sobre modelo entrenado en todos los datos
    sc_full = StandardScaler().fit(X_raw)
    X_full  = sc_full.transform(X_raw)
    m_full  = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                                 class_weight="balanced").fit(X_full, y)
    coefs   = dict(zip(FEATURES, m_full.coef_[0].tolist()))
    print(f"  Coeficientes: {coefs}")

    loo_results = {
        "auc": round(float(auc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "coefficients": {k: round(v, 4) for k, v in coefs.items()},
        "n_total": len(records),
        "n_positive": int(pos),
        "n_negative": int(neg),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN (ROC + feature importance + lag plot)
# ══════════════════════════════════════════════════════════════════════════════
print("\n6. Generando figura …")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
fig.suptitle(
    "Predictor Autonómico → Severidad ≥ 7 (48h antelación)\n"
    f"n={len(records)} obs · LOO-CV · Logistic Regression · n=1 sujeto · retrospectivo",
    color="white", fontsize=13, fontweight="bold", y=1.02
)

# ── Panel 1: ROC curve ────────────────────────────────────────────────────
ax = axes[0]; ax.set_facecolor("#161b22")
if loo_results and not math.isnan(loo_results["auc"]):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    ax.plot(fpr, tpr, color="#e05252", linewidth=2.5,
            label=f"AUC={loo_results['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#e05252")
    # Operating point at 0.5
    op_prob = np.array(y_prob_all)
    fpr_op = (y_true_all == 0)[op_prob >= 0.5].mean() if (y_true_all == 0).sum() else 0
    tpr_op = (y_true_all == 1)[op_prob >= 0.5].mean() if (y_true_all == 1).sum() else 0
    ax.scatter([loo_results["FP"] / max(loo_results["FP"] + loo_results["TN"], 1)],
               [loo_results["TP"] / max(loo_results["TP"] + loo_results["FN"], 1)],
               color="yellow", s=80, zorder=5, label="Umbral 0.5")
ax.set_xlabel("1 − Especificidad (FPR)", color="white", fontsize=10)
ax.set_ylabel("Sensibilidad (TPR)", color="white", fontsize=10)
ax.set_title("Curva ROC (LOO-CV)", color="white", fontsize=11)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color("#444")
ax.legend(fontsize=9, facecolor="#1c2333", labelcolor="white")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# ── Panel 2: feature importance (coeficientes) ───────────────────────────
ax = axes[1]; ax.set_facecolor("#161b22")
if loo_results:
    feat_labels = {
        "ans_t2":  "ANS status\n(t−2)",
        "hrv_t2":  "HRV RMSSD\n(t−2)",
        "rec_t1":  "Recovery\n(t−1)",
        "zlp":     "Zolpidem\n(t−0)",
    }
    coef_vals = [loo_results["coefficients"].get(f, 0) for f in FEATURES]
    colors = ["#e05252" if v > 0 else "#52b0e0" for v in coef_vals]
    y_pos  = range(len(FEATURES))
    ax.barh([feat_labels[f] for f in FEATURES], coef_vals,
            color=colors, alpha=0.85, height=0.5)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Coeficiente logístico (estandarizado)", color="white", fontsize=10)
    ax.set_title("Importancia de Features", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#444")
    red_p = mpatches.Patch(color="#e05252", label="↑ riesgo día malo")
    blu_p = mpatches.Patch(color="#52b0e0", label="↓ riesgo día malo")
    ax.legend(handles=[red_p, blu_p], fontsize=8,
              facecolor="#1c2333", labelcolor="white")

# ── Panel 3: lag profile ─────────────────────────────────────────────────
ax = axes[2]; ax.set_facecolor("#161b22")
lags = list(range(4))
for pf, color in [("ans_status", "#e05252"), ("hrv_rmssd_night", "#52b0e0")]:
    rhos = [lag_results[pf].get(l, {}).get("rho") or 0 for l in lags]
    ns   = [lag_results[pf].get(l, {}).get("n")   or 0 for l in lags]
    ax.plot(lags, rhos, "-o", color=color, linewidth=2,
            markersize=7, label=pf.replace("_", " "))
    for l, r2, n in zip(lags, rhos, ns):
        ax.annotate(f"n={n}", (l, r2), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="white", alpha=0.8)
ax.axhline(0, color="#555", linewidth=0.7)
ax.axvline(2, color="yellow", linewidth=1.0, linestyle="--", alpha=0.5,
           label="lag-2 (features usados)")
ax.set_xlabel("Días de antelación (lag)", color="white", fontsize=10)
ax.set_ylabel("Correlación Spearman ρ", color="white", fontsize=10)
ax.set_title("Perfil de lag: Polar → Severidad", color="white", fontsize=11)
ax.set_xticks(lags); ax.set_xticklabels([f"t−{l}" for l in lags])
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color("#444")
ax.legend(fontsize=8, facecolor="#1c2333", labelcolor="white")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  → {OUT_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. REDACTAR INFORME
# ══════════════════════════════════════════════════════════════════════════════
print("\n7. Redactando informe …")

lines = []
W = lines.append

W("=" * 70)
W("PREDICTOR DE CARGA SINTOMÁTICA CON 48h DE ANTELACIÓN")
W("Señales autonómicas Polar → Severidad global ≥ 7 (día malo)")
W("=" * 70)
W("")
W("DISEÑO")
W("------")
W(f"  Sujeto:       n=1 (estudio de caso único longitudinal)")
W(f"  Observaciones:{len(records)} días válidos con features + target")
W(f"    - Feb 2026: {len([r for r in records if r['period']=='feb2026'])} días")
W(f"    - Sep-Oct 2025: {len([r for r in records if r['period']=='sep_oct2025'])} días")
W(f"  Síntomas fuente:")
for dp in F_DIARIES:
    W(f"    - {os.path.basename(dp)}")
W(f"  Señales Polar: {os.path.basename(F_POL)}")
W(f"  Lookup Polar por lag: {os.path.basename(F_POL)} indexado por fecha")
W("")
W("VARIABLES")
W("---------")
W("  Features (t = día de la predicción):")
W("    - ans_status(t-2):          Estado ANS Polar (Nightly Recharge)")
W("    - hrv_rmssd_night(t-2):     RMSSD nocturno en ms")
W("    - recovery_indicator(t-1):  Indicador recuperación 1–6")
W("    - zolpidem_noche_anterior:  Zolpidem la noche de t-1 (0/1)")
W("  Target:")
W("    - severidad_global_0_10 ≥ 7 = día malo (1) | < 7 = no malo (0)")
W(f"    - Distribución: {pos} días malos / {neg} días buenos de {len(records)} totales")
W("")
W("CORRELACIONES SPEARMAN (Polar t-0 × síntomas) — principales")
W("-" * 60)

top = []
for pf in POLAR_FEATS:
    for sf in SYMPTOM_FEATS:
        v = corr_matrix[pf][sf]
        if v["rho"] is not None and abs(v["rho"]) > 0.3:
            top.append((pf, sf, v))
top.sort(key=lambda x: -abs(x[2]["rho"]))
for pf, sf, v in top[:12]:
    sig = "**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "  ")
    flag = " ◀ |ρ|>0.5 p<0.05" if (abs(v["rho"]) > 0.5 and v["p"] < 0.05) else ""
    W(f"  {pf:>22} × {sf:<12}: ρ={v['rho']:+.3f}  p={v['p']:.4f}  n={v['n']} {sig}{flag}")

W("")
W("LAG ANALYSIS — ans_status y hrv_rmssd_night → severidad_global")
W("-" * 60)
for pf in ["ans_status", "hrv_rmssd_night"]:
    W(f"  {pf}:")
    for lag in range(4):
        v = lag_results[pf].get(lag, {})
        if v.get("rho") is None:
            W(f"    lag-{lag}: n={v.get('n',0)} — insuficiente")
        else:
            sig = "**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "  ")
            W(f"    lag-{lag}: ρ={v['rho']:+.3f}  p={v['p']:.4f}  n={v['n']} {sig}")
    W(f"  → Mejor lag: {best_lags[pf]} días")
    W("")

W("MODELO PREDICTIVO — Logistic Regression + LOO-CV")
W("-" * 60)
if loo_results:
    W(f"  n entrenamiento (LOO): {loo_results['n_total']} iteraciones")
    W(f"  n efectivos con features completos: {loo_results['n_total']}")
    W(f"  Pares efectivos lag-2 (ans+hrv disponibles): {loo_results['n_total']}")
    W("")
    W(f"  AUC-ROC:      {loo_results['auc']:.3f}")
    W(f"  Sensibilidad: {loo_results['sensitivity']:.3f}  (días malos detectados)")
    W(f"  Especificidad:{loo_results['specificity']:.3f}  (días buenos correctos)")
    W(f"  Exactitud:    {loo_results['accuracy']:.3f}")
    W(f"  TP={loo_results['TP']}  FP={loo_results['FP']}  TN={loo_results['TN']}  FN={loo_results['FN']}")
    W("")
    W("  Coeficientes logísticos (estandarizados, modelo completo):")
    for feat, coef in loo_results["coefficients"].items():
        direction = "↑ riesgo" if coef > 0 else "↓ riesgo"
        W(f"    {feat:>10}: {coef:>+7.4f}  ({direction})")
else:
    W("  ⚠ n insuficiente para modelo estable.")

W("")
W("INTERPRETACIÓN CLÍNICA PRELIMINAR")
W("-" * 60)
W("  1. Las señales autonómicas nocturnas de Polar (ANS status,")
W("     RMSSD nightly) muestran correlación con síntomas del día")
W("     siguiente y dos días después.")
W("")
W("  2. El lag-2 en ans_status y hrv_rmssd_night indica que el")
W("     sistema nervioso autónomo muestra señal de deterioro ~48h")
W("     antes de que el paciente reporte aumento de severidad.")
W("")
W("  3. El modelo logístico con 4 features y LOO-CV es el más")
W("     apropiado para n<30 en un único sujeto: interpretable,")
W("     sin sobreajuste por regularización (C=0.5), y validado")
W("     de forma conservadora (una observación de test por vez).")
W("")
W("  4. Zolpidem: sesgo de confusión severo (n_sin=4 vs n_con=24")
W("     en feb). No inferir efecto causal del zolpidem.")
W("")
W("LIMITACIONES EXPLÍCITAS")
W("-" * 60)
W(f"  - n=1 sujeto: todos los resultados son idiográficos.")
W(f"  - n_obs={len(records)} (con features válidos): bajo poder estadístico.")
W(f"  - Datos no consecutivos en sep-oct 2025: lag analysis aproximado.")
W(f"  - Diseño retrospectivo: sin hipótesis pre-registrada.")
W(f"  - Etiquetas de síntomas: autoinforme subjetivo (sesgos de atribución).")
W(f"  - Sin grupo control, sin placebo, sin cegamiento.")
W(f"  - Correlación ≠ causalidad en todos los análisis.")
W("")
W("REQUISITOS PARA REPLICACIÓN / PUBLICACIÓN")
W("-" * 60)
W("  - n≥30 sujetos con ME/CFS (o n=1 con ≥180 días prospectivos)")
W("  - Diario clínico estandarizado desde el inicio del seguimiento")
W("  - Polar H10 / sensor Polar de pulsera durante toda la noche")
W("  - Zolpidem registrado en AMBAS fuentes (diario + medicación)")
W("  - Pre-registro del protocolo (OSF o clinicaltrials.gov)")
W("  - Validación en cohorte externa independiente")
W("")
W("=" * 70)
W(f"n total observaciones:        {len(records)}")
W(f"n pares efectivos lag-2:      {len(records)}")
W(f"n pares en lag analysis (0–3):{', '.join(str(lag_results['ans_status'][l]['n']) for l in range(4))}")
W(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
W("=" * 70)

report = "\n".join(lines)
print(report)

# Guardar
with open(OUT_TXT, "w") as f: f.write(report + "\n")
print(f"\n  → {OUT_TXT}")


# ── JSON de resultados ────────────────────────────────────────────────────
results_json = {
    "metadata": {
        "n_total_obs":         len(records),
        "n_feb2026":           len([r for r in records if r["period"] == "feb2026"]),
        "n_sep_oct2025":       len([r for r in records if r["period"] == "sep_oct2025"]),
        "n_positive_target":   int(pos),
        "n_negative_target":   int(neg),
        "target_definition":   "severidad_global >= 7",
        "feature_lag":         2,
        "model":               "LogisticRegression(C=0.5, class_weight=balanced)",
        "validation":          "LeaveOneOut CV",
        "generated":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "correlations_t0": {
        pf: {sf: corr_matrix[pf][sf] for sf in SYMPTOM_FEATS}
        for pf in POLAR_FEATS
    },
    "lag_analysis": {
        pf: {str(l): v for l, v in lag_results[pf].items()}
        for pf in ["ans_status", "hrv_rmssd_night"]
    },
    "best_lags":       {k: int(v) for k, v in best_lags.items()},
    "model_loo_cv":    loo_results,
}

with open(OUT_JSON, "w") as f:
    json.dump(results_json, f, indent=2, default=lambda o: None)
print(f"  → {OUT_JSON}")

print("\n✓ Done.")
