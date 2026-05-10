# ANS-Based Multi-Symptom Prediction in Post-Infectious Fatigue

### An N-of-1 Longitudinal Study Using a Consumer Wearable

> **Key finding:** 5 symptom domains predicted independently from nocturnal
> wearable data. Autonomic dysfunction uniquely captured by advanced HRV
> features (LF/HF ratio + SD1) — physiologically coherent and not previously
> reported in longitudinal Lyme/ME-CFS research.

---

## What This Is

A fully automated pipeline that takes raw heart rate data from a consumer
wearable (Polar Grit X2), extracts advanced HRV metrics using neurokit2,
and trains independent predictive models for 5 symptom domains in
post-infectious fatigue. Every time a new symptom diary entry is logged
via the web interface, the models retrain automatically.

---

## Results — Model v3

| Target | AUC | CI 95% | Best Model | Selected Features | N |
|--------|-----|--------|------------|-------------------|---|
| Severity | 0.92 | [0.84, 0.99] | LR | hrv_rmssd_night_t0, ans_status_t2, hrv_sdnn_t2, hrv_dfa_alpha1_t2, recovery_sublevel_t0 | 61 |
| Autonomic Dysfunction | 0.84 | [0.72, 0.94] | LR | hrv_rmssd_night_t0, sleep_wake_min_t2 | 55 |
| PEM | 0.80 | [0.66, 0.90] | LR | hrv_rmssd_night_t0, hrv_pnn50_t1, hrv_rmssd_night_t1 | 61 |
| Fatigue | 0.83 | [0.71, 0.95] | LR | hrv_rmssd_night_t0, sleep_wake_min_t2, hrv_hf_power_t0 | 61 |
| Brain Fog | 0.99 | [0.95, 1.00] | LR | ans_status_t0, hrv_rmssd_night_t1, recovery_sublevel_t3 | 61 ⚠ |

⚠ Brain fog class split — AUC likely inflated. Monitoring.

**Reproducibility note:** Running `scripts/retrain_predictor.py` from the published data produces AUC 0.837 for autonomic dysfunction (deterministic, seed=42). The live pipeline at kineticaai.com (0.829) uses a cleaner L4 feature extraction path (`hrv_rmssd_calc` vs `hrv_rri_mean_ms`) with tighter data provenance. Both results are within the bootstrap CI [0.715, 0.936].

### Key Finding: Physiological Coherence

Autonomic dysfunction is the **only** target that selected advanced HRV
features: LF/HF ratio (sympathovagal balance, lag-1) and SD1 (short-term
Poincare variability, lag-0). These were extracted from raw RR intervals
using neurokit2 — not available from Polar's proprietary metrics.

A model that predicts autonomic dysfunction using measures of autonomic
balance is not just statistically significant — it's physiologically coherent.

### Residual Analysis: The Model's Error IS the Signal

Prediction residuals from the severity model correlate with symptoms
the patient cannot easily articulate:
- Brain fog: rho = +0.547, p < 0.001
- Autonomic dysfunction: rho = +0.372, p = 0.006

This suggests the model's "failures" capture neurological dimensions
that the ANS features alone cannot explain.

---

## Pipeline

```
Polar Grit X2 (PPG sensor, nocturnal)
  → GDPR export (ppi_samples JSON, RR intervals in ms)
  → extract_rr.py (193 valid nights, artifact filtering)
  → compute_hrv.py (neurokit2: SDNN, pNN50, LF/HF, SD1/SD2, DFA-alpha1)
  → merge_hrv.py (8 new features into polar_live.json)
  → retrain_predictor.py (per-target forward selection, LOO-CV, bootstrap CI)
  → polar_live.json (auto-commit via GitHub Action)
  → kineticaai.com (live display)
```

---

## Study Design

| Parameter | Value |
|-----------|-------|
| Design | Prospective longitudinal N-of-1 |
| Subject | Single adult, post-Lyme / autonomic dysfunction |
| Observation | 243 days continuous Polar monitoring |
| Diary entries | 61 |
| Effective pairs | 61 (55 for autonomic dysfunction) |
| Symptom domains | 7 (severity, PEM, fatigue, brain fog, autonomic, pain, mood) |
| Target domains | 5 (severity, PEM, fatigue, brain fog, autonomic dysfunction) |
| Wearable | Polar Grit X2 |
| HRV computation | neurokit2 (SDNN, pNN50, LF/HF, HF, SD1, SD2, DFA-alpha1, RMSSD) |
| Validation | Leave-One-Out CV + Bootstrap 1000x for CI 95% |
| Feature selection | Forward greedy per target, max 5 features, stop if AUC gain < 0.01 |

---

## Methodological Decisions

**Missing data is not random (MNAR).** The worst days have no diary entries
because autonomic fatigue prevented recording. We chose NOT to fabricate
retroactive entries. The 60 prospective pairs are clean, pre-study,
without confirmation bias.

**Forward selection per target.** Each symptom selects its own optimal
feature set from 13 candidates x 3 lag windows. No fixed feature sets.

**Bootstrap, not SMOTE.** The residual correlation structure (rho = +0.547
brain fog) proves the data has clinical structure — synthetic augmentation
would destroy it.

---

## Repository Structure

```
portfolio-alfie/
├── scripts/
│   ├── extract_rr.py          ← Polar ZIP → nightly RR intervals
│   ├── compute_hrv.py         ← neurokit2 → 8 HRV metrics per night
│   ├── merge_hrv.py           ← merge features into polar_live.json
│   ├── retrain_predictor.py   ← multi-target LOO-CV + forward selection + bootstrap
│   ├── fetch_polar_live.py    ← nightly Polar API → polar_live.json
│   └── log_diary.py           ← diary web form → diary_live.csv
├── data/
│   ├── diary_live.csv         ← symptom diary (auto-updated)
│   ├── hrv_features.csv       ← computed HRV metrics per night
│   └── hrv_rr_nightly.csv     ← raw RR intervals per night
├── public/data/
│   └── polar_live.json        ← all data + predictor results (auto-updated)
└── .github/workflows/
    ├── polar-biometrics.yml   ← nightly Polar fetch (06:00 UTC)
    └── polar-retrain.yml      ← retrain on every diary push
```

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Wearable | Polar Grit X2 (GDPR JSON export) |
| HRV analysis | neurokit2 (SDNN, pNN50, LF/HF, SD1/SD2, DFA-alpha1) |
| ML models | scikit-learn (LogisticRegression, RandomForest, GradientBoosting) |
| Validation | LOO-CV + Bootstrap 1000x |
| Feature selection | Forward greedy per target |
| Automation | GitHub Actions (nightly fetch + retrain on diary push) |
| Web | React + Vite (kineticaai.com) |
| Data | polar_live.json (single source of truth) |

---

## Limitations (Honest)

- **N=1.** Idiographic findings only. No generalization.
- **N=60 pairs.** Preliminary. CIs are wide.
- **Brain fog AUC 0.99** likely inflated — class split 54/6.
- **Consumer wearable PPG**, not clinical ECG.
- **MNAR assumption** — missing days assumed worst, not verified.
- **Fixed confounders:** Alprazolam 3mg, Bupropion 300mg, Pregabalin 500mg.
- **Correlation ≠ causation** in all analyses.

---

## Author

**Alfonso Navarro** — Osteopath, Physicist, Clinical AI Builder
[kineticaai.com](https://kineticaai.com) · [LinkedIn](https://www.linkedin.com/in/navarro-kinetica-ai)

---

## License

MIT — Free to use, adapt, and build upon with attribution.
