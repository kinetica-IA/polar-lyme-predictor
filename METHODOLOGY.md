> **Archive — v1 (2026-03-01).** Current methodology in README.md.

# Methodology: ANS-Based Multi-Symptom Prediction in Post-Infectious Fatigue

## Model v3 — Multi-Target, Forward Selection, Bootstrap CIs

---

## 1. Study Design

**Design:** Prospective longitudinal N-of-1 study.

**Justification for N=1:** The primary aim was to establish individual-level predictive validity of autonomic wearable signals before scaling to cohort studies. N-of-1 designs are clinically appropriate for chronic conditions with high inter-individual variability, where population-level averages may obscure meaningful individual patterns.

**Observation period:** September 2025 - March 2026 (198 days continuous Polar monitoring); 62 symptom diary entries across this period.

**Ethics:** Personal observational data from the subject (also the researcher). No third-party data. No intervention performed.

---

## 2. Data Sources

### 2a. Wearable — Polar Grit X2

- **Export method:** GDPR full export via account.polar.com
- **Raw RR intervals:** Extracted from `ppi_samples` JSON files using `extract_rr.py`
- **Valid nights:** 193 (after artifact filtering: NN intervals 300-2000ms range)
- **Proprietary metrics:** ANS status, recovery indicator/sublevel, sleep score, RMSSD

### 2b. Advanced HRV — neurokit2

Raw RR intervals were processed through `compute_hrv.py` using neurokit2 to extract 8 time-domain and frequency-domain metrics per night:

| Metric | Domain | Description |
|--------|--------|-------------|
| SDNN | Time | Standard deviation of NN intervals |
| pNN50 | Time | Percentage of successive NN differences > 50ms |
| RMSSD | Time | Root mean square of successive differences |
| LF power | Frequency | Low-frequency power (0.04-0.15 Hz) |
| HF power | Frequency | High-frequency power (0.15-0.40 Hz) |
| LF/HF ratio | Frequency | Sympathovagal balance index |
| SD1 | Nonlinear | Short-term Poincare variability |
| SD2 | Nonlinear | Long-term Poincare variability |
| DFA-alpha1 | Nonlinear | Detrended fluctuation analysis (short-term) |

These features are not available from Polar's proprietary metrics. Their inclusion in v3 is what enabled the autonomic dysfunction finding.

### 2c. Subjective Clinical Diary

**DIARY_v2** (62 entries across study period):
- 7 symptom domains: severity, fatigue, brain fog, PEM, autonomic dysfunction, pain, mood
- 0-10 continuous scale per domain
- Prospective daily self-assessment via web interface (`diary.html`)
- Stored in `diary_live.csv`, auto-committed on push

---

## 3. Target Variable Construction

Each of the 5 target domains was binarized independently:

| Target | Threshold | Positive | Negative | Balance |
|--------|-----------|----------|----------|---------|
| Severity | >= 6 | 35 | 25 | 58/42% |
| PEM | >= 5 | 41 | 19 | 68/32% |
| Fatigue | >= 6 | 37 | 23 | 62/38% |
| Brain Fog | >= 5 | 54 | 6 | 90/10% ⚠ |
| Autonomic Dysfunction | >= 5 | 32 | 22 | 59/41% |

Brain fog class imbalance (54/6) means the AUC of 0.99 should be interpreted with caution. The model may be learning to predict the minority class by exclusion rather than signal.

---

## 4. Feature Space

### 4a. Candidate Features (13)

From Polar proprietary + neurokit2:

| Source | Features |
|--------|----------|
| Polar proprietary | hrv_rmssd_night, ans_status, recovery_sublevel, sleep_wake_min, sleep_interruptions |
| neurokit2 | hrv_sdnn, hrv_pnn50, hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio, hrv_sd1, hrv_sd2, hrv_dfa_alpha1 |

### 4b. Lag Windows

Each feature tested at 3 temporal lags:
- **t0:** Same night as symptom report
- **t-1:** One night prior
- **t-2:** Two nights prior

Total candidate pool per target: 13 features x 3 lags = 39 feature-lag combinations.

### 4c. Forward Selection

For each target independently:

1. Start with empty feature set
2. For each candidate: train LOO-CV model with current features + candidate
3. Select candidate that maximizes AUC
4. Stop when: AUC gain < 0.01, or 5 features reached
5. Return optimal feature set for this target

This approach allows each symptom to discover its own predictors from different biological pathways.

---

## 5. Validation

### 5a. Leave-One-Out Cross-Validation

- N iterations (one per effective pair)
- Each fold: StandardScaler fit on N-1 training points, applied to held-out test point
- No data leakage from scaling
- LogisticRegression with class_weight='balanced'

### 5b. Bootstrap Confidence Intervals

- 1000 bootstrap resamples of the LOO prediction vector
- 95% CI computed on AUC for each target
- Reports lower and upper bounds

### 5c. Model Comparison

Three algorithms evaluated per target:
- Logistic Regression (L2 penalty)
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)

Logistic Regression was best model for all 5 targets.

---

## 6. Results

### 6a. Per-Target Performance

| Target | AUC | CI 95% | Sens | Spec | Features |
|--------|-----|--------|------|------|----------|
| Severity | 0.84 | [0.73, 0.94] | 85.7% | 72.0% | hrv_rmssd_night_t0, ans_status_t2 |
| Autonomic Dysfunction | 0.86 | [0.75, 0.95] | 71.9% | 81.8% | hrv_rmssd_night_t0, hrv_lf_hf_ratio_t1, hrv_sd1_t0 |
| PEM | 0.79 | [0.67, 0.90] | 70.7% | 78.9% | hrv_rmssd_night_t0, recovery_sublevel_t0 |
| Fatigue | 0.79 | [0.66, 0.92] | 70.3% | 69.6% | hrv_rmssd_night_t0, hrv_rmssd_night_t1, sleep_wake_min_t2 |
| Brain Fog | 0.99 | [0.95, 1.00] | 88.9% | 100% | ans_status_t0, hrv_rmssd_night_t1, recovery_sublevel_t3 |

### 6b. Key Finding: Physiological Coherence of Autonomic Dysfunction

Autonomic dysfunction was the only target to select neurokit2-derived features:
- **LF/HF ratio (t-1):** Sympathovagal balance index — the ratio of sympathetic to parasympathetic modulation
- **SD1 (t0):** Short-term beat-to-beat variability from Poincare analysis

A model that predicts autonomic dysfunction using direct measures of autonomic balance demonstrates physiological coherence — the statistical model aligns with the biological mechanism.

### 6c. Residual Analysis

Severity model residuals (prediction error) were correlated with all symptom domains:

| Symptom | rho | p-value | Interpretation |
|---------|-----|---------|---------------|
| Brain Fog | +0.547 | < 0.001 | Strong — the model underestimates severity when brain fog is high |
| Autonomic Dysfunction | +0.372 | 0.006 | Moderate — model misses autonomic flares |

The residuals capture symptom dimensions that nocturnal HRV features cannot predict — specifically cognitive and autonomic symptoms that the patient often cannot articulate. The error is clinical signal, not noise.

### 6d. Feature Selection Patterns

Each target found different optimal predictors:

- **Severity:** Nocturnal RMSSD (same night) + ANS status (2 nights ago) — the classic autonomic prediction
- **Fatigue:** Two consecutive nights of RMSSD + sleep fragmentation — a cumulative burden pattern
- **PEM:** Same-night RMSSD + recovery sublevel — immediate autonomic state
- **Brain Fog:** ANS status + lagged RMSSD + 3-day recovery — complex temporal pattern (⚠ class imbalance)
- **Autonomic Dysfunction:** RMSSD + LF/HF ratio + SD1 — the only target using advanced HRV features

---

## 7. Methodological Decisions

### Missing Data: MNAR, Not Imputed

The worst symptom days have no diary entries because autonomic fatigue prevented recording. This is Missing Not At Random (MNAR) — the missingness mechanism is the phenomenon itself. We chose not to fabricate retroactive entries. The 60 prospective pairs are clean.

### Bootstrap Over SMOTE

Synthetic minority oversampling (SMOTE) would destroy the temporal autocorrelation structure in the data. The residual correlations (rho = +0.547 for brain fog) demonstrate meaningful clinical structure that synthetic data would dilute. Bootstrap resampling preserves the original data distribution while providing uncertainty quantification.

### Forward Selection Over Fixed Features

Model v2 used 4 fixed features for all targets (AUC 0.70). Model v3 allows each target to discover its own predictors from 39 candidates. This increased severity AUC to 0.84 with only 2 features — simpler and better.

---

## 8. Limitations

1. **N=1.** All results are idiographic. No generalization beyond this individual.
2. **N=60 pairs.** Preliminary. Bootstrap CIs are wide (e.g., severity [0.73, 0.94]).
3. **Brain fog class imbalance.** 54/6 split. AUC 0.99 likely inflated.
4. **Consumer wearable PPG.** Not clinical ECG. RR interval accuracy depends on PPG quality.
5. **MNAR assumption.** Missing days assumed worst — not independently verified.
6. **Fixed confounders uncontrolled.** Alprazolam 3mg, Bupropion 300mg, Pregabalin 500mg.
7. **Multiple comparisons.** 5 targets tested independently without family-wise correction.
8. **Correlation ≠ causation** in all analyses.

---

## 9. Reproducibility

All code is public. The pipeline runs automatically:
- `polar-biometrics.yml`: Nightly Polar API fetch (06:00 UTC)
- `polar-retrain.yml`: Retrain on every diary push
- Results auto-committed to `public/data/polar_live.json`
- Live display at [kineticaai.com](https://kineticaai.com)

---

## References

Shaffer F., Ginsberg J.P. (2017). An Overview of Heart Rate Variability Metrics and Norms. *Frontiers in Public Health*, 5, 258.

Stussman B. et al. (2020). Characterization of Post-exertional Malaise in Patients with ME/CFS. *Frontiers in Neurology*, 11, 1025.

Makowski D. et al. (2021). NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing. *Behavior Research Methods*, 53(4), 1689-1696.

---

## Author

**Alfonso Navarro** — Osteopath, Physicist, Clinical AI Builder
[kineticaai.com](https://kineticaai.com) · [LinkedIn](https://www.linkedin.com/in/navarro-kinetica-ai)
