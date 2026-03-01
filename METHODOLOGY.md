# Methodology: ANS-Based Symptom Prediction in Post-Lyme Fatigue

---

## 1. Study Design

**Design:** Retrospective longitudinal N-of-1 study.

**Justification for N=1:** The primary aim was to establish individual-level predictive validity of autonomic wearable signals before scaling to cohort studies. N-of-1 designs are clinically appropriate for chronic conditions with high inter-individual variability, where population-level averages may obscure meaningful individual patterns. This approach follows idiographic methodology recommended in ME/CFS research (Jason & Goudsmit, 2020).

**Observation period:** September 2025 – March 2026 (continuous Polar monitoring); September 2025 and February 2026 (clinical diary).

**Ethics:** Personal observational data from the subject (also the researcher). No third-party data. No intervention performed.

---

## 2. Data Sources

### 2a. Wearable — Polar Grit X2

- **Export method:** GDPR full export via account.polar.com → ZIP archive of JSON files
- **Processing:** `extract_polar.py` parses ~730 JSON files into `polar_daily.csv` (one row per day, 52 columns)
- **Coverage:** 185 days, 2025-08-26 → 2026-03-01
- **Source files used:**

| File type | Content | N files |
|---|---|---|
| `nightly_recovery*.json` | ANS status, RMSSD, recovery indicator | 7 |
| `sleep_result*.json` | Sleep stages, efficiency, interruptions | 7 |
| `sleep_score*.json` | Composite sleep score | 5 |
| `ppi_samples*.json` | Raw PPI intervals (RMSSD/SDNN computed) | 25 |
| `training-session*.json` | Training load, HR zones | 77 |
| `activity*.json` | Steps, calories, MET | 90 |
| `orthostatic-test-result*.json` | Autonomic orthostatic response | 1 |

### 2b. Subjective Clinical Diary

**DIARY_MIN_SCHEMA_v1** (Sep-Oct 2025, 11 entries, non-consecutive):
- Domains: severity, fatigue, cognitive fog, PEM, autonomic symptoms, pain
- Retrospective entries; gaps between observations

**DIARY_v2** (Feb 2026, 28 entries, consecutive daily):
- Domains: all above + mood + zolpidem flag
- Prospective daily self-assessment
- Stored in: `diary_febrero_clean.csv`

**Template for replication:** `data/diary_schema_v2_template.csv`

---

## 3. Clinical Variables (7 Domains, DIARY_v2)

| Variable | Scale | Domain |
|---|---|---|
| `severidad_global_0_10` | 0–10 | Global symptom burden **(primary outcome)** |
| `fatiga_0_10` | 0–10 | Physical and cognitive fatigue |
| `niebla_mental_0_10` | 0–10 | Cognitive fog / brain fog |
| `malestar_post_esfuerzo_0_10` | 0–10 | Post-exertional malaise (PEM) |
| `disfuncion_autonomica_0_10` | 0–10 | Autonomic symptoms (palpitations, orthostatic intolerance) |
| `dolor_0_10` | 0–10 | Pain (musculoskeletal + neuropathic) |
| `animo_0_10` | 0–10 | Mood (higher = better; Feb 2026 only) |

**Binary target for classification:** `severidad_global ≥ 7` = "bad day" (1), `< 7` = "not bad" (0).
Class distribution in effective sample (n=34): 16 bad days / 18 good days.

---

## 4. Physiological Variables (Polar Nightly Recharge)

| Variable | Description | JSON source |
|---|---|---|
| `ans_status` | ANS status score (Polar proprietary; positive = recovered, negative = autonomic stress) | nightly_recovery |
| `hrv_rmssd_night` | Nocturnal RMSSD in ms | nightly_recovery |
| `recovery_indicator` | Nightly recharge level 1–6 | nightly_recovery |
| `recovery_sublevel` | Sub-level within recovery indicator | nightly_recovery |
| `sleep_score` | Composite sleep quality 0–100 (Polar algorithm) | sleep_score |

**Raw HRV (extended analysis only):**
`hrv_rmssd_daily` and `hrv_sdnn_daily` computed directly from PPI samples using standard formulas:
- RMSSD = √(mean of squared successive differences)
- SDNN = standard deviation of all NN intervals

---

## 5. Confounder: Zolpidem

`zolpidem_noche_anterior` (0/1): zolpidem taken the prior night.

**Distribution in Feb 2026:** 4 days without / 24 days with (severe imbalance).
**Handling:** Included as binary feature in the logistic model. No causal inference attempted (confounding by indication: zolpidem taken on nights with worse autonomic state).

**Fixed baseline medication (uncontrolled, constant throughout study):**
Alprazolam 3mg, Bupropión 300mg, Pregabalina 500mg.
These do not vary within the study period. Noted but not modeled.

---

## 6. Analysis Pipeline

### Step 1 — Data Extraction

`extract_polar.py`:
- Parses Polar GDPR JSON export (multiple file types)
- Aggregates training sessions by day (summed load, HR zones)
- Computes RMSSD and SDNN from raw PPI intervals (physiological filter: 300–2000ms)
- Outputs: `polar_daily_6m.csv` (52 columns, 185 rows)

### Step 2 — Dataset Unification

`analyze_predictor.py`:
- Loads `diary_febrero_clean.csv` (28 days) and `dataset_real_15days_clear.csv` (11 days)
- Harmonizes column names across diary schema versions
- Uses `polar_daily_6m.csv` as a date-indexed lookup table for Polar features at any target lag

### Step 3 — Correlation Analysis (Spearman, t=0)

Full matrix: 5 Polar predictors × 6 symptom domains.
N varies by predictor (27–34 pairwise complete observations due to gaps in nightly recovery data).
Significance threshold: p<0.05 (exploratory, no correction for multiple comparisons).

### Step 4 — Lag Analysis

For `ans_status` and `hrv_rmssd_night` vs `severidad_global`:
- Spearman ρ computed at lag 0, 1, 2, 3 days
- Polar feature at `date - lag` looked up in `polar_daily_6m.csv`
- N per lag: 33–34 pairs
- Best lag identified by maximum |ρ| with p<0.05

Note: Sep-Oct 2025 diary entries are non-consecutive. Lag analysis for that period is approximate (the Polar data is available but the symptom data has multi-day gaps between entries).

### Step 5 — Predictive Model

**Features** (t = target symptom date):
- `ans_status(t-2)`: ANS status 2 days prior
- `hrv_rmssd_night(t-2)`: nocturnal RMSSD 2 days prior
- `recovery_indicator(t-1)`: recovery level 1 day prior
- `zolpidem(t-0)`: zolpidem flag from diary (night before symptom assessment)

**Target:** `severidad_global ≥ 7` (binary, 1 = bad day)

**Model:** `LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)`
Regularization C=0.5 provides moderate L2 penalty appropriate for n=34.
class_weight='balanced' compensates for slight class imbalance (16/18).

**Validation:** Leave-One-Out Cross-Validation (n=34 iterations).
In each fold: StandardScaler fit exclusively on training set (n=33), applied to test point.
Prevents data leakage from scaling.

**Missing data:** `recovery_indicator(t-1)` absent for ~6 rows → imputed with within-sample median.

---

## 7. Results Summary

| Analysis | Metric | Value |
|---|---|---|
| Lag analysis | Best lag (ans_status → severity) | lag-2, ρ=+0.431, p=0.011 |
| Lag analysis | Best lag (hrv_rmssd_night) | lag-0, ρ=−0.300, p=0.084 (ns) |
| Cross-sectional (t=0) | Strongest: recovery_indicator × dolor | ρ=−0.588, p=0.001 |
| Cross-sectional (t=0) | hrv_rmssd_night × dolor | ρ=−0.587, p=0.001 |
| Cross-sectional (t=0) | ans_status × dolor | ρ=−0.543, p=0.003 |
| Cross-sectional (t=0) | sleep_score × severidad | ρ=−0.210, p=0.233 (ns) |
| Model AUC-ROC | LOO-CV | 0.656 |
| Model sensitivity | Bad days detected | 81.2% (13/16) |
| Model specificity | Good days correctly classified | 66.7% (12/18) |
| Feature importance | Dominant feature | ans_t2 (coef +0.598) |
| Feature importance | Second | rec_t1 (coef +0.539) |

**Key negative finding:** `sleep_score` shows no predictive relationship with severity (ρ≈0 at t=0, not tested in lag). This contradicts the common assumption in fatigue research that poor sleep drives next-day symptoms.

---

## 8. Limitations

1. **N=1.** No generalization beyond the individual. All results are idiographic.
2. **N=34 effective pairs.** Statistical power is limited to detecting medium-large effects (|ρ|>0.35).
3. **Retrospective.** No pre-registered hypothesis. Exploratory analysis with high risk of overfitting to one person's pattern.
4. **Non-consecutive Sep-Oct 2025 entries.** Lag continuity cannot be guaranteed for that period.
5. **Subjective diary.** Self-report susceptible to recall bias, anchoring, and temporal reference-shift.
6. **Polar ANS status is proprietary.** Algorithm not disclosed. Cannot be cross-validated against clinical HRV standards (Kubios, Task Force Consensus).
7. **Zolpidem confounding.** Severe group imbalance (4 vs 24). Causal inference impossible; included only as covariate.
8. **Multiple comparisons.** No correction applied (Bonferroni, FDR). All reported p-values are exploratory.
9. **Fixed medication baseline.** Alprazolam, Bupropión, Pregabalina may modulate ANS. Their effect is constant and uncontrolled.

---

## 9. Next Steps

| Priority | Action |
|---|---|
| Short-term | Prospective daily diary from March 2026 (continuous, consecutive) |
| Short-term | Kubios HRV analysis: DFA α1, SD1/SD2, pNN50 (validated open-standard metrics) |
| Medium-term | N=3–5 replication (post-Lyme / ME/CFS profiles with available wearable data) |
| Medium-term | Pre-registration on OSF before expanding sample |
| Long-term | Real-time monitoring pipeline: IO ingests daily Polar data, surfaces warnings |
| Long-term | Case series publication (JMIR mHealth, Frontiers in Digital Health, or similar) |

---

## References

Jason L.A., Goudsmit M. (2020). Unravelling long COVID. *Journal of Neurology*. [placeholder]

Shaffer F., Ginsberg J.P. (2017). An Overview of Heart Rate Variability Metrics and Norms. *Frontiers in Public Health*, 5, 258.

Stussman B. et al. (2020). Characterization of Post–exertional Malaise in Patients with Myalgic Encephalomyelitis/Chronic Fatigue Syndrome. *Frontiers in Neurology*, 11, 1025.
