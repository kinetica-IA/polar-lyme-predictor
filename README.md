# ANS-Based Symptom Prediction in Post-Infectious Fatigue (Post-Lyme)
### An N-of-1 Longitudinal Study Using a Consumer Wearable

> **Key finding:** Autonomic nervous system status, measured nocturnally by a consumer wearable, predicts severe symptom burden **48 hours in advance** (ρ=+0.431, p=0.016, AUC=0.656, sensitivity=81%).
>
> **Counterintuitive:** Sleep metrics do **not** predict symptom flares (ρ≈0). The signal lives in the ANS, not in sleep architecture.

---

## Study Design

| Parameter | Value |
|---|---|
| Design | Retrospective longitudinal N-of-1 |
| Subject | Single adult, post-infectious fatigue (post-Lyme) / autonomic dysfunction |
| Observation period | Sep 2025 – Feb 2026 (6 months continuous Polar) |
| Days with symptoms | 43 (15 Sep-Oct 2025 + 28 Feb 2026) |
| Effective pairs (predictor) | 34 |
| Wearable | Polar Grit X2 |
| Clinical diary | DIARY_v2 (7 symptom domains, 0–10 scale) |

---

## Key Results

| Metric | Value |
|---|---|
| Best predictor | ANS status at lag-2 (t−48h) |
| Spearman ρ (ANS status → severity) | +0.431 (p=0.016) |
| AUC-ROC (LOO-CV) | 0.656 |
| Sensitivity (bad days detected) | 81.2% (13/16) |
| Specificity | 66.7% (12/18) |
| Model | Logistic Regression, C=0.5, class_weight=balanced |
| Validation | Leave-One-Out Cross-Validation (n=34 iterations) |

**Note:** lag calculated using exact date matching (`date - timedelta(days=2)`). Earlier positional `.shift(2)` method produced incorrect pairs due to dataset gaps — corrected in this version.

**Interpretation:** A suppressed ANS status two nights before a high-severity day (≥7/10) is the strongest detectable early warning signal. The recovery indicator and RMSSD show strong cross-sectional correlations with pain and autonomic dysfunction (|ρ|>0.5, p<0.01) but weaker predictive lag.

---

## Repository Structure

```
research/polar-lyme-predictor/
├── README.md                          ← this file
├── METHODOLOGY.md                     ← full methodological detail
├── abstract_v1.md                     ← draft abstract (in progress)
├── data/
│   └── diary_schema_v2_template.csv   ← blank template (no personal data)
├── scripts/
│   ├── extract_polar.py               ← Polar GDPR export (JSON) → polar_daily.csv
│   └── analyze_predictor.py           ← correlations + lag analysis + logistic model
└── results/
    ├── predictor_results.json         ← all metrics, machine-readable
    ├── predictor_summary.txt          ← full narrative report
    └── predictor_plot.png             ← ROC curve + feature importance + lag profile
```

**Not included:** personal data (raw Polar JSONs, filled diary CSVs). Only the template and anonymized aggregate results.

---

## How to Replicate

**Step 1 — Get your data**

Export your Polar Flow data: [account.polar.com](https://account.polar.com) → Download your data (GDPR export, ZIP with JSON files).
Fill in the daily symptom diary using `data/diary_schema_v2_template.csv`.

**Step 2 — Extract and align**

```bash
# Point DATASET to your Polar JSON folder, OUTPUT to target CSV
python scripts/extract_polar.py
```

**Step 3 — Run the predictor**

```bash
# Edit paths at the top of the file to match your diary + polar CSVs
python scripts/analyze_predictor.py
# Outputs: predictor_results.json, predictor_plot.png, predictor_summary.txt
```

**Requirements:** Python 3.10+, `numpy scipy scikit-learn matplotlib`

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Technical Stack

| Component | Tool |
|---|---|
| Wearable | Polar Grit X2 (GDPR JSON export, ~730 files / 6 months) |
| Extraction | Python 3.13, custom parser (`extract_polar.py`) |
| Statistical analysis | scipy.stats (Spearman), scikit-learn |
| Predictive model | LogisticRegression + LeaveOneOut CV |
| Visualization | matplotlib |
| AI pipeline | IO: LangGraph + Ollama qwen3:14b + ChromaDB (local) |

The entire analysis runs locally. No cloud APIs. No external data transfer.

---

## Limitations (Honest)

- **N=1.** Idiographic findings only. No generalization to other patients.
- **N=34 effective pairs.** Low statistical power. Effects should be considered preliminary.
- **Retrospective.** No pre-registered hypothesis.
- **Subjective self-report.** Symptom diary subject to anchoring and reference-shift.
- **Polar ANS status is proprietary.** Non-validated against clinical HRV standards (Kubios, Task Force).
- **Fixed confounders uncontrolled:** Alprazolam 3mg, Bupropión 300mg, Pregabalina 500mg.
- **Zolpidem imbalance:** n=4 without / n=24 with — causal inference not possible.
- **Correlation ≠ causation** in all analyses.

---

## For Replication / Publication

To achieve clinical validity:
- N≥30 subjects with ME/CFS or post-infectious fatigue, OR N=1 with ≥180 prospective consecutive days
- Pre-registered protocol (OSF or ClinicalTrials.gov)
- Standardized diary from day 0 of monitoring
- Kubios or open-standard HRV metrics (replacing proprietary ANS score)
- External validation cohort

---

## Paper / Abstract

→ `abstract_v1.md` (draft in progress)

---

## Author

**Alfonso Navarro** — Osteopath, Clinical Biomechanics, AI/Healthtech
[kineticaai.com](https://kineticaai.com)

---

## License

MIT — Free to use, adapt, and build upon with attribution.
