# Usage: python extract_polar.py --dataset /path/to/polar/export --output ./polar_daily.csv
"""
Polar Flow GDPR export → polar_daily.csv

Parses Polar JSON files into a unified daily CSV with one row per date.
Sources: sleep_result, sleep_score, nightly_recovery, ppi_samples,
         training-session, activity, orthostatic-test-result, jump-test-result
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Extract Polar Flow GDPR export (JSON folder) to a unified daily CSV."
)
parser.add_argument(
    "--dataset", required=True,
    help="Path to folder containing Polar GDPR export JSON files"
)
parser.add_argument(
    "--output", default="./polar_daily.csv",
    help="Output CSV path (default: ./polar_daily.csv)"
)
args = parser.parse_args()

DATASET = args.dataset
OUTPUT  = args.output


# ── helpers ────────────────────────────────────────────────────────────────

def iso_dur_to_min(s):
    """PT8H1M30S → 481.5  |  PT0S / None → 0.0"""
    if not s or s == "PT0S":
        return 0.0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", s)
    if not m:
        return 0.0
    h = float(m.group(1) or 0)
    mi = float(m.group(2) or 0)
    sec = float(m.group(3) or 0)
    return h * 60 + mi + sec / 60

def rmssd(intervals):
    if len(intervals) < 2:
        return None
    diffs = [intervals[i+1] - intervals[i] for i in range(len(intervals) - 1)]
    return math.sqrt(sum(d*d for d in diffs) / len(diffs))

def sdnn(intervals):
    if len(intervals) < 2:
        return None
    mean = sum(intervals) / len(intervals)
    return math.sqrt(sum((x - mean)**2 for x in intervals) / len(intervals))

def r(v, ndigits=1):
    """Round only real numbers; pass through empty string / None."""
    if v is None or v == "":
        return ""
    return round(v, ndigits)


# ── 1. SLEEP RESULT ─────────────────────────────────────────────────────────

print("Cargando sleep_result …")
sleep = {}
for f in glob.glob(os.path.join(DATASET, "sleep_result*.json")):
    for night in json.load(open(f)):
        date = night.get("night")
        if not date:
            continue
        ev   = night.get("evaluation") or {}
        ph   = ev.get("phaseDurations") or {}
        intr = ev.get("interruptions") or {}
        sleep[date] = {
            "sleep_duration_h":    r(iso_dur_to_min(ev.get("sleepSpan"))    / 60, 2),
            "sleep_asleep_h":      r(iso_dur_to_min(ev.get("asleepDuration"))/ 60, 2),
            "sleep_efficiency_pct":r(ev.get("efficiencyPercent")),
            "sleep_rem_pct":       r(ph.get("remPercentage")),
            "sleep_deep_pct":      r(ph.get("deepPercentage")),
            "sleep_wake_min":      r(iso_dur_to_min(ph.get("wake")), 0),
            "sleep_interruptions": intr.get("totalCount", ""),
            "sleep_long_interruptions": intr.get("longCount", ""),
        }

print(f"  → {len(sleep)} noches")


# ── 2. SLEEP SCORE ──────────────────────────────────────────────────────────

print("Cargando sleep_score …")
sscore = {}
for f in glob.glob(os.path.join(DATASET, "sleep_score*.json")):
    for night in json.load(open(f)):
        date = night.get("night")
        if not date:
            continue
        sc = night.get("sleepScoreResult") or {}
        sscore[date] = {
            "sleep_score":            r(sc.get("sleepScore")),
            "sleep_continuity_score": r(sc.get("continuityScore")),
            "sleep_efficiency_score": r(sc.get("efficiencyScore")),
            "sleep_rem_score":        r(sc.get("remScore")),
            "sleep_n3_score":         r(sc.get("n3Score")),
            "sleep_long_int_score":   r(sc.get("longInterruptionsScore")),
        }

print(f"  → {len(sscore)} noches")


# ── 3. NIGHTLY RECOVERY (HRV + indicador) ───────────────────────────────────

print("Cargando nightly_recovery …")
recovery = {}
for f in glob.glob(os.path.join(DATASET, "nightly_recovery*.json")):
    if "blob" in os.path.basename(f):
        continue
    for night in json.load(open(f)):
        date = night.get("night")
        if not date:
            continue
        recovery[date] = {
            "hrv_rmssd_night":  night.get("meanNightlyRecoveryRmssd", ""),
            "hrv_rri_mean_ms":  night.get("meanNightlyRecoveryRri", ""),
            "hrv_resp_ms":      night.get("meanNightlyRecoveryRespirationInterval", ""),
            "recovery_indicator":  night.get("recoveryIndicator", ""),
            "recovery_sublevel":   night.get("recoveryIndicatorSubLevel", ""),
            "ans_status":          r(night.get("ansStatus"), 4),
            "ans_rate":            night.get("ansRate", ""),
            "baseline_rmssd":      night.get("meanBaselineRmssd", ""),
            "vitality_tip":        night.get("vitalityTip", ""),
            "exercise_tip":        night.get("exerciseTip", ""),
            "sleep_tip":           night.get("sleepTip", ""),
        }

print(f"  → {len(recovery)} noches")


# ── 4. PPI SAMPLES → RMSSD / SDNN diarios ──────────────────────────────────

print("Cargando ppi_samples (puede tardar ~30 s) …")
ppi_by_date = defaultdict(list)
ppi_files   = sorted(glob.glob(os.path.join(DATASET, "ppi_samples*.json")))

for i, f in enumerate(ppi_files):
    print(f"  [{i+1}/{len(ppi_files)}] {os.path.basename(f)}")
    for day_entry in json.load(open(f)):
        date = day_entry.get("date")
        if not date:
            continue
        for device in day_entry.get("devicePpiSamplesList") or []:
            for sample in device.get("ppiSamples") or []:
                pl = sample.get("pulseLength")
                if pl and 300 < pl < 2000:          # rango fisiológico
                    ppi_by_date[date].append(pl)

ppi_hrv = {}
for date, intervals in ppi_by_date.items():
    if len(intervals) >= 10:
        ppi_hrv[date] = {
            "hrv_rmssd_daily": r(rmssd(intervals)),
            "hrv_sdnn_daily":  r(sdnn(intervals)),
            "ppi_count":       len(intervals),
        }

print(f"  → {len(ppi_hrv)} días con datos PPI")


# ── 5. TRAINING SESSIONS (agregado por día) ──────────────────────────────────

print("Cargando training-sessions …")
t_raw = defaultdict(list)

for f in glob.glob(os.path.join(DATASET, "training-session*.json")):
    data = json.load(open(f))
    start = data.get("startTime", "")
    if not start:
        continue
    date = start[:10]

    dur_min  = iso_dur_to_min(data.get("duration", "PT0S"))
    calories = data.get("kiloCalories") or 0
    avg_hr   = data.get("averageHeartRate") or ""
    max_hr   = data.get("maximumHeartRate") or ""

    sports = []
    zones  = {z: 0.0 for z in range(1, 6)}

    for ex in data.get("exercises") or []:
        sport = ex.get("sport")
        if sport:
            sports.append(sport)
        for z in (ex.get("zones") or {}).get("heart_rate") or []:
            zi  = z.get("zoneIndex")
            dur = iso_dur_to_min(z.get("inZone", "PT0S"))
            if zi in zones:
                zones[zi] += dur

    t_raw[date].append({
        "dur_min": dur_min, "calories": calories,
        "avg_hr": avg_hr,   "max_hr": max_hr,
        "sports": sports,   "zones": zones,
    })

training = {}
for date, sessions in t_raw.items():
    total_dur = sum(s["dur_min"] for s in sessions)
    total_cal = sum(s["calories"] for s in sessions)
    max_hrs   = [s["max_hr"] for s in sessions if s["max_hr"] != ""]
    avg_hrs   = [s["avg_hr"] for s in sessions if s["avg_hr"] != ""]
    sports    = sorted(set(sp for s in sessions for sp in s["sports"]))
    agg_z     = {z: sum(s["zones"][z] for s in sessions) for z in range(1, 6)}

    training[date] = {
        "training_sessions":  len(sessions),
        "training_total_min": r(total_dur, 0),
        "training_calories":  r(total_cal, 0),
        "training_avg_hr":    r(sum(avg_hrs)/len(avg_hrs), 0) if avg_hrs else "",
        "training_max_hr":    max(max_hrs) if max_hrs else "",
        "training_sport":     "|".join(sports),
        "zone1_min": r(agg_z[1], 0),
        "zone2_min": r(agg_z[2], 0),
        "zone3_min": r(agg_z[3], 0),
        "zone4_min": r(agg_z[4], 0),
        "zone5_min": r(agg_z[5], 0),
    }

print(f"  → {len(training)} días con entrenamiento")


# ── 6. ACTIVITY (pasos, calorías, MET) ──────────────────────────────────────

print("Cargando activity …")
activity = {}
for f in glob.glob(os.path.join(DATASET, "activity*.json")):
    data = json.load(open(f))
    date = data.get("date")
    if not date:
        continue
    s = data.get("summary") or {}
    activity[date] = {
        "steps":          s.get("stepCount", ""),
        "distance_m":     r(s.get("stepsDistance"), 0),
        "daily_calories": s.get("calories", ""),
        "met_minutes":    r(s.get("dailyMetMinutes"), 1),
    }

print(f"  → {len(activity)} días")


# ── 7. ORTHOSTATIC TEST ──────────────────────────────────────────────────────

print("Cargando orthostatic-test-result …")
ortho = {}
for f in glob.glob(os.path.join(DATASET, "orthostatic-test-result*.json")):
    data = json.load(open(f))
    start = data.get("startTime", "")
    if not start:
        continue
    date = start[:10]
    res = data.get("orthostaticTestResult") or {}
    ortho[date] = {
        "ortho_rmssd_supine":  res.get("rmssdSupine", ""),
        "ortho_rmssd_stand":   res.get("rmssdStand", ""),
        "ortho_rr_supine":     res.get("rrAvgSupine", ""),
        "ortho_rr_stand":      res.get("rrAvgStand", ""),
        "ortho_rr_min_standup": res.get("rrMinStandup", ""),
    }

print(f"  → {len(ortho)} tests ortostáticos")


# ── 8. JUMP TEST ─────────────────────────────────────────────────────────────

print("Cargando jump-test-result …")
jump = {}
for f in glob.glob(os.path.join(DATASET, "jump-test-result*.json")):
    data = json.load(open(f))
    start = data.get("startTime", "")
    if not start:
        continue
    date = start[:10]
    res   = data.get("jumpTestResult") or {}
    times = [j["flightTime"] for j in (res.get("jumps") or []) if "flightTime" in j]
    if not times:
        continue
    avg_ms = sum(times) / len(times)
    # h = g * (t/2)^2  → t en segundos, g=9.81
    avg_h_cm = 9.81 * (avg_ms / 1000 / 2) ** 2 * 100
    jump[date] = {
        "jump_flight_avg_ms":  r(avg_ms, 0),
        "jump_height_avg_cm":  r(avg_h_cm, 1),
        "jump_count":          len(times),
    }

print(f"  → {len(jump)} días con jump test")


# ── 9. MERGE Y ESCRITURA ─────────────────────────────────────────────────────

all_dates = sorted(set(
    list(sleep) + list(sscore) + list(recovery) +
    list(ppi_hrv) + list(training) + list(activity) +
    list(ortho) + list(jump)
))

COLUMNS = [
    "date",
    # — Sueño
    "sleep_duration_h", "sleep_asleep_h", "sleep_efficiency_pct",
    "sleep_rem_pct", "sleep_deep_pct", "sleep_wake_min",
    "sleep_interruptions", "sleep_long_interruptions",
    "sleep_score", "sleep_continuity_score", "sleep_efficiency_score",
    "sleep_rem_score", "sleep_n3_score", "sleep_long_int_score",
    # — HRV nocturno y recuperación
    "hrv_rmssd_night", "hrv_rri_mean_ms", "hrv_resp_ms",
    "recovery_indicator", "recovery_sublevel", "ans_status", "ans_rate",
    "baseline_rmssd",
    "vitality_tip", "exercise_tip", "sleep_tip",
    # — HRV diario (PPI raw)
    "hrv_rmssd_daily", "hrv_sdnn_daily", "ppi_count",
    # — Carga entrenamiento
    "training_sessions", "training_total_min", "training_calories",
    "training_avg_hr", "training_max_hr", "training_sport",
    "zone1_min", "zone2_min", "zone3_min", "zone4_min", "zone5_min",
    # — Actividad diaria
    "steps", "distance_m", "daily_calories", "met_minutes",
    # — Test ortostático
    "ortho_rmssd_supine", "ortho_rmssd_stand",
    "ortho_rr_supine", "ortho_rr_stand", "ortho_rr_min_standup",
    # — Jump test
    "jump_flight_avg_ms", "jump_height_avg_cm", "jump_count",
]

print(f"\nEscribiendo {len(all_dates)} filas → {OUTPUT}")
with open(OUTPUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for date in all_dates:
        row = {"date": date}
        row.update(sleep.get(date, {}))
        row.update(sscore.get(date, {}))
        row.update(recovery.get(date, {}))
        row.update(ppi_hrv.get(date, {}))
        row.update(training.get(date, {}))
        row.update(activity.get(date, {}))
        row.update(ortho.get(date, {}))
        row.update(jump.get(date, {}))
        writer.writerow(row)

print("✓ Listo.")
print(f"  Columnas: {len(COLUMNS)}")
print(f"  Filas:    {len(all_dates)}")
print(f"  Rango:    {all_dates[0]} → {all_dates[-1]}")
