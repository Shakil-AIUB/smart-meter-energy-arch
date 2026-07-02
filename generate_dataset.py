"""
Synthetic dataset generator for the Building Energy Optimisation project.

Generates a realistic-looking time series of:
- sub-metered loads (HVAC, lighting, plug loads)
- occupancy (0..1, fraction of building occupied)
- indoor temperature (affected by HVAC + occupancy + outdoor temp)
- outdoor temperature (simple seasonal + daily cycle)

Why simulate instead of using real hardware:
Real smart meters / occupancy sensors are not required for this course project.
A believable simulation with clear, documented assumptions is an accepted
substitute for the prototype (see the guideline, Section 8).

Usage:
    python generate_dataset.py --days 14 --interval-min 5 --out building_energy.csv

Output columns:
    timestamp, occupancy, hvac_kw, lighting_kw, plug_kw, total_kw,
    outdoor_temp_c, indoor_temp_c
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_dataset(days: int, interval_min: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = datetime(2026, 1, 5, 0, 0, 0)  # a Monday, for clean weekday/weekend cycles
    n_steps = int(days * 24 * 60 / interval_min)
    timestamps = [start + timedelta(minutes=interval_min * i) for i in range(n_steps)]

    hour_of_day = np.array([t.hour + t.minute / 60 for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])  # 0=Mon ... 6=Sun
    day_index = np.array([(t - start).days for t in timestamps])
    is_weekend = (day_of_week >= 5).astype(float)

    # ---- Occupancy: office-like pattern, ~9am-6pm on weekdays, near-zero weekends ----
    # Smooth "bell" shaped occupancy centred at 13:00, width controls how sharp it is.
    occ_weekday = np.exp(-((hour_of_day - 13.0) ** 2) / (2 * 3.2 ** 2))
    occ_weekday = np.clip(occ_weekday, 0, 1)
    occ_weekend = 0.05 * np.exp(-((hour_of_day - 13.0) ** 2) / (2 * 5.0 ** 2))  # skeleton crew
    occupancy = np.where(is_weekend == 1, occ_weekend, occ_weekday)
    occupancy += rng.normal(0, 0.03, size=n_steps)  # sensor/behavioural noise
    occupancy = np.clip(occupancy, 0, 1)

    # ---- Outdoor temperature: seasonal drift + daily cycle + noise ----
    seasonal = 8 * np.sin(2 * np.pi * day_index / 365.0 - np.pi / 2) + 15  # deg C, mild seasonal swing
    daily_cycle = 5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)  # warmest mid-afternoon
    outdoor_temp = seasonal + daily_cycle + rng.normal(0, 0.8, size=n_steps)

    # ---- HVAC load: driven by occupancy + how far outdoor temp is from comfort band ----
    comfort_mid = 22.0  # deg C target
    temp_gap = np.abs(outdoor_temp - comfort_mid)
    hvac_base = 0.6 * temp_gap  # more load when it's far from comfortable outside
    hvac_occupancy_boost = 3.0 * occupancy  # more load when building is occupied
    hvac_kw = np.clip(hvac_base + hvac_occupancy_boost + rng.normal(0, 0.3, size=n_steps), 0, None)

    # ---- Lighting load: mostly occupancy-driven, small baseline for corridors/security ----
    daylight_factor = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0, 1)  # ~6am-6pm daylight
    lighting_kw = 0.3 + 2.5 * occupancy * (1 - 0.5 * daylight_factor) + rng.normal(0, 0.1, size=n_steps)
    lighting_kw = np.clip(lighting_kw, 0.1, None)

    # ---- Plug loads: computers/equipment, follows occupancy with more inertia/noise ----
    plug_kw = 0.5 + 1.8 * occupancy + rng.normal(0, 0.2, size=n_steps)
    plug_kw = np.clip(plug_kw, 0.2, None)

    total_kw = hvac_kw + lighting_kw + plug_kw

    # ---- Indoor temperature: drifts toward comfort_mid, pulled by HVAC effort, pushed by outdoor ----
    indoor_temp = np.empty(n_steps)
    indoor_temp[0] = comfort_mid
    for i in range(1, n_steps):
        # simple first-order model: indoor temp moves toward outdoor temp unless HVAC counteracts it
        drift_to_outdoor = 0.02 * (outdoor_temp[i] - indoor_temp[i - 1])
        hvac_correction = -0.15 * np.sign(indoor_temp[i - 1] - comfort_mid) * min(hvac_kw[i], 4.0) / 4.0
        noise = rng.normal(0, 0.05)
        indoor_temp[i] = indoor_temp[i - 1] + drift_to_outdoor + hvac_correction + noise

    df = pd.DataFrame({
        "timestamp": timestamps,
        "occupancy": np.round(occupancy, 3),
        "hvac_kw": np.round(hvac_kw, 3),
        "lighting_kw": np.round(lighting_kw, 3),
        "plug_kw": np.round(plug_kw, 3),
        "total_kw": np.round(total_kw, 3),
        "outdoor_temp_c": np.round(outdoor_temp, 2),
        "indoor_temp_c": np.round(indoor_temp, 2),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic building energy dataset.")
    parser.add_argument("--days", type=int, default=14, help="Number of days to simulate (default: 14)")
    parser.add_argument("--interval-min", type=int, default=5, help="Sampling interval in minutes (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="building_energy.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(days=args.days, interval_min=args.interval_min, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({args.days} days at {args.interval_min}-min intervals) to {args.out}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
