"""Physics-informed multi-hazard dataset builder for training classification and risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geoagent_ai.config import DATA_DIR, FEATURE_COLUMNS


def _sample_class_rows(
    rng: np.random.Generator,
    label: str,
    n_samples: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for _ in range(n_samples):
        if label == "No Risk":
            rain_24 = rng.uniform(0.0, 42.0)
            rain_3 = rng.uniform(0.0, 12.0)
            wind = rng.uniform(4.0, 36.0)
            pressure = rng.uniform(1006.0, 1020.0)
            temp = rng.uniform(16.0, 34.0)
            humidity = rng.uniform(35.0, 78.0)
            soil = rng.uniform(0.15, 0.48)
            ndvi = rng.uniform(0.30, 0.78)
            ndwi = rng.uniform(-0.10, 0.20)
            hotspots = int(rng.integers(0, 2))
            frp = rng.uniform(0.0, 4.0) if hotspots else 0.0
            elev = rng.uniform(10.0, 1200.0)
            slope = rng.uniform(0.5, 16.0)
            seismic = rng.uniform(0.5, 2.8)
            severity = rng.uniform(4.0, 28.0)
        elif label == "Flood":
            rain_24 = rng.uniform(95.0, 360.0)
            rain_3 = rng.uniform(30.0, 130.0)
            wind = rng.uniform(15.0, 65.0)
            pressure = rng.uniform(992.0, 1008.0)
            temp = rng.uniform(21.0, 30.0)
            humidity = rng.uniform(80.0, 99.0)
            soil = rng.uniform(0.62, 0.98)
            ndvi = rng.uniform(0.15, 0.45)
            ndwi = rng.uniform(0.32, 0.78)
            hotspots = 0
            frp = 0.0
            elev = rng.uniform(2.0, 95.0)
            slope = rng.uniform(0.3, 6.5)
            seismic = rng.uniform(0.5, 2.5)
            severity = min(98.0, 32.0 + (rain_24 / 360.0) * 45.0 + ndwi * 25.0 - min(elev, 95.0) * 0.1)
        elif label == "Fire":
            rain_24 = rng.uniform(0.0, 5.0)
            rain_3 = rng.uniform(0.0, 1.0)
            wind = rng.uniform(18.0, 68.0)
            pressure = rng.uniform(1002.0, 1016.0)
            temp = rng.uniform(33.0, 47.0)
            humidity = rng.uniform(8.0, 34.0)
            soil = rng.uniform(0.04, 0.22)
            ndvi = rng.uniform(0.08, 0.35)
            ndwi = rng.uniform(-0.45, -0.05)
            hotspots = int(rng.integers(5, 65))
            frp = rng.uniform(22.0, 260.0)
            elev = rng.uniform(80.0, 1500.0)
            slope = rng.uniform(2.0, 24.0)
            seismic = rng.uniform(0.5, 2.2)
            severity = min(98.0, 34.0 + (hotspots / 65.0) * 32.0 + (frp / 260.0) * 24.0 + (temp - 30.0) * 0.8)
        elif label == "Cyclone":
            rain_24 = rng.uniform(85.0, 320.0)
            rain_3 = rng.uniform(25.0, 115.0)
            wind = rng.uniform(76.0, 210.0)
            pressure = rng.uniform(930.0, 991.0)
            temp = rng.uniform(24.0, 30.0)
            humidity = rng.uniform(84.0, 99.0)
            soil = rng.uniform(0.50, 0.92)
            ndvi = rng.uniform(0.20, 0.50)
            ndwi = rng.uniform(0.25, 0.65)
            hotspots = 0
            frp = 0.0
            elev = rng.uniform(1.0, 65.0)
            slope = rng.uniform(0.2, 5.0)
            seismic = rng.uniform(0.5, 2.4)
            severity = min(99.0, 36.0 + ((wind - 70.0) / 140.0) * 42.0 + ((1000.0 - pressure) / 70.0) * 22.0)
        elif label == "Landslide":
            rain_24 = rng.uniform(105.0, 340.0)
            rain_3 = rng.uniform(28.0, 110.0)
            wind = rng.uniform(12.0, 52.0)
            pressure = rng.uniform(994.0, 1009.0)
            temp = rng.uniform(12.0, 27.0)
            humidity = rng.uniform(82.0, 99.0)
            soil = rng.uniform(0.68, 0.98)
            ndvi = rng.uniform(0.20, 0.62)
            ndwi = rng.uniform(0.22, 0.58)
            hotspots = 0
            frp = 0.0
            elev = rng.uniform(280.0, 2800.0)
            slope = rng.uniform(20.0, 48.0)
            seismic = rng.uniform(0.8, 4.1)
            severity = min(98.0, 35.0 + ((slope - 18.0) / 30.0) * 32.0 + (rain_24 / 340.0) * 24.0 + soil * 10.0)
        else:  # Earthquake
            rain_24 = rng.uniform(0.0, 35.0)
            rain_3 = rng.uniform(0.0, 10.0)
            wind = rng.uniform(5.0, 35.0)
            pressure = rng.uniform(1004.0, 1018.0)
            temp = rng.uniform(14.0, 38.0)
            humidity = rng.uniform(25.0, 75.0)
            soil = rng.uniform(0.15, 0.50)
            ndvi = rng.uniform(0.15, 0.65)
            ndwi = rng.uniform(-0.15, 0.20)
            hotspots = int(rng.integers(0, 3))
            frp = rng.uniform(0.0, 8.0) if hotspots else 0.0
            elev = rng.uniform(15.0, 2200.0)
            slope = rng.uniform(1.0, 28.0)
            seismic = rng.uniform(4.4, 8.1)
            severity = min(99.0, 32.0 + ((seismic - 4.0) / 4.1) * 62.0)

        severity = float(np.clip(severity + rng.normal(0.0, 2.5), 2.0, 99.0))
        rows.append(
            {
                "rainfall_mm_24h": round(float(rain_24), 2),
                "rainfall_mm_3h": round(float(rain_3), 2),
                "wind_speed_kmh": round(float(wind), 2),
                "pressure_hpa": round(float(pressure), 2),
                "temperature_c": round(float(temp), 2),
                "humidity_pct": round(float(humidity), 2),
                "soil_moisture": round(float(soil), 3),
                "ndvi": round(float(ndvi), 3),
                "ndwi": round(float(ndwi), 3),
                "fire_hotspots_count": int(hotspots),
                "frp_mw": round(float(frp), 2),
                "elevation_m": round(float(elev), 1),
                "slope_deg": round(float(slope), 1),
                "seismic_magnitude": round(float(seismic), 2),
                "disaster_type": label,
                "severity_score": round(severity, 2),
            }
        )
    return rows


def generate_training_dataframe(samples_per_class: int = 320, seed: int = 42) -> pd.DataFrame:
    """Build balanced multi-hazard training DataFrame and persist CSV for reproducibility."""
    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, float | str]] = []
    for label in ["No Risk", "Flood", "Fire", "Cyclone", "Landslide", "Earthquake"]:
        all_rows.extend(_sample_class_rows(rng, label, samples_per_class))

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "training_dataset.csv", index=False)
    return df


def extract_feature_row(data_dict: dict[str, float]) -> pd.DataFrame:
    """Convert raw multi-source feature dict into ordered DataFrame."""
    return pd.DataFrame([{col: float(data_dict.get(col, 0.0)) for col in FEATURE_COLUMNS}])
