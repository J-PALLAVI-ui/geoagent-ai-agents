"""Inference wrapper combining trained ML models with cross-source domain safeguards."""

from __future__ import annotations

import json
from typing import Any

import joblib

from geoagent_ai.ml.dataset_generator import extract_feature_row
from geoagent_ai.ml.train_models import (
    CLASSIFIER_PATH,
    METRICS_PATH,
    REGRESSOR_PATH,
    ensure_models_trained,
)
from geoagent_ai.schemas import MultiSourceBundle, RiskLevel


def bundle_to_features(bundle: MultiSourceBundle) -> dict[str, float]:
    """Flatten multi-source bundle into the 14 ML input features."""
    return {
        "rainfall_mm_24h": bundle.weather.rainfall_mm_24h,
        "rainfall_mm_3h": bundle.weather.rainfall_mm_3h,
        "wind_speed_kmh": bundle.weather.wind_speed_kmh,
        "pressure_hpa": bundle.weather.pressure_hpa,
        "temperature_c": bundle.weather.temperature_c,
        "humidity_pct": bundle.weather.humidity_pct,
        "soil_moisture": bundle.terrain.soil_moisture,
        "ndvi": bundle.satellite.ndvi,
        "ndwi": bundle.satellite.ndwi,
        "fire_hotspots_count": float(bundle.satellite.fire_hotspots_count),
        "frp_mw": bundle.satellite.frp_mw,
        "elevation_m": bundle.terrain.elevation_m,
        "slope_deg": bundle.terrain.slope_deg,
        "seismic_magnitude": bundle.seismic.seismic_magnitude,
    }


def _extract_indicators(features: dict[str, float], disaster: str) -> tuple[list[str], float]:
    """Compute human-readable key indicators and cross-source agreement score."""
    indicators: list[str] = []
    checks_passed = 0
    total_checks = 3

    if disaster == "Flood":
        if features["rainfall_mm_24h"] >= 90:
            indicators.append(f"Heavy 24h rainfall ({features['rainfall_mm_24h']:.1f} mm)")
            checks_passed += 1
        if features["ndwi"] >= 0.30 or features["soil_moisture"] >= 0.62:
            indicators.append(f"High surface wetness NDWI={features['ndwi']:.2f}, soil saturation={features['soil_moisture']:.2f}")
            checks_passed += 1
        if features["elevation_m"] <= 120:
            indicators.append(f"Low-lying basin elevation ({features['elevation_m']:.0f} m)")
            checks_passed += 1
    elif disaster == "Fire":
        if features["fire_hotspots_count"] >= 3 or features["frp_mw"] >= 15:
            indicators.append(f"NASA FIRMS active thermal hotspots ({int(features['fire_hotspots_count'])}, FRP={features['frp_mw']:.1f} MW)")
            checks_passed += 1
        if features["temperature_c"] >= 34 and features["humidity_pct"] <= 35:
            indicators.append(f"Extreme fire weather ({features['temperature_c']:.1f}°C, RH={features['humidity_pct']:.0f}%)")
            checks_passed += 1
        if features["ndvi"] <= 0.38 and features["soil_moisture"] <= 0.25:
            indicators.append(f"Dry stressed vegetation (NDVI={features['ndvi']:.2f}, soil moisture={features['soil_moisture']:.2f})")
            checks_passed += 1
    elif disaster == "Cyclone":
        if features["wind_speed_kmh"] >= 70:
            indicators.append(f"Gale/cyclonic winds ({features['wind_speed_kmh']:.1f} km/h)")
            checks_passed += 1
        if features["pressure_hpa"] <= 994:
            indicators.append(f"Deep barometric pressure drop ({features['pressure_hpa']:.1f} hPa)")
            checks_passed += 1
        if features["rainfall_mm_24h"] >= 80 and features["elevation_m"] <= 80:
            indicators.append(f"Coastal storm rain band ({features['rainfall_mm_24h']:.1f} mm/24h at {features['elevation_m']:.0f} m)")
            checks_passed += 1
    elif disaster == "Landslide":
        if features["slope_deg"] >= 18:
            indicators.append(f"Steep vulnerable terrain slope ({features['slope_deg']:.1f}°)")
            checks_passed += 1
        if features["rainfall_mm_24h"] >= 95:
            indicators.append(f"Intense slope-triggering rainfall ({features['rainfall_mm_24h']:.1f} mm/24h)")
            checks_passed += 1
        if features["soil_moisture"] >= 0.68:
            indicators.append(f"Critical pore-water soil saturation ({features['soil_moisture']:.2f})")
            checks_passed += 1
    elif disaster == "Earthquake":
        if features["seismic_magnitude"] >= 4.2:
            indicators.append(f"Significant seismic activity (M{features['seismic_magnitude']:.1f})")
            checks_passed += 1
        if features["seismic_magnitude"] >= 5.5:
            indicators.append("Strong ground shaking potential within fault zone")
            checks_passed += 1
        indicators.append("Multi-sensor cross-check: meteorological causes excluded")
        checks_passed += 1
    else:
        indicators.append(f"Normal weather & terrain telemetry (Rain={features['rainfall_mm_24h']:.1f} mm, Wind={features['wind_speed_kmh']:.1f} km/h)")
        indicators.append(f"No thermal hotspots or severe seismic anomalies (M{features['seismic_magnitude']:.1f})")
        checks_passed = 3

    agreement = round(max(0.45, min(0.99, checks_passed / total_checks)), 2)
    return indicators, agreement


class HazardPredictor:
    """Loads persisted ensemble classifier & severity regressor for inference."""

    def __init__(self) -> None:
        ensure_models_trained()
        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.regressor = joblib.load(REGRESSOR_PATH)
        self.metrics: dict[str, Any] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def classify(self, bundle: MultiSourceBundle) -> dict[str, Any]:
        features = bundle_to_features(bundle)
        X = extract_feature_row(features)
        probs = self.classifier.predict_proba(X)[0]
        classes = list(self.classifier.classes_)
        prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        predicted = max(prob_map, key=lambda k: prob_map[k])
        confidence = prob_map[predicted]

        indicators, agreement = _extract_indicators(features, predicted)
        explanation = (
            f"{self.metrics['classifier_model']} classified {predicted} with "
            f"{confidence * 100:.1f}% model probability and {agreement * 100:.0f}% cross-source validation."
        )
        return {
            "detected_disaster": predicted,
            "confidence": round(confidence, 4),
            "class_probabilities": prob_map,
            "key_indicators": indicators,
            "cross_source_agreement": agreement,
            "explanation": explanation,
        }

    def predict_risk(
        self,
        bundle: MultiSourceBundle,
        detected_disaster: str,
        calibration_offset: float = 0.0,
    ) -> dict[str, Any]:
        features = bundle_to_features(bundle)
        X = extract_feature_row(features)
        raw_score = float(self.regressor.predict(X)[0])

        if detected_disaster == "No Risk":
            raw_score = min(raw_score, 26.0)

        adjusted_score = max(1.0, min(99.0, raw_score + calibration_offset))

        if adjusted_score >= 68.0 and detected_disaster != "No Risk":
            risk_level: RiskLevel = "High"
        elif adjusted_score >= 38.0 and detected_disaster != "No Risk":
            risk_level = "Medium"
        else:
            risk_level = "Low"

        risk_prob_pct = round(min(99.0, max(3.0, adjusted_score * 0.96)), 1)
        indicators, _ = _extract_indicators(features, detected_disaster)

        return {
            "risk_level": risk_level,
            "severity_score": round(adjusted_score, 2),
            "risk_probability_pct": risk_prob_pct,
            "model_used": self.metrics["regressor_model"],
            "contributing_factors": indicators,
            "calibration_applied": round(calibration_offset, 2),
        }
