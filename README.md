# GeoAgent AI: Autonomous Disaster Monitoring and Response System

## Overview

**GeoAgent AI** is an autonomous multi-agent system built with **LangGraph**, **Scikit-learn**, **FastAPI**, **Leaflet/Folium**, and **SQLite** that monitors any user-selected region and coordinates end-to-end disaster detection, geospatial impact mapping, risk prediction, automated alerting, and emergency response planning.

Unlike manual monitoring tools, the user supplies **only a location or region** (by name, preset, or clicking anywhere on the map). Six autonomous agents collaborate sequentially via a shared `LangGraph` `StateGraph`:

1. **Data Collection Agent** (`geoagent_ai/agents/data_collection.py`) — Aggregates OpenWeatherMap / Open-Meteo weather, NASA FIRMS fire hotspots, Sentinel-2 spectral indices (`NDVI`, `NDWI`, `NBR`), SRTM DEM elevation/slope, USGS earthquake feeds, and nearby historical disaster records with automatic fault-tolerant fallback.
2. **Disaster Classification Agent** (`geoagent_ai/agents/hazard_classification.py`) — Uses an Ensemble ML Classifier (`RandomForest` + `HistGradientBoosting` + `RBF-SVM`, plus `XGBoost` if installed) combined with cross-source indicator validation to classify `Flood`, `Fire`, `Cyclone`, `Landslide`, `Earthquake`, or `No Risk`.
3. **Geo-Location Agent** (`geoagent_ai/agents/geo_location.py`) — Computes GeoJSON impact polygons, affected area (`km²`), monitoring nodes, and estimated population exposure.
4. **Risk Prediction Agent** (`geoagent_ai/agents/risk_prediction.py`) — Predicts severity (`0–100`), risk probability (`%`), and risk level (`Low` / `Medium` / `High`), calibrated dynamically via the SQLite authority feedback loop.
5. **Alert & Communication Agent** (`geoagent_ai/agents/alert_agent.py`) — Automatically triggers structured SMS, Email, and Push advisories whenever risk is `Medium` or `High` (suppresses alerts on `Low` / `No Risk`), with optional live SMTP/Webhook dispatch.
6. **Response Planning Agent** (`geoagent_ai/agents/response_agent.py`) — Recommends disaster-specific actions, citizen safety measures, resource allocations, Haversine-ranked nearest relief shelters, multi-waypoint evacuation routes, and regional emergency contacts.

---

## Quick Start

### 1. Train ML Models (runs automatically if artifacts are absent)
```bash
python run.py train
```

### 2. Run the Web Command Center (FastAPI + Interactive Leaflet Map)
```bash
python run.py server
```
Open `http://127.0.0.1:8000` in your browser.

### 3. Run via CLI
```bash
python run.py analyze --location "Ernakulam, Kerala" --mode simulation
python run.py analyze --location "Bengaluru" --mode live
```

### 4. Optional: Run the Streamlit + Folium Dashboard
If `streamlit` and `folium` are installed (`pip install streamlit folium xgboost`):
```bash
python run.py streamlit
```

### 5. Run Automated Test Suite
```bash
python -m unittest discover -s tests -v
```

---

## Optional Live API Environment Variables
GeoAgent AI works **immediately out of the box** using free Open-Meteo / Open-Elevation / USGS feeds and offline regional baselines. To enable authenticated feeds or live notifications, set:
- `OPENWEATHER_API_KEY` — OpenWeatherMap live weather API
- `NASA_FIRMS_MAP_KEY` — NASA FIRMS VIIRS active fire hotspots API
- `ALERT_WEBHOOK_URL` — Webhook endpoint for live alert push
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` — Live email alert dispatch
