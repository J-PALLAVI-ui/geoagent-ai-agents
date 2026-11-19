const state = {
  map: null,
  layerGroup: null,
  lastResult: null,
  regions: [],
};

const RISK_COLORS = {
  High: "#ef4444",
  Medium: "#f59e0b",
  Low: "#10b981",
};

function initMap() {
  const container = document.getElementById("mapContainer");
  if (typeof L === "undefined") {
    container.innerHTML =
      '<div style="padding:24px;color:#93a4c3;font-size:0.85rem;">Leaflet CDN unavailable offline — geospatial coordinates, polygons, shelters, and routes are computed in the panels on the right.</div>';
    return;
  }

  state.map = L.map("mapContainer").setView([15.5, 78.5], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);

  state.layerGroup = L.layerGroup().addTo(state.map);

  state.map.on("click", (e) => {
    const lat = Number(e.latlng.lat.toFixed(4));
    const lon = Number(e.latlng.lng.toFixed(4));
    document.getElementById("locationInput").value = `Custom Map Pin (${lat}, ${lon})`;
    document.getElementById("latInput").value = String(lat);
    document.getElementById("lonInput").value = String(lon);
    runAnalysis();
  });
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

async function loadInitialMetadata() {
  try {
    const [healthRes, regRes] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/regions"),
    ]);
    const health = await healthRes.json();
    const regData = await regRes.json();

    const acc = (health.model_metrics.classification_accuracy * 100).toFixed(1);
    document.getElementById("healthBadge").textContent = `System Online · ML Acc ${acc}%`;

    state.regions = regData.regions || [];
    const chipsBox = document.getElementById("regionChips");
    chipsBox.innerHTML = "";
    state.regions.forEach((r) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = r.name;
      btn.addEventListener("click", () => {
        document.getElementById("locationInput").value = r.name;
        document.getElementById("latInput").value = "";
        document.getElementById("lonInput").value = "";
        document.querySelectorAll("#regionChips .chip").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        runAnalysis();
      });
      chipsBox.appendChild(btn);
    });

    const scenarioSelect = document.getElementById("scenarioSelect");
    (regData.scenarios || []).forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s.replaceAll("_", " ").toUpperCase();
      scenarioSelect.appendChild(opt);
    });
  } catch (err) {
    document.getElementById("healthBadge").textContent = "API Connection Error";
  }
}

async function runAnalysis() {
  const btn = document.getElementById("analyzeBtn");
  const locationName = document.getElementById("locationInput").value.trim();
  if (!locationName) return;

  const latVal = document.getElementById("latInput").value;
  const lonVal = document.getElementById("lonInput").value;
  const mode = document.getElementById("modeSelect").value;
  const scenario = document.getElementById("scenarioSelect").value || null;

  btn.disabled = true;
  btn.textContent = "Agents Collaborating...";

  try {
    const payload = {
      location_name: locationName,
      latitude: latVal ? parseFloat(latVal) : null,
      longitude: lonVal ? parseFloat(lonVal) : null,
      mode,
      scenario,
    };

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.lastResult = data;
    renderAll(data);
    await refreshMonitoring();
  } catch (err) {
    alert(`Failed to execute GeoAgent pipeline: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Autonomous Agents";
  }
}

function renderAll(data) {
  renderPipeline(data);
  renderMap(data);
  renderTelemetry(data);
  renderHistorical(data);
  renderHazardTab(data);
  renderAlertsTab(data);
  renderResponseTab(data);
  document.getElementById("calibrationBadge").textContent = `Offset: ${data.risk_assessment.calibration_applied}`;
}

function renderPipeline(data) {
  document.getElementById("runMeta").textContent =
    `Run ${data.run_id} · Total Latency: ${data.total_duration_ms} ms`;

  const grid = document.getElementById("agentPipeline");
  grid.innerHTML = "";
  data.agent_logs.forEach((log, idx) => {
    const card = document.createElement("div");
    card.className = `agent-card status-${log.status}`;
    card.innerHTML = `
      <div class="agent-top">
        <span>Step ${idx + 1} · ${log.status.toUpperCase()}</span>
        <span>${log.duration_ms} ms</span>
      </div>
      <div class="agent-title">${log.agent_name}</div>
      <div class="agent-summary">${log.summary}</div>
    `;
    grid.appendChild(card);
  });
}

function renderMap(data) {
  if (!state.map || !state.layerGroup) return;
  state.layerGroup.clearLayers();

  const geo = data.geo_analysis;
  const risk = data.risk_assessment.risk_level;
  const hazard = data.classification.detected_disaster;
  const color = RISK_COLORS[risk] || "#3b82f6";

  const polyLayer = L.geoJSON(geo.affected_polygon_geojson, {
    style: {
      color,
      fillColor: color,
      fillOpacity: 0.24,
      weight: 2.5,
    },
  }).bindPopup(
    `<strong>${data.collected_data.resolved_location}</strong><br/>Hazard: ${hazard} (${risk} Risk)<br/>Area: ${geo.affected_area_km2} km²`
  );
  polyLayer.addTo(state.layerGroup);

  L.circleMarker([geo.center_lat, geo.center_lon], {
    radius: 8,
    color: "#fff",
    weight: 2,
    fillColor: color,
    fillOpacity: 0.95,
  })
    .bindPopup(
      `<strong>Monitored Center</strong><br/>${data.collected_data.resolved_location}<br/>Severity: ${data.risk_assessment.severity_score}/100`
    )
    .addTo(state.layerGroup);

  (geo.hotspot_points_geojson || []).forEach((pt) => {
    const [lon, lat] = pt.geometry.coordinates;
    L.circleMarker([lat, lon], {
      radius: 4.5,
      color: "#fde047",
      fillColor: color,
      fillOpacity: 0.85,
      weight: 1.5,
    })
      .bindPopup(`${pt.properties.label} (${pt.properties.id})`)
      .addTo(state.layerGroup);
  });

  data.response_plan.nearest_shelters.forEach((s) => {
    L.circleMarker([s.latitude, s.longitude], {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: "#3b82f6",
      fillOpacity: 0.95,
    })
      .bindPopup(
        `<strong>${s.name}</strong><br/>${s.type} · Cap: ${s.capacity}<br/>Distance: ${s.distance_km} km`
      )
      .addTo(state.layerGroup);
  });

  data.response_plan.evacuation_routes.forEach((route, idx) => {
    L.polyline(route.waypoints, {
      color: idx === 0 ? "#60a5fa" : "#93c5fd",
      weight: idx === 0 ? 3.5 : 2,
      dashArray: "7, 6",
    })
      .bindPopup(
        `<strong>${route.route_id}</strong> → ${route.target_shelter}<br/>${route.distance_km} km (~${route.estimated_time_min} min) · ${route.safety_status}`
      )
      .addTo(state.layerGroup);
  });

  const bounds = polyLayer.getBounds();
  if (bounds.isValid()) {
    state.map.fitBounds(bounds.pad(0.35));
  } else {
    state.map.setView([geo.center_lat, geo.center_lon], 11);
  }
}

function renderTelemetry(data) {
  const c = data.collected_data;
  const w = c.weather;
  const sat = c.satellite;
  const ter = c.terrain;
  const sei = c.seismic;

  document.getElementById("telemetryGrid").innerHTML = `
    <div class="tele-box">
      <h4>Weather (${w.source})</h4>
      <p>Rainfall: <strong>${w.rainfall_mm_24h} mm/24h</strong> (${w.rainfall_mm_3h} mm/3h)</p>
      <p>Wind: <strong>${w.wind_speed_kmh} km/h</strong> · Pressure: <strong>${w.pressure_hpa} hPa</strong></p>
      <p>Temp: ${w.temperature_c}°C · Humidity: ${w.humidity_pct}%</p>
    </div>
    <div class="tele-box">
      <h4>Satellite &amp; Fire (${sat.source})</h4>
      <p>NASA FIRMS Hotspots: <strong>${sat.fire_hotspots_count}</strong> (FRP: ${sat.frp_mw} MW)</p>
      <p>NDVI: <strong>${sat.ndvi}</strong> · NDWI: <strong>${sat.ndwi}</strong> · NBR: ${sat.nbr}</p>
      <p>Cloud Cover: ${sat.cloud_cover_pct}%</p>
    </div>
    <div class="tele-box">
      <h4>SRTM / DEM Terrain (${ter.source})</h4>
      <p>Elevation: <strong>${ter.elevation_m} m</strong> · Slope: <strong>${ter.slope_deg}°</strong></p>
      <p>Soil Moisture: <strong>${ter.soil_moisture}</strong> · Drainage: ${ter.drainage_density}</p>
      <p>Cover: ${ter.land_cover}</p>
    </div>
    <div class="tele-box">
      <h4>Seismic Monitor (${sei.source})</h4>
      <p>Peak Magnitude: <strong>M${sei.seismic_magnitude}</strong></p>
      <p>Depth: ${sei.depth_km} km · Recent Events (7d): ${sei.recent_events_count}</p>
    </div>
  `;

  const fbBox = document.getElementById("fallbackBanner");
  if (c.fallbacks_triggered && c.fallbacks_triggered.length > 0) {
    fbBox.classList.remove("hidden");
    fbBox.textContent = `Fault Handling Active: ${c.fallbacks_triggered.join(" | ")}`;
  } else {
    fbBox.classList.add("hidden");
  }
}

function renderHistorical(data) {
  const box = document.getElementById("historicalList");
  const recs = data.collected_data.historical_records || [];
  if (!recs.length) {
    box.innerHTML = '<div class="list-card">No matching historical records within 450 km in baseline dataset.</div>';
    return;
  }
  box.innerHTML = recs
    .map(
      (r) => `
      <div class="list-card">
        <strong>${r.year} · ${r.disaster_type} (${r.severity})</strong> — ${r.region} (${r.distance_km} km away)<br/>
        <span class="muted-text">${r.casualties_or_impact}</span>
      </div>`
    )
    .join("");
}

function renderHazardTab(data) {
  const cls = data.classification;
  const risk = data.risk_assessment;
  const geo = data.geo_analysis;

  document.getElementById("hazardSummary").innerHTML = `
    <div>
      <span class="muted-text">${data.collected_data.resolved_location} · ${data.collected_data.state_or_country}</span>
      <div class="hazard-title">${cls.detected_disaster}</div>
      <span class="risk-pill risk-${risk.risk_level}">${risk.risk_level} Risk</span>
      <span class="muted-text" style="margin-left:8px;">Agreement: ${(cls.cross_source_agreement * 100).toFixed(0)}%</span>
    </div>
    <div class="score-ring">
      <span class="muted-text">Severity Score</span>
      <div class="score-big">${risk.severity_score.toFixed(1)}<span style="font-size:0.9rem;color:#93a4c3;"> / 100</span></div>
      <span class="muted-text">Risk Prob: ${risk.risk_probability_pct}%</span>
    </div>
  `;

  document.getElementById("classificationExplanation").textContent = cls.explanation;
  document.getElementById("indicatorsList").innerHTML = cls.key_indicators
    .map((i) => `<li>${i}</li>`)
    .join("");

  const sortedProbs = Object.entries(cls.class_probabilities).sort((a, b) => b[1] - a[1]);
  document.getElementById("probBars").innerHTML = sortedProbs
    .map(
      ([label, p]) => `
      <div class="prob-row">
        <span>${label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <span>${(p * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("");

  document.getElementById("geoMetrics").innerHTML = `
    <div class="kv-item"><span>Impact Radius</span><strong>${geo.impact_radius_km} km</strong></div>
    <div class="kv-item"><span>Affected Area</span><strong>${geo.affected_area_km2} km²</strong></div>
    <div class="kv-item"><span>Exposed Population (Est.)</span><strong>${geo.estimated_exposed_population.toLocaleString()}</strong></div>
    <div class="kv-item"><span>Zone Classification</span><strong>${geo.zone_classification}</strong></div>
  `;
}

function renderAlertsTab(data) {
  const a = data.alert;
  const box = document.getElementById("alertStatusBox");
  box.className = "alert-banner " + (a.alert_triggered ? (a.severity_level === "High" ? "alert-active" : "alert-medium") : "alert-safe");
  box.innerHTML = `
    <strong>${a.headline}</strong><br/>
    <span>${a.alert_message}</span><br/>
    <span style="font-size:0.75rem;opacity:0.85;">ID: ${a.alert_id} · Status: ${a.dispatch_status}</span>
  `;

  document.getElementById("smsPreview").textContent = a.sms_preview;
  document.getElementById("emailSubject").textContent = a.email_subject;
  document.getElementById("emailBody").textContent = a.email_body;
  document.getElementById("pushPreview").textContent = a.push_notification;
  document.getElementById("channelBadges").innerHTML = (a.channels.length ? a.channels : ["Suppressed (Low Risk)"])
    .map((c) => `<span class="chip">${c}</span>`)
    .join("");
}

function renderResponseTab(data) {
  const plan = data.response_plan;
  const risk = data.risk_assessment.risk_level;
  const banner = document.getElementById("responsePriority");
  banner.className = "priority-banner " + (risk === "High" ? "alert-active" : risk === "Medium" ? "alert-medium" : "alert-safe");
  banner.innerHTML = `<strong>Response Priority:</strong> ${plan.priority_level}`;

  document.getElementById("actionsList").innerHTML = plan.recommended_actions.map((x) => `<li>${x}</li>`).join("");
  document.getElementById("safetyList").innerHTML = plan.safety_measures.map((x) => `<li>${x}</li>`).join("");
  document.getElementById("resourcesList").innerHTML = plan.resource_allocation.map((x) => `<li>${x}</li>`).join("");

  const routeMap = new Map(plan.evacuation_routes.map((r) => [r.target_shelter, r]));
  const sheltersBox = document.getElementById("sheltersList");
  sheltersBox.innerHTML = "";
  plan.nearest_shelters.forEach((s) => {
    const route = routeMap.get(s.name);
    const div = document.createElement("div");
    div.className = "list-card clickable";
    div.innerHTML = `
      <strong>${s.name}</strong> (${s.type})<br/>
      <span>Distance: ${s.distance_km} km · Capacity: ${s.capacity.toLocaleString()}</span>
      ${route ? `<br/><span class="muted-text">${route.route_id}: ~${route.estimated_time_min} min (${route.road_dist || route.distance_km} km) · ${route.safety_status}</span>` : ""}
    `;
    div.addEventListener("click", () => {
      if (state.map) state.map.setView([s.latitude, s.longitude], 13);
    });
    sheltersBox.appendChild(div);
  });

  document.getElementById("contactsList").innerHTML = plan.emergency_contacts
    .map(
      (c) => `
      <div class="kv-item">
        <span>${c.role}</span>
        <strong>${c.agency}: ${c.number}</strong>
      </div>`
    )
    .join("");
}

async function refreshMonitoring() {
  try {
    const res = await fetch("/api/monitoring/summary");
    const payload = await res.json();
    const m = payload.model_metrics;

    document.getElementById("modelMetricsBox").innerHTML = `
      <div class="kv-item"><span>Model Family</span><strong>${m.classifier_model}</strong></div>
      <div class="kv-item"><span>Classification Accuracy / F1</span><strong>${(m.classification_accuracy * 100).toFixed(1)}% / ${(m.classification_weighted_f1 * 100).toFixed(1)}%</strong></div>
      <div class="kv-item"><span>Severity MAE / R²</span><strong>${m.severity_mae} / ${m.severity_r2}</strong></div>
      <div class="kv-item"><span>Total Runs / Alerts</span><strong>${payload.summary.total_pipeline_runs} runs · ${payload.summary.total_alerts_issued} alerts</strong></div>
    `;

    const agents = payload.summary.agents || [];
    document.getElementById("agentRegistryTable").innerHTML = `
      <table>
        <thead><tr><th>Agent</th><th>Runs</th><th>Avg ms</th><th>Fallbacks</th></tr></thead>
        <tbody>
          ${agents
            .map(
              (a) =>
                `<tr><td>${a.name}</td><td>${a.invocations}</td><td>${a.avg_duration_ms}</td><td>${a.fallback_count}</td></tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;

    const watchBox = document.getElementById("watchlistBox");
    watchBox.innerHTML = "";
    (payload.watchlist || []).forEach((w) => {
      const card = document.createElement("div");
      card.className = "list-card clickable";
      card.innerHTML = `<strong>${w.location_name}</strong> — Last: ${w.last_disaster} (${w.last_risk} Risk)`;
      card.addEventListener("click", () => {
        document.getElementById("locationInput").value = w.location_name;
        document.getElementById("latInput").value = String(w.latitude);
        document.getElementById("lonInput").value = String(w.longitude);
        runAnalysis();
      });
      watchBox.appendChild(card);
    });

    const alertsBox = document.getElementById("alertAuditBox");
    const alerts = payload.recent_alerts || [];
    alertsBox.innerHTML = alerts.length
      ? alerts
          .slice(0, 6)
          .map(
            (a) =>
              `<div class="list-card"><strong>${a.alert_id}</strong> · ${a.headline}<br/><span class="muted-text">${a.dispatch_status}</span></div>`
          )
          .join("")
      : '<div class="list-card">No Medium/High alerts dispatched yet.</div>';
  } catch (err) {
    console.warn("Monitoring refresh error", err);
  }
}

function initFeedbackAndRetrain() {
  document.querySelectorAll(".btn-fb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!state.lastResult) return;
      const d = state.lastResult;
      const body = {
        run_id: d.run_id,
        location_name: d.collected_data.resolved_location,
        detected_disaster: d.classification.detected_disaster,
        predicted_risk: d.risk_assessment.risk_level,
        verdict: btn.dataset.verdict,
        notes: "Operator feedback via GeoAgent Command Center",
      };
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const out = await res.json();
      document.getElementById("feedbackResult").textContent =
        `Recorded '${body.verdict}' (delta ${out.applied_delta >= 0 ? "+" : ""}${out.applied_delta}). ` +
        `Next calibration offset for ${body.location_name}: ${out.updated_calibration_offset}.`;
      document.getElementById("calibrationBadge").textContent = `Offset: ${out.updated_calibration_offset}`;
      await refreshMonitoring();
    });
  });

  document.getElementById("retrainBtn").addEventListener("click", async () => {
    const btn = document.getElementById("retrainBtn");
    btn.disabled = true;
    btn.textContent = "Retraining...";
    try {
      await fetch("/api/models/retrain", { method: "POST" });
      await refreshMonitoring();
      await loadInitialMetadata();
    } finally {
      btn.disabled = false;
      btn.textContent = "Retrain Ensemble";
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initMap();
  initTabs();
  initFeedbackAndRetrain();
  document.getElementById("analyzeForm").addEventListener("submit", (e) => {
    e.preventDefault();
    document.getElementById("latInput").value = "";
    document.getElementById("lonInput").value = "";
    runAnalysis();
  });
  await loadInitialMetadata();
  await runAnalysis();
});
