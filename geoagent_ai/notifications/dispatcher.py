"""Builds SMS, Email, and Push alert payloads and optionally dispatches via SMTP/Webhook."""

from __future__ import annotations

import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
import uuid

import requests

from geoagent_ai.config import get_settings
from geoagent_ai.schemas import AlertPayload, RiskLevel


def dispatch_alert_notifications(
    location_name: str,
    detected_disaster: str,
    risk_level: RiskLevel,
    severity_score: float,
    affected_area_km2: float,
    key_indicators: list[str],
) -> AlertPayload:
    """Create structured alerts when Risk is Medium or High; suppress otherwise."""
    now = datetime.now(timezone.utc).isoformat()
    alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"

    if risk_level == "Low" or detected_disaster == "No Risk":
        return AlertPayload(
            alert_triggered=False,
            alert_id=alert_id,
            severity_level=risk_level,
            headline=f"No Active Alert for {location_name}",
            alert_message="Conditions are within safe monitoring thresholds. Routine autonomous surveillance continues.",
            affected_area=f"{location_name} ({affected_area_km2:.1f} km² monitored)",
            channels=[],
            sms_preview="No alert dispatched (Risk Level: Low).",
            email_subject=f"[GeoAgent AI] Routine Status: {location_name} - Low Risk",
            email_body="No emergency dispatch required.",
            push_notification="Monitoring active: conditions normal.",
            dispatch_status="SUPPRESSED_LOW_RISK",
            timestamp_utc=now,
        )

    indicator_str = "; ".join(key_indicators[:2]) if key_indicators else "Multi-sensor threshold exceeded"
    headline = f"{risk_level.upper()} RISK {detected_disaster.upper()} ALERT — {location_name}"
    alert_message = (
        f"GeoAgent AI detected {detected_disaster} risk ({risk_level}, severity {severity_score:.1f}/100) "
        f"affecting ~{affected_area_km2:.1f} km² around {location_name}. Key signals: {indicator_str}."
    )
    sms_preview = (
        f"[GEOAGENT {risk_level.upper()}] {detected_disaster} alert in {location_name}. "
        f"Severity {severity_score:.0f}/100. Follow evacuation advisories. Emergency: 112."
    )
    email_subject = f"[URGENT - {risk_level.upper()}] GeoAgent AI {detected_disaster} Warning: {location_name}"
    email_body = (
        f"Automated Advisory from GeoAgent AI\n"
        f"Alert ID: {alert_id}\n"
        f"Location: {location_name}\n"
        f"Detected Hazard: {detected_disaster}\n"
        f"Risk Level: {risk_level} ({severity_score:.1f}/100)\n"
        f"Affected Area: {affected_area_km2:.1f} km²\n"
        f"Key Indicators: {indicator_str}\n"
        f"Action: Review evacuation routes and nearest relief shelters on the GeoAgent Dashboard."
    )
    push_notification = f"{headline} — Tap to view shelters & evacuation route."

    settings = get_settings()
    channels = ["Dashboard Feed", "SMS Gateway (Simulated)", "Email Advisory (Simulated)", "Mobile Push (Simulated)"]
    dispatch_results = ["Logged to Audit Outbox"]

    if settings.alert_webhook_url:
        try:
            requests.post(
                settings.alert_webhook_url,
                json={"alert_id": alert_id, "headline": headline, "message": alert_message},
                timeout=3.0,
            )
            channels.append("Webhook Live")
            dispatch_results.append("Webhook Dispatched")
        except Exception:
            dispatch_results.append("Webhook Failed (Fallback to Outbox)")

    if settings.smtp_host and settings.smtp_user and settings.smtp_password:
        try:
            msg = EmailMessage()
            msg["Subject"] = email_subject
            msg["From"] = settings.smtp_user
            msg["To"] = settings.smtp_user
            msg.set_content(email_body)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=4.0) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)
            channels.append("SMTP Live")
            dispatch_results.append("SMTP Email Sent")
        except Exception:
            dispatch_results.append("SMTP Failed (Fallback to Outbox)")

    return AlertPayload(
        alert_triggered=True,
        alert_id=alert_id,
        severity_level=risk_level,
        headline=headline,
        alert_message=alert_message,
        affected_area=f"{location_name} ({affected_area_km2:.1f} km²)",
        channels=channels,
        sms_preview=sms_preview,
        email_subject=email_subject,
        email_body=email_body,
        push_notification=push_notification,
        dispatch_status=" | ".join(dispatch_results),
        timestamp_utc=now,
    )
