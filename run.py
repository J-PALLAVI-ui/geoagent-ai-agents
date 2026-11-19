"""Unified CLI launcher for GeoAgent AI (train, analyze, web server, or Streamlit)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from geoagent_ai.agents import GeoAgentOrchestrator
from geoagent_ai.ml import train_and_save_models
from geoagent_ai.schemas import LocationQuery


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoAgent AI Command Launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train and evaluate ML hazard & severity models")

    analyze_p = sub.add_parser("analyze", help="Run autonomous multi-agent workflow via CLI")
    analyze_p.add_argument("--location", required=True, help="Location/region name")
    analyze_p.add_argument("--mode", choices=["live", "simulation"], default="simulation")
    analyze_p.add_argument("--scenario", default=None, help="Optional scenario preset")

    srv_p = sub.add_parser("server", help="Start FastAPI server + Geospatial Command Center")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8000)

    sub.add_parser("streamlit", help="Start Streamlit + Folium dashboard")

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_and_save_models()
        print(json.dumps(metrics, indent=2))
    elif args.command == "analyze":
        orchestrator = GeoAgentOrchestrator()
        res = orchestrator.run(
            LocationQuery(location_name=args.location, mode=args.mode, scenario=args.scenario)
        )
        print(res.model_dump_json(indent=2))
    elif args.command == "server":
        import uvicorn

        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)
    elif args.command == "streamlit":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_app.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
