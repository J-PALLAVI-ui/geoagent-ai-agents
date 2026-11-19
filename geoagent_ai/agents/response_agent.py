"""Agent 6: Response Planning Agent — shelters, evacuation routes, safety actions & resources."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.geo import compute_shelters_and_routes
from geoagent_ai.schemas import EmergencyContact, ResponsePlan

DISASTER_PLAYBOOKS: dict[str, dict[str, list[str]]] = {
    "Flood": {
        "actions": [
            "Initiate phased evacuation of low-lying riverbanks and basement dwellings.",
            "Deploy inflatable rescue boats and high-clearance vehicles to staging hubs.",
            "Open floodgates/storm drains and isolate submerged electrical feeders.",
        ],
        "safety": [
            "Avoid walking or driving through moving floodwater deeper than 15 cm.",
            "Move essential documents, medicines, and dry rations to elevated floors.",
            "Boil or chlorinate drinking water until municipal supply is certified safe.",
        ],
        "resources": [
            "NDRF Swift-Water Rescue Teams & Inflatable Boats",
            "Portable Dewatering Pumps & Mobile Medical Clinics",
            "Emergency Food/Water Packets & Solar Lantern Kits",
        ],
    },
    "Fire": {
        "actions": [
            "Establish firebreak containment lines downwind of active NASA FIRMS hotspots.",
            "Evacuate settlements and wildlife-fringe villages within 5 km burn corridor.",
            "Coordinate aerial water-bombing / forest fire tenders along access tracks.",
        ],
        "safety": [
            "Keep N95 masks or wet cloths ready to filter dense smoke particulate.",
            "Clear dry brush and flammable fuel tanks around residential perimeters.",
            "Follow designated smoke-free evacuation corridors away from wind direction.",
        ],
        "resources": [
            "Forest Fire Rapid Response Tenders & Foam Units",
            "Thermal Drone Reconnaissance & Respiratory Trauma Kits",
            "Water Bowsers & Bulldozers for Fireline Trenching",
        ],
    },
    "Cyclone": {
        "actions": [
            "Suspend all coastal fishing/maritime operations and secure port vessels.",
            "Relocate coastal residents to reinforced multipurpose cyclone shelters before landfall.",
            "Pre-position power restoration crews, telecom-on-wheels, and tree-clearing teams.",
        ],
        "safety": [
            "Stay indoors away from glass windows during peak eye-wall wind gusts.",
            "Do not venture outside during temporary calm of the cyclone eye.",
            "Unplug appliances and keep battery-operated radio/power banks charged.",
        ],
        "resources": [
            "Multipurpose Cyclone Shelters & Backup Diesel Generators",
            "Chainsaws, Earthmovers & Emergency Telecom Towers",
            "NDMA/SDRF Urban Search & Coastal Surge Rescue Teams",
        ],
    },
    "Landslide": {
        "actions": [
            "Close vulnerable hill highways and reroute traffic away from saturated slopes.",
            "Evacuate hillside settlements showing tension cracks, muddy springs, or tilting trees.",
            "Deploy geotechnical sensors and earth-moving excavators for debris clearance.",
        ],
        "safety": [
            "Move immediately to stable ridge or valley-center assembly zones.",
            "Listen for rumbling debris flows and stay alert during continuous rainfall.",
            "Do not re-enter destabilized structures until geotechnical clearance is issued.",
        ],
        "resources": [
            "SDRF Mountain Rescue & K9 Canine Search Units",
            "Heavy Excavators, JCB Loaders & Portable Bailey Bridge Kits",
            "Satellite Phones & Slope Stability Radar Monitors",
        ],
    },
    "Earthquake": {
        "actions": [
            "Inspect bridges, hospitals, dams, and high-rise structures for aftershock safety.",
            "Establish open-ground triage and assembly shelters away from tall buildings.",
            "Shut off compromised gas mains and activate urban search-and-rescue units.",
        ],
        "safety": [
            "Drop, Cover, and Hold On during shaking; evacuate via stairs once tremors stop.",
            "Stay clear of damaged walls, power lines, and glass facades due to aftershocks.",
            "Use SMS or emergency apps rather than voice calls to keep networks uncongested.",
        ],
        "resources": [
            "USAR Acoustic Listening Devices & Concrete Cutters",
            "Mobile Field Hospitals, Blood Bank Units & Tents",
            "Structural Engineering Rapid Assessment Teams",
        ],
    },
    "No Risk": {
        "actions": [
            "Maintain automated multi-source telemetry polling across regional sensors.",
            "Keep municipal drainage, shelter readiness, and emergency rosters on standby.",
        ],
        "safety": [
            "No protective evacuation required; stay informed via official advisories.",
            "Review household emergency kit and local shelter locations periodically.",
        ],
        "resources": [
            "Routine Sensor Network Diagnostics",
            "Regional Standby Emergency Control Room",
        ],
    },
}

DEFAULT_CONTACTS = [
    {"agency": "National Emergency Number", "number": "112", "role": "Unified Police/Fire/Ambulance"},
    {"agency": "NDMA Disaster Helpline", "number": "1078 / 011-26701728", "role": "National Disaster Management"},
    {"agency": "Fire & Rescue Services", "number": "101", "role": "Rapid Fire & Extraction"},
]


def response_planning_node(state: GeoAgentState) -> dict:
    """Autonomously recommend evacuation routes, shelters, safety steps, and resources."""

    def _execute():
        bundle = state["collected_data"]
        classification = state["classification"]
        geo = state["geo_analysis"]
        risk = state["risk_assessment"]
        catalog_entry = state.get("catalog_entry")
        assert bundle and classification and geo and risk

        shelters, routes = compute_shelters_and_routes(
            center_lat=geo.center_lat,
            center_lon=geo.center_lon,
            impact_radius_km=geo.impact_radius_km,
            detected_disaster=classification.detected_disaster,
            catalog_entry=catalog_entry,
        )

        playbook = DISASTER_PLAYBOOKS.get(
            classification.detected_disaster, DISASTER_PLAYBOOKS["No Risk"]
        )
        raw_contacts = (catalog_entry or {}).get("contacts") or DEFAULT_CONTACTS
        contacts = [EmergencyContact(**c) for c in raw_contacts]

        priority = (
            "IMMEDIATE DISPATCH (CODE RED)"
            if risk.risk_level == "High"
            else ("ELEVATED STANDBY (CODE AMBER)" if risk.risk_level == "Medium" else "ROUTINE MONITORING (CODE GREEN)")
        )

        plan = ResponsePlan(
            priority_level=priority,
            recommended_actions=playbook["actions"],
            safety_measures=playbook["safety"],
            resource_allocation=playbook["resources"],
            nearest_shelters=shelters,
            evacuation_routes=routes,
            emergency_contacts=contacts,
        )
        msg = AIMessage(
            content=(
                f"[ResponseAgent] {priority}. Mapped {len(shelters)} nearest shelters and "
                f"{len(routes)} evacuation routes (top shelter: {shelters[0].name} at {shelters[0].distance_km} km)."
            )
        )
        return (
            {"response_plan": plan, "messages": [msg]},
            "completed",
            f"{priority} — {len(shelters)} shelters & {len(routes)} routes computed",
            {
                "priority_level": priority,
                "top_shelter": shelters[0].name if shelters else None,
                "routes_count": len(routes),
            },
        )

    return run_timed_step("agent-6-response-planning", "Response Planning Agent", _execute)
