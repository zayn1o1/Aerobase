import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "backend/model.pkl"))

app = FastAPI(title="Airport AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in backend/.env",
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    a_start = parse_time(start_a)
    a_end = parse_time(end_a)
    b_start = parse_time(start_b)
    b_end = parse_time(end_b)
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return a_start < b_end and a_end > b_start


def gate_conflicts(
    gate_id: int,
    flight: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> bool:
    return any(
        assignment.get("gate_id") == gate_id
        and overlaps(
            assignment.get("start_time"),
            assignment.get("end_time"),
            flight.get("scheduled_departure"),
            flight.get("scheduled_arrival"),
        )
        for assignment in assignments
    )


def gate_cost(gate: dict[str, Any], flight: dict[str, Any]) -> float:
    terminal = gate.get("terminal") or {}
    airport = (terminal.get("airport") or {}).get("iata_code")
    origin = ((flight.get("route") or {}).get("origin") or {}).get("iata_code")
    cost = 10.0
    if gate.get("has_jetbridge"):
        cost -= 3.0
    if origin and airport == origin:
        cost -= 3.0
    elif origin and airport != origin:
        cost += 8.0
    return max(1.0, cost)


def a_star_gate_plan(
    flights: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    existing_assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Best-first greedy gate assignment: for each flight pick the lowest-cost conflict-free gate."""
    result: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []

    for flight in flights:
        all_assignments = existing_assignments + planned
        available = [
            gate for gate in gates
            if not gate_conflicts(gate["gate_id"], flight, all_assignments)
        ]

        if available:
            best = min(available, key=lambda g: gate_cost(g, flight))
            assignment = {
                "flight_id": flight["flight_id"],
                "gate": best,
                "gate_id": best["gate_id"],
                "cost": gate_cost(best, flight),
                "start_time": flight.get("scheduled_departure"),
                "end_time": flight.get("scheduled_arrival"),
            }
        else:
            assignment = {
                "flight_id": flight["flight_id"],
                "gate": None,
                "cost": 80.0,
                "start_time": flight.get("scheduled_departure"),
                "end_time": flight.get("scheduled_arrival"),
            }

        result.append(assignment)
        planned.append(assignment)

    return result


def load_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def flight_features(flight: dict[str, Any]) -> list[float]:
    dep = parse_time(flight.get("scheduled_departure"))
    route = flight.get("route") or {}
    return [
        float(flight.get("airline_id") or 0),
        float(flight.get("route_id") or 0),
        float(dep.hour if dep else 0),
        float(dep.weekday() if dep else 0),
        float(route.get("distance_km") or 0),
        float(flight.get("available_seats") or 0),
        float(flight.get("price_economy") or 0),
        float(flight.get("price_business") or 0),
    ]


def fallback_delay_probability(flight: dict[str, Any]) -> float:
    status = flight.get("flight_status")
    delay_minutes = flight.get("delay_minutes") or 0
    if status == "cancelled":
        return 1.0
    if status == "delayed":
        return min(0.95, 0.45 + delay_minutes / 240)
    if delay_minutes:
        return min(0.8, delay_minutes / 180)
    return 0.2


def predict_delay_probability(flight: dict[str, Any]) -> float:
    model_bundle = load_model()
    if not model_bundle:
        return fallback_delay_probability(flight)
    model = model_bundle["model"]
    probability = model.predict_proba([flight_features(flight)])[0][1]
    return round(float(probability), 4)


def insert_recommendations(sb: Client, recs: list[dict[str, Any]]) -> int:
    if not recs:
        return 0
    result = sb.table("recommendations").insert(recs).execute()
    return len(result.data or recs)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_loaded": MODEL_PATH.exists()}


@app.post("/recommendations/gates")
def recommend_gates() -> dict[str, Any]:
    sb = get_supabase()
    flights = (
        sb.table("flights")
        .select(
            "flight_id, flight_number, scheduled_departure, scheduled_arrival, "
            "flight_status, route:routes(origin:airports!routes_origin_id_fkey(iata_code))"
        )
        .in_("flight_status", ["scheduled", "boarding"])
        .gt("scheduled_departure", now_iso())
        .order("scheduled_departure")
        .limit(20)
        .execute()
        .data
        or []
    )
    assignments = (
        sb.table("gate_assignments")
        .select("flight_id, gate_id, start_time, end_time")
        .execute()
        .data
        or []
    )
    gates = (
        sb.table("gates")
        .select(
            "gate_id, gate_code, gate_type, has_jetbridge, max_aircraft_size, "
            "terminal:terminals(terminal_code, airport:airports(iata_code))"
        )
        .eq("is_operational", True)
        .execute()
        .data
        or []
    )
    assigned_flight_ids = {a.get("flight_id") for a in assignments}
    unassigned = [flight for flight in flights if flight.get("flight_id") not in assigned_flight_ids]
    planned_assignments = a_star_gate_plan(unassigned, gates, assignments)
    flights_by_id = {flight["flight_id"]: flight for flight in unassigned}

    recs: list[dict[str, Any]] = []
    for planned in planned_assignments:
        flight = flights_by_id[planned["flight_id"]]
        dep = parse_time(flight.get("scheduled_departure"))
        dep_label = dep.strftime("%Y-%m-%d %H:%M") if dep else "scheduled time"
        best = planned.get("gate")
        if best:
            terminal = (best.get("terminal") or {}).get("terminal_code", "?")
            airport = ((best.get("terminal") or {}).get("airport") or {}).get("iata_code", "?")
            score = round(max(40.0, 100.0 - planned["cost"]), 1)
            recs.append(
                {
                    "rec_type": "gate_assignment",
                    "generated_for": "flight",
                    "entity_id": flight["flight_id"],
                    "recommendation_text": (
                        f"A* selected Gate {best['gate_code']} for flight {flight['flight_number']} "
                        f"departing {dep_label}. "
                        f"Recommend Gate {best['gate_code']} at Terminal {terminal} ({airport}); "
                        "it is free during the flight window"
                        + (", and has a jetbridge." if best.get("has_jetbridge") else ".")
                    ),
                    "score": score,
                }
            )
        else:
            recs.append(
                {
                    "rec_type": "gate_assignment",
                    "generated_for": "flight",
                    "entity_id": flight["flight_id"],
                    "recommendation_text": (
                        f"A* could not find a free gate for flight {flight['flight_number']} departing {dep_label} "
                        "during its scheduled window. Review timing or add capacity."
                    ),
                    "score": 20,
                }
            )

    inserted = insert_recommendations(sb, recs)
    return {"inserted": inserted, "recommendations": recs}


@app.post("/recommendations/alternative-flights")
def recommend_alternative_flights() -> dict[str, Any]:
    sb = get_supabase()
    flights = (
        sb.table("flights")
        .select(
            "flight_id, flight_number, airline_id, route_id, flight_status, delay_minutes, "
            "scheduled_departure, scheduled_arrival, available_seats, price_economy, price_business, "
            "route:routes(distance_km)"
        )
        .in_("flight_status", ["scheduled", "boarding", "delayed", "cancelled"])
        .gt("scheduled_departure", now_iso())
        .order("scheduled_departure")
        .limit(80)
        .execute()
        .data
        or []
    )
    recs: list[dict[str, Any]] = []

    for flight in flights:
        delay_probability = predict_delay_probability(flight)
        is_problem = flight.get("flight_status") in ["delayed", "cancelled"] or delay_probability >= 0.5
        if not is_problem:
            continue

        alternatives = [
            alt
            for alt in flights
            if alt.get("route_id") == flight.get("route_id")
            and alt.get("flight_id") != flight.get("flight_id")
            and alt.get("flight_status") in ["scheduled", "boarding"]
            and (alt.get("available_seats") or 0) > 0
        ]
        if not alternatives:
            continue

        base_price = float(flight.get("price_economy") or flight.get("price_business") or 0)

        def score_alt(alt: dict[str, Any]) -> float:
            alt_delay = predict_delay_probability(alt)
            reliability = 1 - alt_delay
            alt_price = float(alt.get("price_economy") or alt.get("price_business") or base_price or 1)
            price_score = 1.0 if not base_price else max(0.0, min(1.0, base_price / alt_price))
            availability = min(1.0, float(alt.get("available_seats") or 0) / 100)
            return round((reliability * 0.5 + price_score * 0.3 + availability * 0.2) * 100, 1)

        best = sorted(alternatives, key=score_alt, reverse=True)[0]
        best_score = score_alt(best)
        dep = parse_time(best.get("scheduled_departure"))
        dep_label = dep.strftime("%Y-%m-%d %H:%M") if dep else "upcoming departure"
        recs.append(
            {
                "rec_type": "alternative_flight",
                "generated_for": "flight",
                "entity_id": flight["flight_id"],
                "recommendation_text": (
                    f"Flight {flight['flight_number']} has delay risk {round(delay_probability * 100)}%. "
                    f"Recommend alternative {best['flight_number']} departing {dep_label}: "
                    f"{best.get('available_seats') or 0} seats available, economy fare "
                    f"${best.get('price_economy') or 'N/A'}, ranked by delay risk, price, and availability."
                ),
                "score": best_score,
            }
        )

    inserted = insert_recommendations(sb, recs)
    return {"inserted": inserted, "recommendations": recs}


@app.post("/recommendations/run-all")
def run_all_recommendations() -> dict[str, Any]:
    gate_result = recommend_gates()
    alt_result = recommend_alternative_flights()
    return {
        "inserted": gate_result["inserted"] + alt_result["inserted"],
        "gate_recommendations": gate_result["inserted"],
        "alternative_flight_recommendations": alt_result["inserted"],
    }


@app.get("/recommend-gate/{flight_id}")
def recommend_gate_for_flight(flight_id: int) -> dict[str, Any]:
    sb = get_supabase()
    flight = (
        sb.table("flights")
        .select(
            "flight_id, flight_number, scheduled_departure, scheduled_arrival, "
            "flight_status, route:routes(origin:airports!routes_origin_id_fkey(iata_code))"
        )
        .eq("flight_id", flight_id)
        .single()
        .execute()
        .data
    )
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")

    assignments = (
        sb.table("gate_assignments")
        .select("flight_id, gate_id, start_time, end_time")
        .execute()
        .data
        or []
    )
    gates = (
        sb.table("gates")
        .select(
            "gate_id, gate_code, gate_type, has_jetbridge, max_aircraft_size, "
            "terminal:terminals(terminal_code, airport:airports(iata_code))"
        )
        .eq("is_operational", True)
        .execute()
        .data
        or []
    )

    planned = a_star_gate_plan([flight], gates, assignments)
    result = planned[0] if planned else None
    best = result.get("gate") if result else None

    if best:
        terminal = (best.get("terminal") or {}).get("terminal_code", "?")
        airport = ((best.get("terminal") or {}).get("airport") or {}).get("iata_code", "?")
        dep = parse_time(flight.get("scheduled_departure"))
        dep_label = dep.strftime("%Y-%m-%d %H:%M") if dep else "scheduled time"
        score = round(max(40.0, 100.0 - result["cost"]), 1)
        rec_text = (
            f"Recommend Gate {best['gate_code']} at Terminal {terminal} ({airport}) "
            f"for flight {flight['flight_number']} departing {dep_label}; "
            f"gate is free during the flight window"
            + (", and has a jetbridge." if best.get("has_jetbridge") else ".")
        )
        insert_recommendations(sb, [{
            "rec_type": "gate_assignment",
            "generated_for": "flight",
            "entity_id": flight_id,
            "recommendation_text": rec_text,
            "score": score,
        }])
        return {
            "flight_id": flight_id,
            "flight_number": flight.get("flight_number"),
            "gate_code": best["gate_code"],
            "terminal": terminal,
            "airport": airport,
            "has_jetbridge": best.get("has_jetbridge", False),
            "gate_type": best.get("gate_type"),
            "score": score,
            "assigned": True,
        }

    return {
        "flight_id": flight_id,
        "flight_number": flight.get("flight_number"),
        "assigned": False,
        "gate_code": None,
        "terminal": None,
        "airport": None,
    }


@app.get("/predict-delay/{flight_id}")
def predict_delay(flight_id: int) -> dict[str, Any]:
    sb = get_supabase()
    flight = (
        sb.table("flights")
        .select(
            "flight_id, flight_number, airline_id, route_id, flight_status, delay_minutes, "
            "scheduled_departure, scheduled_arrival, available_seats, price_economy, price_business, "
            "route:routes(distance_km)"
        )
        .eq("flight_id", flight_id)
        .single()
        .execute()
        .data
    )
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")
    probability = predict_delay_probability(flight)
    return {
        "flight_id": flight_id,
        "flight_number": flight.get("flight_number"),
        "delay_probability": probability,
        "risk_level": "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low",
        "model_loaded": MODEL_PATH.exists(),
    }
