import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "backend/model.pkl"))


def require_env() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in backend/.env")


def fetch_flights() -> list[dict]:
    require_env()
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return (
        sb.table("flights")
        .select(
            "flight_id, airline_id, route_id, flight_status, delay_minutes, scheduled_departure, "
            "available_seats, price_economy, price_business, route:routes(distance_km)"
        )
        .execute()
        .data
        or []
    )


def build_dataset(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        dep = pd.to_datetime(row.get("scheduled_departure"), errors="coerce")
        delay_minutes = row.get("delay_minutes") or 0
        status = row.get("flight_status")
        route = row.get("route") or {}
        records.append(
            {
                "airline_id": row.get("airline_id") or 0,
                "route_id": row.get("route_id") or 0,
                "departure_hour": 0 if pd.isna(dep) else dep.hour,
                "departure_day": 0 if pd.isna(dep) else dep.dayofweek,
                "distance_km": route.get("distance_km") or 0,
                "available_seats": row.get("available_seats") or 0,
                "price_economy": row.get("price_economy") or 0,
                "price_business": row.get("price_business") or 0,
                "is_delayed": int(delay_minutes >= 15 or status in {"delayed", "cancelled"}),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    rows = fetch_flights()
    df = build_dataset(rows)
    if df.empty:
        raise RuntimeError("No flight rows found in Supabase.")
    if df["is_delayed"].nunique() < 2:
        raise RuntimeError("Need both delayed and non-delayed examples to train a classifier.")

    features = [
        "airline_id",
        "route_id",
        "departure_hour",
        "departure_day",
        "distance_km",
        "available_seats",
        "price_economy",
        "price_business",
    ]
    class_counts = df["is_delayed"].value_counts()
    can_stratify = len(class_counts) == 2 and class_counts.min() >= 2
    if not can_stratify:
        print(
            "Warning: one delay class has fewer than 2 examples, so training without stratified split. "
            "Add more delayed and on-time historical flights for a more reliable model."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        df[features],
        df["is_delayed"],
        test_size=0.25,
        random_state=42,
        stratify=df["is_delayed"] if can_stratify else None,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
