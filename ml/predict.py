
import argparse
import json
import joblib
import pandas as pd

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "air_quality_index",
    "sleep_hours",
    "sleep_quality_1_5",
    "mood_readiness_1_5",
    "planned_duration_min",
    "planned_intensity_1_10",
    "workout_type",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True, help="JSON string with one pre-session record")
    args = parser.parse_args()

    model = joblib.load(args.model)
    payload = json.loads(args.json)

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    df = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)
    pred = model.predict(df)[0]
    print(float(pred) if hasattr(pred, "__float__") else pred)

if __name__ == "__main__":
    main()
