import pandas as pd
from pathlib import Path
import joblib

from feature_engineering import create_features
from train_model import MODEL_PATH



def calculate_delay(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delay"] = out["actual_time"] - out["scheduled_time"]
    out["delay"] = out["delay"].apply(lambda x: max(x, 0))
    return out



def calculate_risk_score(row) -> int:
    score = 0

    if row["delay"] > 10:
        score += 30
    if row["delay"] > 20:
        score += 20

    if row["priority"] == "high":
        score += 25
    elif row["priority"] == "medium":
        score += 10

    if row["traffic_level"] == "heavy":
        score += 15
    elif row["traffic_level"] == "medium":
        score += 5

    if row["status"] != "delivered":
        score += 10

    return score



def assign_risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"



def recommend_action(row) -> str:
    if row["risk_level"] == "High":
        status = str(row.get("status", "")).lower()
        delay = row.get("delay", 0)

        is_picked_and_delayed = (
            status in {"picked_up", "in_transit", "delayed", "on_route"}
            and delay > 0
        )

        if is_picked_and_delayed:
            return "Expedite current driver and proactively notify customer"

        return "Reassign driver or notify operations immediately"

    elif row["risk_level"] == "Medium":
        return "Monitor closely and prepare contingency"

    else:
        return "No immediate action required"



def apply_risk_logic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = calculate_delay(out)
    out = create_features(out)

    model_file = Path(MODEL_PATH)

    try:
        if model_file.exists():
            bundle = joblib.load(model_file)
            model = bundle["model"]
            feature_columns = bundle["feature_columns"]

            X_live = out[feature_columns].copy()
            out["failure_probability"] = model.predict_proba(X_live)[:, 1]
        else:
            raise FileNotFoundError("Model not found")

    except Exception as e:
        print("⚠️ ML failed, using fallback:", str(e))
        out["risk_score"] = out.apply(calculate_risk_score, axis=1)
        out["failure_probability"] = (out["risk_score"] / 100).clip(upper=1.0)

    out["risk_score"] = (out["failure_probability"] * 100).round(1)
    out["risk_level"] = out["risk_score"].apply(assign_risk_level)
    out["recommended_action"] = out.apply(recommend_action, axis=1)

    return out
