import math
from pathlib import Path
from typing import Optional

import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two latitude/longitude points in kilometers."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None

    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c



def normalize_driver_columns(drivers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize driver telemetry columns.
    Supports:
    - driver_id
    - lat / lon
    - timestamp
    - status
    """
    out = drivers_df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    rename_map = {
        "lat": "driver_lat",
        "latitude": "driver_lat",
        "lon": "driver_lon",
        "lng": "driver_lon",
        "longitude": "driver_lon",
        "timestamp": "last_telemetry_utc",
    }

    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = {"driver_id", "driver_lat", "driver_lon"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Driver file missing required columns: {sorted(missing)}")

    out["driver_id"] = out["driver_id"].astype(str).str.strip()
    out["driver_lat"] = pd.to_numeric(out["driver_lat"], errors="coerce")
    out["driver_lon"] = pd.to_numeric(out["driver_lon"], errors="coerce")

    if "last_telemetry_utc" not in out.columns:
        out["last_telemetry_utc"] = None

    if "status" in out.columns:
        out["status"] = out["status"].astype(str).str.strip().str.lower()

    out = out.dropna(subset=["driver_lat", "driver_lon"])
    return out



def load_available_drivers(driver_file_path: str) -> pd.DataFrame:
    path = Path(driver_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Driver file not found: {driver_file_path}")

    drivers_df = pd.read_csv(path)
    drivers_df = normalize_driver_columns(drivers_df)

    if "status" in drivers_df.columns:
        available_df = drivers_df[drivers_df["status"] == "available"].copy()

        if not available_df.empty:
            return available_df

        print("⚠️ No available drivers found, using all drivers")
        return drivers_df.copy()

    print("⚠️ No 'status' column found, assuming all drivers are available")
    return drivers_df.copy()



def estimate_speed_kmph(traffic_level: str) -> float:
    traffic = str(traffic_level).strip().lower()

    if traffic == "heavy":
        return 20.0
    elif traffic == "medium":
        return 30.0
    else:
        return 40.0



def km_to_minutes(distance_km: Optional[float], speed_kmph: float) -> Optional[float]:
    if distance_km is None or pd.isna(distance_km):
        return None
    if speed_kmph <= 0:
        return None
    return (distance_km / speed_kmph) * 60.0



def assign_pre_assignment_risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"



def assign_pre_assignment_alert(score: float) -> str:
    if score >= 70:
        return "urgent"
    elif score >= 40:
        return "high"
    return "normal"



def recommend_pre_assignment_action(row) -> str:
    risk_level = row.get("pre_assignment_risk_level", "Low")
    sla_buffer = row.get("sla_buffer_min")
    nearby_count = row.get("available_driver_count_nearby", 0)

    if risk_level == "High":
        if nearby_count == 0:
            return "No nearby driver available. Escalate immediately or reject order."
        if sla_buffer is not None and sla_buffer < 0:
            return "Projected SLA breach before assignment. Rebalance fleet or escalate."
        return "Assign closest available driver immediately and notify ops."

    elif risk_level == "Medium":
        return "Assign nearest available driver and monitor closely."

    return "Safe to assign normally"



def evaluate_pre_assignment_risk(
    orders_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    nearby_radius_km: float = 5.0,
) -> pd.DataFrame:
    """
    Evaluate risk before assigning a driver.
    """
    orders = orders_df.copy()
    drivers = drivers_df.copy()

    for col in ["priority", "traffic_level"]:
        if col in orders.columns:
            orders[col] = orders[col].astype(str).str.strip().str.lower()

    results = []

    for _, order in orders.iterrows():
        pickup_lat = order.get("pickup_lat")
        pickup_lon = order.get("pickup_lon")
        drop_lat = order.get("drop_lat")
        drop_lon = order.get("drop_lon")
        traffic_level = order.get("traffic_level", "low")
        scheduled_time = pd.to_numeric(order.get("scheduled_time", 0), errors="coerce")
        priority = str(order.get("priority", "low")).lower()

        speed_kmph = estimate_speed_kmph(traffic_level)

        candidates = drivers.copy()
        candidates["driver_to_pickup_km"] = candidates.apply(
            lambda d: haversine_km(
                d.get("driver_lat"),
                d.get("driver_lon"),
                pickup_lat,
                pickup_lon,
            ),
            axis=1,
        )

        candidates["pickup_eta_min"] = candidates["driver_to_pickup_km"].apply(
            lambda km: km_to_minutes(km, speed_kmph)
        )

        candidates = candidates.dropna(subset=["driver_to_pickup_km", "pickup_eta_min"]).copy()
        candidates = candidates.sort_values(by="pickup_eta_min", ascending=True)

        nearest_driver_id = None
        nearest_driver_distance_km = None
        nearest_driver_eta_min = None
        available_driver_count_nearby = 0
        avg_top3_driver_eta_min = None
        projected_trip_km = None
        projected_trip_time_min = None
        projected_total_time_min = None
        sla_buffer_min = None

        if not candidates.empty:
            nearest = candidates.iloc[0]
            nearest_driver_id = nearest["driver_id"]
            nearest_driver_distance_km = round(nearest["driver_to_pickup_km"], 2)
            nearest_driver_eta_min = round(nearest["pickup_eta_min"], 1)

            available_driver_count_nearby = int(
                (candidates["driver_to_pickup_km"] <= nearby_radius_km).sum()
            )

            avg_top3_driver_eta_min = round(candidates.head(3)["pickup_eta_min"].mean(), 1)

            projected_trip_km = haversine_km(
                pickup_lat,
                pickup_lon,
                drop_lat,
                drop_lon,
            )

            if projected_trip_km is not None:
                projected_trip_time_min = round(km_to_minutes(projected_trip_km, speed_kmph), 1)
                projected_total_time_min = round(nearest_driver_eta_min + projected_trip_time_min, 1)

                if pd.notna(scheduled_time):
                    sla_buffer_min = round(scheduled_time - projected_total_time_min, 1)

        score = 0

        if candidates.empty:
            score = 100
        else:
            if nearest_driver_eta_min is not None:
                if nearest_driver_eta_min > 20:
                    score += 35
                elif nearest_driver_eta_min > 10:
                    score += 20
                elif nearest_driver_eta_min > 5:
                    score += 10

            if available_driver_count_nearby == 0:
                score += 30
            elif available_driver_count_nearby == 1:
                score += 15
            elif available_driver_count_nearby <= 3:
                score += 5

            if sla_buffer_min is not None:
                if sla_buffer_min < 0:
                    score += 40
                elif sla_buffer_min <= 5:
                    score += 25
                elif sla_buffer_min <= 10:
                    score += 10

            if priority == "high":
                score += 20
            elif priority == "medium":
                score += 10

            if traffic_level == "heavy":
                score += 15
            elif traffic_level == "medium":
                score += 5

        score = min(score, 100)
        risk_level = assign_pre_assignment_risk_level(score)
        alert_level = assign_pre_assignment_alert(score)

        result_row = order.to_dict()
        result_row.update(
            {
                "best_candidate_driver": nearest_driver_id,
                "nearest_driver_distance_km": nearest_driver_distance_km,
                "nearest_driver_eta_min": nearest_driver_eta_min,
                "available_driver_count_nearby": available_driver_count_nearby,
                "avg_top3_driver_eta_min": avg_top3_driver_eta_min,
                "projected_trip_km": round(projected_trip_km, 2) if projected_trip_km is not None else None,
                "projected_trip_time_min": projected_trip_time_min,
                "projected_total_time_min": projected_total_time_min,
                "sla_buffer_min": sla_buffer_min,
                "pre_assignment_risk_score": score,
                "pre_assignment_risk_level": risk_level,
                "pre_assignment_alert_level": alert_level,
            }
        )

        results.append(result_row)

    out = pd.DataFrame(results)
    out["pre_assignment_recommended_action"] = out.apply(recommend_pre_assignment_action, axis=1)
    return out
