from data_cleaning import load_data, inspect_data, clean_data
from risk_engine import apply_risk_logic
from llm_agent import generate_ai_brief
from phase1 import build_phase1_operational_view
from pre_assignment_risk import load_available_drivers, evaluate_pre_assignment_risk



def main():
    file_path = "data/orders_with_locations.csv"

    df = load_data(file_path)

    print("\n===== RAW DATA =====")
    inspect_data(df)

    df_clean = clean_data(df)

    print("\n===== CLEAN DATA =====")
    inspect_data(df_clean)

    # -------------------------
    # PRE-ASSIGNMENT RISK VIEW
    # -------------------------
    print("\n===== PRE-ASSIGNMENT RISK VIEW =====")

    drivers_df = load_available_drivers("data/driver_locations.csv")

    pre_assign_df = evaluate_pre_assignment_risk(
        orders_df=df_clean,
        drivers_df=drivers_df,
        nearby_radius_km=5.0, # tune this radius based on typical driver density and city size
    )

    print(
        pre_assign_df[
            [
                "job_id",
                "best_candidate_driver",
                "nearest_driver_distance_km",
                "nearest_driver_eta_min",
                "available_driver_count_nearby",
                "projected_total_time_min",
                "sla_buffer_min",
                "pre_assignment_risk_score",
                "pre_assignment_risk_level",
                "pre_assignment_alert_level",
                "pre_assignment_recommended_action",
            ]
        ].sort_values(by="pre_assignment_risk_score", ascending=False).head(10)
    )

    # -------------------------
    # POST-ASSIGNMENT RISK VIEW
    # -------------------------
    df_final = apply_risk_logic(df_clean)

    ops_df = build_phase1_operational_view(
        df_final,
        telemetry_path="data/driver_locations.csv",
        fallback_region="uk",
    )

    print("\n===== SUMMARY =====")
    print("Total jobs:", len(ops_df))
    print("High risk jobs:", (ops_df["risk_level"] == "High").sum())
    print("Medium risk jobs:", (ops_df["risk_level"] == "Medium").sum())
    print("Urgent alerts:", (ops_df["alert_level"] == "urgent").sum())

    print("\n===== FINAL OUTPUT =====")
    print(
        ops_df[
            [
                "job_id",
                "driver_id",
                "delay",
                "failure_probability",
                "risk_score",
                "risk_level",
                "alert_level",
                "recommended_action",
                "ops_action",
            ]
        ].sort_values(by="failure_probability", ascending=False)
    )

    top_jobs = ops_df.sort_values(by="failure_probability", ascending=False).head(5).copy()
    # top_jobs = ops_df.sort_values(by="risk_score", ascending=False).head(5).copy()
    top_jobs["explanation"] = top_jobs.apply(generate_ai_brief, axis=1)

    print("\n===== TOP 5 LLM EXPLANATIONS =====")
    for _, row in top_jobs.iterrows():
        print(f"\nJob ID: {row['job_id']}")
        print(row["explanation"])


if __name__ == "__main__":
    main()
