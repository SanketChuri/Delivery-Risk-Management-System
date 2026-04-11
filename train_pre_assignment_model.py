import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRE_ASSIGNMENT_MODEL_PATH = "pre_assignment_model.pkl"



def build_training_pipeline():
    numeric_features = [
        "scheduled_time",
        "pickup_lat",
        "pickup_lon",
        "drop_lat",
        "drop_lon",
        "nearest_driver_eta_min",
        "available_driver_count_nearby",
        "projected_total_time_min",
        "sla_buffer_min",
    ]

    categorical_features = [
        "priority",
        "traffic_level",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline, numeric_features + categorical_features



def main():
    file_path = "data/pre_assignment_training.csv"

    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    print("\n===== TARGET CHECK =====")
    print(df["will_fail"].value_counts(dropna=False))

    if df["will_fail"].nunique() < 2:
        raise ValueError(
            "Training data has only one target class. "
            "Need both fail (1) and non-fail (0) examples."
        )

    pipeline, feature_columns = build_training_pipeline()

    X = df[feature_columns]
    y = df["will_fail"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("\n===== PRE-ASSIGNMENT MODEL EVALUATION =====")
    print(classification_report(y_test, preds))
    print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))

    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": feature_columns,
        },
        PRE_ASSIGNMENT_MODEL_PATH,
    )

    print(f"\nModel saved to {PRE_ASSIGNMENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
