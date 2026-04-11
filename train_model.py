import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import load_data, clean_data
from feature_engineering import create_features, create_target


MODEL_PATH = "risk_model.pkl"



def build_training_pipeline():
    numeric_features = [
        "scheduled_time",
        "actual_time",
        "delay",
        "pickup_lat",
        "pickup_lon",
        "drop_lat",
        "drop_lon",
        "distance_km",
        "traffic_severity",
        "priority_score",
        "is_late_start",
    ]

    categorical_features = [
        "priority",
        "traffic_level",
        "status",
    ]

    numeric_transformer = Pipeline(
        # Steps to impute missing values and scale numeric features.
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        # Steps to impute missing values and one-hot encode categorical features.
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        # Define how to preprocess numeric and categorical features.
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        # A robust ensemble model that can handle nonlinearities and interactions without much tuning.
        n_estimators=200, # More trees for better performance, especially on a small dataset.
        max_depth=8, # Limit depth to prevent overfitting on small dataset.
        min_samples_split=5, # Increased to prevent overfitting on small dataset.
        min_samples_leaf=2, 
        random_state=42, # For reproducibility.
    )

    pipeline = Pipeline(
        # Final pipeline that combines preprocessing and modeling steps.
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline, numeric_features + categorical_features



def main():
    file_path = "data/post_assignment_training.csv"

    df = load_data(file_path)
    df = clean_data(df)
    df = create_features(df)
    df = create_target(df, fail_delay_threshold=15)

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

    print("\n===== MODEL EVALUATION =====")
    print(classification_report(y_test, preds))

    classifier = pipeline.named_steps["classifier"]
    if len(classifier.classes_) > 1:
        probs = pipeline.predict_proba(X_test)[:, 1]
        print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))
    else:
        print("ROC AUC: skipped because model trained on only one class")

    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": feature_columns,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
