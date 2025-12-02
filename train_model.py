"""
Train a synthetic regression model to predict surgical tool usage.
Generates synthetic data, trains a RandomForest pipeline, evaluates it, and saves model.pkl.
"""

import random
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    operation_types: List[str] = [
        "Appendectomy",
        "C-Section",
        "Knee Replacement",
        "CABG",
        "Cholecystectomy",
    ]
    tools: List[str] = [
        "Scalpel",
        "Forceps",
        "Retractor",
        "Suction",
        "Laparoscope",
    ]

    rows = []
    for _ in range(n_samples):
        op = random.choice(operation_types)
        tool = random.choice(tools)
        duration = np.random.randint(30, 301)  # 30 to 300 min
        complexity = np.random.randint(1, 6)  # 1 to 5
        experience = np.random.randint(1, 31)  # 1 to 30 years

        base_tool = {
            "Scalpel": 10,
            "Forceps": 15,
            "Retractor": 8,
            "Suction": 20,
            "Laparoscope": 5,
        }[tool]

        op_factor = {
            "Appendectomy": 0.8,
            "C-Section": 1.0,
            "Knee Replacement": 1.2,
            "CABG": 1.5,
            "Cholecystectomy": 1.1,
        }[op]

        usage = (
            base_tool * op_factor
            + 0.05 * duration
            + 1.5 * complexity
            - 0.1 * experience
            + np.random.normal(0, 3)
        )

        usage = max(1, round(usage))

        rows.append(
            {
                "operation_type": op,
                "tool_name": tool,
                "surgery_duration_min": duration,
                "complexity_score": complexity,
                "surgeon_experience_years": experience,
                "usage_count": usage,
            }
        )

    df = pd.DataFrame(rows)
    return df


def train_and_save_model():
    df = generate_synthetic_data(n_samples=3000)
    print("Sample of training data:")
    print(df.head())

    X = df.drop(columns=["usage_count"])
    y = df["usage_count"]

    categorical_features = ["operation_type", "tool_name"]
    numeric_features = [
        "surgery_duration_min",
        "complexity_score",
        "surgeon_experience_years",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on test: {mae:.2f} usages")

    artifacts = {
        "pipeline": pipeline,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    joblib.dump(artifacts, "model.pkl")
    print("Trained model saved to model.pkl")


if __name__ == "__main__":
    train_and_save_model()
