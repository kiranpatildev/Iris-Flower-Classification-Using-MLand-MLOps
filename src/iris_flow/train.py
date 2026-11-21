"""Train, evaluate, and log a reproducible Random Forest experiment."""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from iris_flow.constants import FEATURES, SPECIES

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import mlflow

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / "data" / "iris.csv")
    x = frame[FEATURES]
    y = frame["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"], stratify=y,
    )
    model = RandomForestClassifier(**params["model"])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    report = {
        "accuracy": round(float(accuracy), 4),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "per_class": classification_report(
            y_test, predicted, target_names=SPECIES, output_dict=True,
            zero_division=0,
        ),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "iris_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("iris-random-forest")
    with mlflow.start_run():
        mlflow.log_params({**params["model"], **params["split"]})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(str(ROOT / "reports" / "metrics.json"))
        mlflow.log_artifact(str(ROOT / "models" / "iris_model.joblib"))
    print(f"Accuracy: {accuracy:.4f} on {len(x_test)} test samples")


if __name__ == "__main__":
    main()

