"""Export scikit-learn's bundled Iris data for DVC tracking."""
from pathlib import Path

from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "iris.csv"


def main() -> None:
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    })
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(frame)} observations to {DATA_PATH}")


if __name__ == "__main__":
    main()

