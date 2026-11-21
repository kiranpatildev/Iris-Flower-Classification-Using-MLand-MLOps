# Iris Flow: Iris classification with MLOps

An end-to-end Iris flower classifier. It uses scikit-learn's bundled 150-row dataset, a Random Forest model, DVC for a reproducible pipeline, MLflow for experiment logs, pytest for checks, FastAPI for predictions, GitHub Actions for CI, and Docker for packaging.

## Run locally (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
$env:PATH = "$(Resolve-Path .venv\Scripts);$env:PATH"
dvc repro
python -m pytest -q
uvicorn iris_flow.api:app --reload
```

Open **http://127.0.0.1:8000**. API documentation is at **http://127.0.0.1:8000/docs**. DVC is already initialized in this repository; use `dvc repro` after changing data, code, or parameters. To inspect experiment runs, start `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open http://127.0.0.1:5000.

## API example

```powershell
$body = @{ sepal_length=5.1; sepal_width=3.5; petal_length=1.4; petal_width=0.2 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The response contains the predicted species, its confidence, and probabilities for all three species. Inputs must be positive and at most 10 cm. The model uses a stratified 80/20 split with fixed random seeds. The generated data and model are stored in the local DVC cache; commit `dvc.lock` to record exact pipeline outputs. MLflow stores experiment metadata in `mlflow.db` and artifacts under `mlruns/`.

## Docker

First run `dvc repro` so `data/iris.csv` and `models/iris_model.joblib` exist, then:

```powershell
docker build -t iris-flow .
docker run --rm -p 8000:8000 iris-flow
```

## GitHub

Git is the version control tool; GitHub is the hosting service. GitHub CLI (`gh`) creates a remote repository. From this directory, after signing in:

```powershell
gh auth login
git add .
git commit -m "Initial Iris MLOps project"
gh repo create iris-flow --private --source . --remote origin --push
```

Do not commit generated `mlruns/` or the local DVC cache. DVC's local cache is not uploaded to GitHub by this command; configure a DVC remote separately if collaborators need the exact generated data and model without reproducing the pipeline.

## What to show in a project demo

1. **Problem and dataset:** Four centimeter measurements, three Iris species, 150 bundled examples (50 per class).
2. **DVC pipeline:** Show `dvc.yaml`, then run `dvc repro`; explain the prepare and train stages. Show `dvc.lock` and `dvc metrics show`.
3. **Training evidence:** Show `reports/metrics.json` accuracy, per-class metrics, and confusion matrix. Open MLflow and show the logged parameters, accuracy, and model artifact.
4. **Working UI:** Open the page, try all three preset buttons, then enter your own values and show the confidence bars.
5. **API and validation:** Open `/docs`, send a valid `/predict` request, and show that an invalid negative value is rejected.
6. **Quality and delivery:** Run `pytest -q`, show the GitHub Actions workflow, and show the Docker build and running container.

For screenshots or a presentation, capture the terminal after `dvc repro`, the MLflow run, one prediction in the UI, the API docs, and a passing test/CI run. Report the accuracy measured by your run; don't claim a fixed accuracy before running the pipeline.

