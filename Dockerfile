FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY requirements-runtime.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-runtime.txt && pip install --no-cache-dir --no-deps .
COPY web ./web
COPY data ./data
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "iris_flow.api:app", "--host", "0.0.0.0", "--port", "8000"]

