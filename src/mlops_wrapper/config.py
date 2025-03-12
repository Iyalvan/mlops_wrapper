import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "42_experiment")
# S3_BUCKET = os.getenv("S3_BUCKET", "mlflow-models-bucket")
