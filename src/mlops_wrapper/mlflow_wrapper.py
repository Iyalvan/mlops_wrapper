import mlflow
import functools
import uuid
import datetime
import os


class MLflowWrapper:
    def __init__(
        self,
        experiment_name: str,
        run_name_prefix: str = "run",
        tracking_uri: str = None,
    ):
        """
        initialize the mlflowwrapper with the experiment name and optional run name prefix.
        optionally, set the mlflow tracking uri.
        """
        self.experiment_name = experiment_name
        self.run_name_prefix = run_name_prefix
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _generate_run_name(self) -> str:
        """
        generate a unique run name using a prefix, current timestamp, and a short unique identifier.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        return f"{self.run_name_prefix}_{timestamp}_{unique_id}"

    def start_run(self, run_name: str = None):
        """
        start an mlflow run. if a run name is not provided, generate one.
        """
        if not run_name:
            run_name = self._generate_run_name()
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.set_tag("experiment_name", self.experiment_name)
        return self.run

    def log_params(self, params: dict):
        """
        log a dictionary of parameters.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        log a dictionary of metrics. optionally, include a step value.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        log artifacts (files) stored at the specified local directory.
        """
        if os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        else:
            raise FileNotFoundError(f"local directory {local_dir} does not exist.")

    def register_model(self, model_local_path: str, model_name: str):
        """
        register a model. this assumes the model is saved locally at model_local_path.
        the model is registered under model_name with mlflow's model registry.
        """
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_local_path}"
        result = mlflow.register_model(model_uri, model_name)
        mlflow.set_tag("registered_model", model_name)
        return result

    def end_run(self):
        """
        end the current mlflow run.
        """
        mlflow.end_run()


def mlflow_experiment(
    experiment_name: str, run_name_prefix: str = "run", tracking_uri: str = None
):
    """
    decorator for wrapping training functions. it ensures:
    - the mlflow run is started with standardized naming.
    - logs parameters, metrics, and artifacts from the function's return.
    - catches exceptions and logs them.

    the wrapped function should return a dictionary with keys:
        - 'params': dict of hyperparameters
        - 'metrics': dict of evaluation metrics
        - optional 'artifacts': path to artifacts (can be a file or directory)
        - optional 'model_local_path': path to the saved model for registration
        - optional 'model_name': model registry name for automatic registration
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow_wrapper = MLflowWrapper(
                experiment_name, run_name_prefix, tracking_uri
            )
            run = mlflow_wrapper.start_run()
            try:
                result = func(*args, **kwargs)

                if "params" in result and isinstance(result["params"], dict):
                    mlflow_wrapper.log_params(result["params"])
                if "metrics" in result and isinstance(result["metrics"], dict):
                    mlflow_wrapper.log_metrics(result["metrics"])

                if "artifacts" in result:
                    mlflow_wrapper.log_artifacts(result["artifacts"])

                if "model_local_path" in result and "model_name" in result:
                    reg_result = mlflow_wrapper.register_model(
                        result["model_local_path"], result["model_name"]
                    )
                    mlflow.log_param("model_version", reg_result.version)

                return result
            except Exception as e:
                mlflow.log_param("failure_reason", str(e))
                raise
            finally:
                mlflow_wrapper.end_run()

        return wrapper

    return decorator
