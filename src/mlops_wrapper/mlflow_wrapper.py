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
        autolog_enabled: bool = False,
        autolog_config: dict = None,
    ):
        """
        initialize the mlflow wrapper with experiment name, optional run prefix, tracking uri,
        autolog_enabled flag, and autolog_config parameters dynamically passed.
        """
        self.experiment_name = experiment_name
        self.run_name_prefix = run_name_prefix
        self.autolog_enabled = autolog_enabled
        self.autolog_config = autolog_config or {}

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _generate_run_name(self) -> str:
        """
        generate a unique run name using prefix, timestamp, and uuid.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        return f"{self.run_name_prefix}_{timestamp}_{unique_id}"

    def start_run(self, run_name: str = None, tags: dict = None):
        """
        start an mlflow run. if autolog is enabled, it's called before starting the run.
        autolog must be enabled before the run starts to properly hook into training APIs.

        args:
            run_name: optional run name (if None, generates uuid-based name)
            tags: optional dict of tags to log (project-specific metadata)
        """
        # setup aws credentials if using s3 backend
        aws_profile = os.getenv('AWS_PROFILE')
        if aws_profile:
            try:
                import boto3
                boto3.setup_default_session(profile_name=aws_profile)
                print(f"  aws profile: {aws_profile}")
            except Exception as e:
                print(f"warning: aws profile setup failed: {e}")

        if not run_name:
            run_name = self._generate_run_name()

        # enable autolog BEFORE starting run if requested (per mlflow docs)
        if self.autolog_enabled:
            mlflow.autolog(**self.autolog_config)

        # always explicitly start a run with proper name
        self.run = mlflow.start_run(run_name=run_name)

        # common tagging for both autolog and manual modes
        mlflow.set_tag("experiment_name", self.experiment_name)
        mlflow.set_tag("run_name", run_name)

        # add project-specific tags if provided
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, str(value))
            print(f"  custom tags: {len(tags)} tags added")

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
        mlflow.log_param("model_version", result.version)
        return result

    def end_run(self):
        """
        end the current mlflow run.
        """
        mlflow.end_run()


def mlflow_experiment(
    experiment_name: str,
    run_name: str = None,
    tags: dict = None,
    tracking_uri: str = None,
    autolog_enabled: bool = True,
    autolog_config: dict = None,
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
                experiment_name,
                tracking_uri=tracking_uri,
                autolog_enabled=autolog_enabled,
                autolog_config=autolog_config,
            )

            try:
                mlflow_wrapper.start_run(run_name=run_name, tags=tags)
                result = func(*args, **kwargs) or {}
                active_run = mlflow.active_run()
                if active_run is None:
                    raise RuntimeError("no active mlflow run after training")

                # check if autolog captured anything
                autolog_params = set(active_run.data.params.keys())
                autolog_metrics = set(active_run.data.metrics.keys())

                # detect autolog failure (when enabled but captured nothing)
                if autolog_enabled and len(autolog_params) == 0:
                    print("warning: autolog enabled but captured 0 params")
                    print("  framework may not support autolog (e.g., catboost)")
                    print("  falling back to manual param logging")

                    # require params in return dict
                    if not isinstance(result.get("params"), dict) or len(result.get("params", {})) == 0:
                        raise ValueError(
                            "autolog failed and no params in return dict. "
                            "please extract model hyperparameters manually and return {'params': {...}}"
                        )

                # detect collisions BEFORE logging custom params/metrics
                custom_params = result.get("params", {})
                custom_metrics = result.get("metrics", {})

                param_collisions = autolog_params & set(custom_params.keys())
                metric_collisions = autolog_metrics & set(custom_metrics.keys())

                if param_collisions:
                    print(f"warning: param collision detected: {sorted(param_collisions)}")
                    print("  autolog values retained, skipping duplicate custom params")
                    custom_params = {k: v for k, v in custom_params.items() if k not in param_collisions}

                if metric_collisions:
                    print(f"warning: metric collision detected: {sorted(metric_collisions)}")
                    print("  autolog values retained, skipping duplicate custom metrics")
                    print("  suggestion: prefix custom metrics (e.g., 'custom.metric_name')")
                    custom_metrics = {k: v for k, v in custom_metrics.items() if k not in metric_collisions}

                # log non-colliding custom params/metrics
                if custom_params:
                    mlflow_wrapper.log_params(custom_params)
                if custom_metrics:
                    mlflow_wrapper.log_metrics(custom_metrics)

                if "artifacts" in result:
                    mlflow_wrapper.log_artifacts(result["artifacts"])

                if result.get("model_local_path") and result.get("model_name"):
                    reg_result = mlflow_wrapper.register_model(
                        result["model_local_path"], result["model_name"]
                    )
                    mlflow.log_param("model_version", reg_result.version)
                
                # mark run as successful
                mlflow.set_tag("mlops.status", "success")
            except Exception as e:
                # log error as tags, not params (params are for hyperparameters)
                mlflow.set_tag("mlops.status", "failed")
                mlflow.set_tag("mlops.error_type", type(e).__name__)
                mlflow.set_tag("mlops.error_message", str(e)[:500])  # truncate long errors
                raise
            finally:
                mlflow_wrapper.end_run()
            return result
        return wrapper
    return decorator
