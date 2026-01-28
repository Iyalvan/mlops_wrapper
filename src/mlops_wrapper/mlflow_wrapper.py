import mlflow
import functools
import uuid
import datetime
import os


def _detect_model_framework(model) -> str:
    """
    detect ml framework from model instance

    returns framework name: 'catboost', 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow', or 'unknown'
    """
    model_module = type(model).__module__
    model_class = type(model).__name__

    # check module path for framework identification
    if 'catboost' in model_module.lower():
        return 'catboost'
    elif 'sklearn' in model_module.lower():
        return 'sklearn'
    elif 'xgboost' in model_module.lower():
        return 'xgboost'
    elif 'lightgbm' in model_module.lower() or 'lgb' in model_module.lower():
        return 'lightgbm'
    elif 'torch' in model_module.lower() or 'pytorch' in model_module.lower():
        return 'pytorch'
    elif 'tensorflow' in model_module.lower() or 'keras' in model_module.lower():
        return 'tensorflow'

    return 'unknown'


def _create_robust_signature(model_inputs, model_predictions):
    """
    create mlflow signature with integer columns converted to double

    prevents schema enforcement errors when integer columns may have missing values
    at inference time. common for boolean features encoded as integers.

    note: this is an opinionated choice. if your model genuinely uses integer
    features that should never have missing values, override by providing your
    own signature in the return dict.

    args:
        model_inputs: training features (dataframe or array)
        model_predictions: model predictions for signature output

    returns:
        mlflow ModelSignature or None if inference fails
    """
    try:
        from mlflow.models.signature import infer_signature
        from mlflow.types.schema import Schema, ColSpec, DataType

        # infer base signature
        base_signature = infer_signature(model_inputs, model_predictions)

        # convert integer columns to double to handle missing values
        if base_signature and base_signature.inputs:
            updated_inputs = []
            for col_spec in base_signature.inputs.inputs:
                if col_spec.type in [DataType.integer, DataType.long]:
                    updated_inputs.append(ColSpec(type=DataType.double, name=col_spec.name))
                else:
                    updated_inputs.append(col_spec)

            return mlflow.models.signature.ModelSignature(
                inputs=Schema(updated_inputs),
                outputs=base_signature.outputs
            )

        return base_signature

    except Exception as e:
        print(f"  warning: signature inference failed: {e}")
        return None


def _log_model_by_framework(
    model,
    artifact_path: str = 'model',
    signature=None,
    input_example=None,
    model_inputs=None,
    model_predictions=None,
    **kwargs
):
    """
    log model using framework-specific mlflow logging function

    automatically detects framework and uses appropriate mlflow.<framework>.log_model()
    falls back to mlflow.pyfunc for unknown frameworks

    args:
        model: trained model instance
        artifact_path: path within run artifacts (default: 'model') - uses 'name' param internally
        signature: mlflow model signature (optional - auto-generated if model_inputs provided)
        input_example: sample input for inference testing (optional - auto-generated from model_inputs)
        model_inputs: training inputs for auto-generating signature (optional)
        model_predictions: training predictions for auto-generating signature (optional)
        **kwargs: additional arguments passed to framework log_model function

    raises:
        exception if model logging fails
    """
    framework = _detect_model_framework(model)

    print(f"logging model (framework: {framework}, path: {artifact_path})")

    # auto-generate signature if not provided but inputs available
    if signature is None and model_inputs is not None and model_predictions is not None:
        signature = _create_robust_signature(model_inputs, model_predictions)

    # auto-generate input_example if not provided but inputs available
    if input_example is None and model_inputs is not None:
        try:
            import pandas as pd
            if isinstance(model_inputs, pd.DataFrame):
                input_example = model_inputs.head(1)
            elif hasattr(model_inputs, '__getitem__'):
                input_example = model_inputs[:1]
        except Exception as e:
            print(f"  info: input_example generation failed: {e}")
            pass

    # use 'name' instead of deprecated 'artifact_path'
    model_kwargs = {'name': artifact_path, 'signature': signature, 'input_example': input_example, **kwargs}

    try:
        if framework == 'catboost':
            import mlflow.catboost
            mlflow.catboost.log_model(cb_model=model, **model_kwargs)
        elif framework == 'sklearn':
            import mlflow.sklearn
            mlflow.sklearn.log_model(sk_model=model, **model_kwargs)
        elif framework == 'xgboost':
            import mlflow.xgboost
            mlflow.xgboost.log_model(xgb_model=model, **model_kwargs)
        elif framework == 'lightgbm':
            import mlflow.lightgbm
            mlflow.lightgbm.log_model(lgb_model=model, **model_kwargs)
        elif framework == 'pytorch':
            import mlflow.pytorch
            mlflow.pytorch.log_model(pytorch_model=model, **model_kwargs)
        elif framework == 'tensorflow':
            import mlflow.tensorflow
            mlflow.tensorflow.log_model(model=model, **model_kwargs)
        else:
            # fallback to pyfunc for unknown frameworks
            print(f"  warning: unknown framework '{framework}', using mlflow.pyfunc")
            import mlflow.pyfunc
            mlflow.pyfunc.log_model(python_model=model, **model_kwargs)

        print(f"  ✓ model logged successfully")

        # tag framework for later reference
        mlflow.set_tag('mlops.model_framework', framework)

        # warn if signature/input_example missing (best practice)
        if signature is None:
            print("  info: consider adding 'signature' for better model serving")
        if input_example is None:
            print("  info: consider adding 'input_example' for inference validation")

    except Exception as e:
        print(f"  ✗ model logging failed: {e}")
        raise


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
    log_model: bool = True,
):
    """
    decorator for wrapping training functions. it ensures:
    - the mlflow run is started with standardized naming.
    - logs parameters, metrics, models, and artifacts from the function's return.
    - handles framework-specific model logging (catboost, sklearn, xgboost, lightgbm, pytorch, tensorflow).
    - catches exceptions and logs them.

    args:
        experiment_name: name of the mlflow experiment
        run_name: optional name for the run
        tags: optional dict of tags to add to the run
        tracking_uri: mlflow tracking server uri
        autolog_enabled: enable mlflow autolog for supported frameworks
        autolog_config: optional config dict for autolog
        log_model: if True, logs model artifact to run. set False for local
                   development runs to save storage and time. default: True.

    the wrapped function should return a dictionary with keys:
        - 'params': dict of hyperparameters (required if autolog fails)
        - 'metrics': dict of evaluation metrics (required if autolog fails)
        - optional 'model': trained model instance (recommended - enables automatic framework-specific logging)
        - optional 'signature': mlflow model signature (recommended for model serving)
        - optional 'input_example': sample input data (recommended for inference validation)
        - optional 'artifact_path': custom path for model artifact (default: 'model')
        - optional 'artifacts': path to additional artifacts directory
        - optional 'model_local_path': path to pre-logged model for registration (legacy)
        - optional 'model_name': model registry name for automatic registration (legacy)

    note: if you return 'model', the wrapper automatically:
        - detects the framework (catboost, sklearn, xgboost, etc.)
        - logs using appropriate mlflow.<framework>.log_model()
        - handles cases where autolog doesn't work (e.g., catboost)
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
                    print("  this typically means:")
                    print("    - framework has no autolog support (e.g., catboost)")
                    print("    - model was not trained within the mlflow run context")
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

                # handle model logging (framework-agnostic)
                if 'model' in result and log_model:
                    try:
                        _log_model_by_framework(
                            model=result['model'],
                            artifact_path=result.get('artifact_path', 'model'),
                            signature=result.get('signature'),
                            input_example=result.get('input_example'),
                            model_inputs=result.get('model_inputs'),
                            model_predictions=result.get('model_predictions')
                        )
                        mlflow.set_tag('mlops.model_logged_via', 'explicit')
                    except Exception as model_log_error:
                        # if model logging fails, warn but don't fail the whole run
                        print(f"warning: model logging failed: {model_log_error}")
                        mlflow.set_tag('mlops.model_log_error', str(model_log_error)[:250])

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
