# mlops_wrapper

schema-agnostic mlflow wrapper for consistent experiment tracking across ML projects.

## what is this?

a python wrapper around mlflow that handles:
- run lifecycle (start, log, end)
- autolog orchestration with intelligent fallback
- framework-agnostic model logging (catboost, sklearn, xgboost, lightgbm, pytorch, tensorflow)
- automatic signature inference with type conversion for robust serving
- collision detection for params/metrics
- error handling and tagging
- AWS credentials setup for S3 artifact storage

**core principle:** wrapper is generic. your project builds metadata (tags, run names) from its config and passes them in.

## why use this?

**without wrapper:**
```python
# 30+ lines per training script
mlflow.set_tracking_uri(...)
mlflow.set_experiment(...)
mlflow.start_run(run_name=...)
mlflow.set_tag("tag1", value1)
mlflow.set_tag("tag2", value2)
# ... 20 more tags
mlflow.log_param("param1", value1)
# ... handle autolog failures
# ... detect collisions
# ... error handling
mlflow.end_run()
```

**with wrapper:**
```python
@mlflow_experiment(
    experiment_name="my-project",
    run_name="exp1_catboost_20251225_143052",
    tags={...},  # project builds these
)
def train():
    return {'params': {...}, 'metrics': {...}}
```

## design principles

1. **schema-agnostic:** no hardcoded assumptions about your config structure
2. **metadata comes from project:** wrapper accepts pre-built tags/run_name
3. **autolog with fallback:** tries autolog first, falls back to manual if it fails
4. **collision detection:** warns if your metrics clash with autolog
5. **error transparency:** logs failures as tags, doesn't swallow errors

## responsibilities

**wrapper handles:**
- mlflow lifecycle (start_run, end_run)
- autolog enablement (before start_run per mlflow docs)
- autolog failure detection (0 params logged = failure)
- param/metric collision detection and warnings
- error logging as tags (mlops.status, mlops.error_type)
- AWS credentials setup (boto3 session from AWS_PROFILE)

**your project handles:**
- building run names from your config schema
- extracting tags from your domain (cost filters, data splits, etc.)
- extracting model hyperparameters manually when autolog fails

## installation

```bash
pip install mlops-wrapper
```

or for development:
```bash
pip install -e /path/to/mlops_wrapper_repo
```

## usage

### basic example (sklearn - autolog works)

```python
from mlops_wrapper.mlflow_wrapper import mlflow_experiment

@mlflow_experiment(
    experiment_name="sklearn-test",
    run_name="rf_experiment_20251225",
    tags={'model': 'random_forest', 'team': 'ml-ops'},
    autolog_enabled=True
)
def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20)
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X, y)

    # autolog captures params automatically
    # return custom metrics
    return {
        'metrics': {'custom_score': 0.95}
    }

train()
```

### catboost example (autolog fails - manual fallback)

```python
@mlflow_experiment(
    experiment_name="catboost-test",
    run_name="cb_experiment_20251225",
    tags={'model': 'catboost', 'task': 'regression'},
    autolog_enabled=True  # will fail, wrapper detects and requires manual params
)
def train():
    from catboost import CatBoostRegressor
    import numpy as np

    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    model = CatBoostRegressor(iterations=10, depth=3, verbose=False)
    model.fit(X, y)

    # autolog will capture 0 params (known mlflow issue #6073)
    # wrapper detects this and requires manual params
    return {
        'params': {  # required when autolog fails
            'model.iterations': 10,
            'model.depth': 3,
            'n_features': 10
        },
        'metrics': {
            'train_rmse': 0.15
        }
    }

train()
```

**what happens:**
```
warning: autolog enabled but captured 0 params
  this typically means:
    - framework has no autolog support (e.g., catboost)
    - model was not trained within the mlflow run context
  falling back to manual param logging
```

### model logging example

```python
@mlflow_experiment(
    experiment_name="catboost-model-logging",
    run_name="cb_with_model_20260128",
    tags={'model': 'catboost'},
    log_model=True  # enable model artifact logging (default: True)
)
def train():
    from catboost import CatBoostRegressor
    import pandas as pd
    import numpy as np

    # prepare data
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y_train = np.random.rand(100)

    model = CatBoostRegressor(iterations=50, depth=4, verbose=False)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    # return model with inputs for signature inference
    return {
        'params': {'iterations': 50, 'depth': 4},
        'metrics': {'train_rmse': 0.12},
        'model': model,  # triggers automatic framework detection and logging
        'model_inputs': X_train.head(5),  # for signature inference
        'model_predictions': predictions[:5]  # for signature output schema
    }

train()
```

**what happens:**
- wrapper detects framework (catboost)
- calls `mlflow.catboost.log_model()` with proper signature
- converts int64 features to double (prevents serving errors with boolean features)
- logs input_example for inference validation
- adds tags: `mlops.model_framework`, `mlops.model_logged_via`

**model logging control:**
```python
# local development - skip model logging to save storage
@mlflow_experiment(..., log_model=False)

# or via environment variable
export MLFLOW_LOG_MODEL=false  # overrides decorator parameter
```

**important:** to avoid duplicate model logging, disable autolog's model logging:
```python
@mlflow_experiment(
    ...,
    autolog_config={'log_models': False}  # let explicit logging handle it
)
```

### project-level integration pattern

```python
# in your project's training/train.py

from mlflow_integration import build_run_name, build_tags  # your helpers
from mlops_wrapper.mlflow_wrapper import mlflow_experiment

def main():
    config = load_configs()

    # project builds metadata from config
    tags = build_tags(config)  # extracts 21+ domain-specific tags
    run_name = build_run_name(
        model_type=config['model']['type'],
        filters=config['data']['filters'],
        timestamp='20251225_143052'
    )

    # pass pre-built metadata to wrapper
    wrapped_train = mlflow_experiment(
        experiment_name="cost-prediction",
        run_name=run_name,
        tags=tags,
        tracking_uri="https://mlflow.example.com",
        autolog_enabled=True
    )(train_pipeline)

    wrapped_train(config)
```

## autolog behavior

**autolog enabled (default):**
- wrapper calls `mlflow.autolog()` before `start_run()`
- after training, checks if params were captured
- if 0 params logged: warns and requires manual params in return dict
- if params logged: autolog worked, custom params/metrics optional

**supported frameworks:**
- sklearn: ✅ works
- xgboost: ✅ works
- lightgbm: ✅ works
- pytorch: ✅ works
- tensorflow: ✅ works
- catboost: ❌ broken (mlflow issue #6073) - manual fallback required

**collision detection:**
```python
# autolog captures: accuracy, loss
# you return: accuracy, custom_metric

# wrapper behavior:
# - keeps autolog's accuracy (autolog wins)
# - logs your custom_metric
# - warns: "param collision detected: ['accuracy']"
# - suggests: "prefix custom metrics (e.g., 'custom.accuracy')"
```

## model logging

**supported frameworks (auto-detected):**
- catboost: ✅ `mlflow.catboost.log_model()`
- sklearn: ✅ `mlflow.sklearn.log_model()`
- xgboost: ✅ `mlflow.xgboost.log_model()`
- lightgbm: ✅ `mlflow.lightgbm.log_model()`
- pytorch: ✅ `mlflow.pytorch.log_model()`
- tensorflow: ✅ `mlflow.tensorflow.log_model()`

**signature handling:**
- auto-generates mlflow signature from `model_inputs` and `model_predictions`
- converts int64 columns to double (prevents schema errors with boolean features)
- handles missing values at inference time

**when to use:**
- training runs: enable with `log_model=True`
- local development: disable with `log_model=False` to save storage
- CI/CD: control via `MLFLOW_LOG_MODEL` environment variable

## return dict contract

```python
{
    'params': dict,       # required if autolog fails, optional otherwise
    'metrics': dict,      # optional (custom metrics)
    'model': object,      # optional (trained model - triggers auto logging)
    'model_inputs': DataFrame/array,  # optional (for signature inference, recommend .head(5))
    'model_predictions': array,  # optional (for signature output schema)
    'signature': ModelSignature,  # optional (auto-generated if model_inputs provided)
    'input_example': DataFrame/array,  # optional (auto-generated from model_inputs)
    'artifact_path': str,  # optional (custom model artifact path, default: 'model')
    'artifacts': str,     # optional (path to additional artifact dir/file)
    'model_local_path': str,  # optional (legacy - for model registry)
    'model_name': str     # optional (legacy - for model registry)
}
```

## error handling

```python
@mlflow_experiment(experiment_name="test")
def train():
    raise ValueError("something broke")

# wrapper behavior:
# - catches exception
# - logs tags:
#   - mlops.status = "failed"
#   - mlops.error_type = "ValueError"
#   - mlops.error_message = "something broke"
# - re-raises exception
# - closes mlflow run
```

## ci/cd integration


### github actions example

```yaml
- name: train model
  env:
    MLFLOW_TRACKING_URI: https://mlflow.company.com
    AWS_PROFILE: ml-dev # or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY or github aws jwt auth
  run: |
    python training/train.py --run-prefix ci-${{ github.run_id }}
```

wrapper automatically picks up:
- `MLFLOW_TRACKING_URI` env var
- `AWS_PROFILE` for boto3 session

## testing

### verify wrapper is working

```python
# test_wrapper.py
from mlops_wrapper.mlflow_wrapper import mlflow_experiment
import os

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

@mlflow_experiment(
    experiment_name="test-run",
    run_name="verify-wrapper-works",
    tags={'test': 'true'}
)
def test_train():
    return {
        'params': {'test_param': 123},
        'metrics': {'test_metric': 0.99}
    }

test_train()
print("✓ check ./mlruns for logged experiment")
```

run:
```bash
python test_wrapper.py
mlflow ui --backend-store-uri file:./mlruns
```

open http://localhost:5000 and verify:
- experiment "test-run" exists
- run "verify-wrapper-works" logged
- tags, params, metrics present

## migration from manual mlflow

**before (manual):**
```python
def train():
    mlflow.set_tracking_uri(...)
    mlflow.set_experiment(...)

    with mlflow.start_run(run_name=...):
        mlflow.set_tag(...)
        # training
        mlflow.log_param(...)
        mlflow.log_metric(...)
        mlflow.log_artifacts(...)
    # 50+ lines
```

**after (wrapper):**
```python
@mlflow_experiment(experiment_name=..., run_name=..., tags={...})
def train():
    # training
    return {'params': {...}, 'metrics': {...}}
# 5 lines
```

## requirements

- python >=3.8
- mlflow >=2.9.0 (supports 3.x)
- boto3 >=1.28.0 (for S3 artifact storage)

## contributing

issues and PRs welcome at https://github.com/Iyalvan/mlops_wrapper

## license

see LICENSE file
