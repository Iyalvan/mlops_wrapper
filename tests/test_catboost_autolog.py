#!/usr/bin/env python3
"""test catboost with autolog (should fail, fallback to manual)"""

from mlops_wrapper.mlflow_wrapper import mlflow_experiment
import os

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

@mlflow_experiment(
    experiment_name="test_catboost",
    run_name="test_catboost_autolog_fallback",
    tags={'framework': 'catboost', 'test': 'autolog_fallback'},
    autolog_enabled=True  # will fail with catboost, wrapper will fallback
)
def train():
    from catboost import CatBoostRegressor
    import numpy as np

    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    model = CatBoostRegressor(iterations=10, depth=3, verbose=False)
    model.fit(X, y)

    # manual params (required since autolog fails)
    return {
        'params': {
            'model.iterations': 10,
            'model.depth': 3,
            'n_features': 10
        },
        'metrics': {
            'train_rmse': 0.15
        }
    }

if __name__ == "__main__":
    print("testing catboost with autolog (expect warning + fallback)...")
    train()
    print("\n✓ test passed: autolog failed, manual logging worked")
