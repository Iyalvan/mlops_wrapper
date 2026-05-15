"""mlops_wrapper - unified mlflow integration for ML projects"""

from .mlflow_wrapper import mlflow_experiment, MLflowWrapper

from .compare import (
    compare_runs,
    compare_model_versions,
    fetch_recent_runs,
    fetch_model_versions,
    fetch_run_artifact_json,
    build_comparison_html,
    build_model_version_comparison_html,
    # disk-based comparison
    compare_disk_runs,
    load_disk_runs,
    load_run_from_disk,
    build_disk_comparison_html,
)

__all__ = [
    'mlflow_experiment',
    'MLflowWrapper',
    'compare_runs',
    'compare_model_versions',
    'fetch_recent_runs',
    'fetch_model_versions',
    'fetch_run_artifact_json',
    'build_comparison_html',
    'build_model_version_comparison_html',
    # disk-based comparison
    'compare_disk_runs',
    'load_disk_runs',
    'load_run_from_disk',
    'build_disk_comparison_html',
]