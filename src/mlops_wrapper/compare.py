# Compare MLflow runs or registered model versions and generate
"""
interactive Plotly HTML reports.

schema-agnostic: does NOT know project-specific metric names.
caller provides metric keys, formatting, and lower-is-better hints.

supports:
  - compare recent experiment runs
  - compare registered model versions
"""

import html as html_lib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any

import mlflow
from mlflow.tracking import MlflowClient

try:
    import pandas as pd  # noqa: F401
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    pd = None
    go = None
    _PLOTLY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# metric formatting
# ═══════════════════════════════════════════════════════════════════════════════

Formatter = Callable[[float], str]

_FORMAT_REGISTRY: Dict[str, Formatter] = {
    'dollar':     lambda v: f"${v:,.2f}",
    'dollar_int': lambda v: f"${v:,.0f}",
    'percent':    lambda v: f"{v * 100:.2f}%",   # input 0.0–1.0
    'pct':        lambda v: f"{v:.1f}%",         # input already 0–100
    'int':        lambda v: f"{int(v):,}",
    '.2f':        lambda v: f"{v:,.2f}",
    '.3f':        lambda v: f"{v:.3f}",
    '.4f':        lambda v: f"{v:.4f}",
    'default':    lambda v: f"{v:,.4f}",
}


def _get_formatter(key: str, metric_formats: Optional[Dict[str, Any]] = None) -> Formatter:
    if metric_formats and key in metric_formats:
        fmt = metric_formats[key]
        if isinstance(fmt, str):
            if fmt in _FORMAT_REGISTRY:
                return _FORMAT_REGISTRY[fmt]
            return lambda v, f=fmt: f"{v:{f}}"
        if callable(fmt):
            return fmt
    return _FORMAT_REGISTRY['default']


def _format_value(value: Optional[float], key: str, metric_formats: Optional[Dict[str, Any]] = None) -> str:
    if value is None:
        return '–'
    try:
        return _get_formatter(key, metric_formats)(value)
    except Exception:
        return str(value)


# ═══════════════════════════════════════════════════════════════════════════════
# MLflow fetch helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_recent_runs(
    experiment_name: str,
    n_runs: int = 5,
    tracking_uri: str = None,
    filter_string: str = "",
    only_successful: bool = True,
) -> List[dict]:
    """
    Fetch last N runs from an MLflow experiment.

    Returns newest first.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment '{experiment_name}' not found on {mlflow.get_tracking_uri()}")

    filters = []
    if only_successful:
        filters.append("tags.`mlops.status` = 'success'")
    if filter_string:
        filters.append(filter_string)
    combined_filter = " AND ".join(filters) if filters else ""

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=combined_filter,
        order_by=["start_time DESC"],
        max_results=n_runs,
    )

    results = []
    for run in runs:
        results.append({
            'run_id':     run.info.run_id,
            'run_name':   run.info.run_name or run.info.run_id[:8],
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
            'end_time':   datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            'metrics':    dict(run.data.metrics),
            'params':     dict(run.data.params),
            'tags':       dict(run.data.tags),
            'status':     run.info.status,
            'source_type': 'run',
        })
    return results


def fetch_model_versions(
    model_name: str,
    n_versions: int = 5,
    tracking_uri: str = None,
    stages: List[str] = None,
    aliases: List[str] = None,
) -> List[dict]:
    """
    Fetch last N versions of a registered model and hydrate linked run data.

    Returns newest version first.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        raise ValueError(f"registered model '{model_name}' not found")

    if not all_versions:
        raise ValueError(f"no versions found for model '{model_name}'")

    all_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

    if stages:
        stages_lower = {s.lower() for s in stages}
        all_versions = [
            v for v in all_versions
            if getattr(v, 'current_stage', '').lower() in stages_lower
        ]

    if aliases:
        aliases_set = set(aliases)
        all_versions = [
            v for v in all_versions
            if aliases_set & set(getattr(v, 'aliases', []))
        ]

    versions = all_versions[:n_versions]
    results = []

    for mv in versions:
        run_id = getattr(mv, 'run_id', None)
        if not run_id:
            continue

        try:
            run = client.get_run(run_id)
        except Exception:
            continue

        results.append({
            'run_id':         run.info.run_id,
            'run_name':       run.info.run_name or run.info.run_id[:8],
            'start_time':     datetime.fromtimestamp(run.info.start_time / 1000),
            'end_time':       datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            'metrics':        dict(run.data.metrics),
            'params':         dict(run.data.params),
            'tags':           dict(run.data.tags),
            'status':         run.info.status,
            'model_version':  int(mv.version),
            'model_stage':    getattr(mv, 'current_stage', ''),
            'model_aliases':  list(getattr(mv, 'aliases', [])),
            'model_name':     model_name,
            'source_type':    'model_version',
        })

    return results


def fetch_run_artifact_json(
    run_id: str,
    artifact_path: str,
    tracking_uri: str = None,
) -> Optional[dict]:
    """
    Download and parse a JSON artifact for a run.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
            with open(local_path, 'r') as f:
                return json.load(f)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# discovery/grouping helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_metric_keys(items: List[dict]) -> List[str]:
    all_keys = set()
    for item in items:
        all_keys.update(item['metrics'].keys())
    return sorted(all_keys)


def _group_metrics_by_split(metric_keys: List[str]) -> tuple:
    prefix_to_split = {
        'train_': 'train',
        'val_': 'val',
        'validation_': 'validation',
        'test_': 'test',
    }

    grouped = {}
    ungrouped = []

    for key in metric_keys:
        matched = False
        for prefix, split in prefix_to_split.items():
            if key.startswith(prefix):
                base = key[len(prefix):]
                grouped.setdefault(base, {})[split] = key
                matched = True
                break
        if not matched:
            ungrouped.append(key)

    return grouped, ungrouped


# ═══════════════════════════════════════════════════════════════════════════════
# HTML helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _trend_arrow(current: float, previous: float, lower_is_better: bool = False) -> str:
    if previous is None or current is None:
        return ''
    diff = current - previous
    if abs(diff) < 1e-12:
        return '<span style="color:#888">–</span>'

    improved = (diff < 0) if lower_is_better else (diff > 0)
    color = '#00CC96' if improved else '#EF553B'
    arrow = '↓' if diff < 0 else '↑'
    pct = abs(diff / previous) * 100 if abs(previous) > 1e-12 else 0.0
    return f'<span style="color:{color}">{arrow} {pct:.1f}%</span>'


def _wrap_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ border-bottom: 3px solid #636EFA; padding-bottom: .4rem; }}
  h2 {{ color: #444; margin-top: 2.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
  th {{ background: #f7f7f7; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .best {{ background: #d4edda; font-weight: bold; }}
  .worst {{ background: #f8d7da; }}
  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
             font-size: 0.85rem; color: #888; }}
</style>
</head><body>
{body}
<div class="footer">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · mlops_wrapper compare</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# common sections
# ═══════════════════════════════════════════════════════════════════════════════

def _duration_str(item: dict) -> str:
    if item.get('start_time') and item.get('end_time'):
        dur = item['end_time'] - item['start_time']
        minutes = int(dur.total_seconds() // 60)
        seconds = int(dur.total_seconds() % 60)
        return f"{minutes}m {seconds}s"
    return ''


def _run_overview_table(items: List[dict], tag_columns: List[str] = None) -> str:
    tag_columns = tag_columns or []

    header = "<tr><th>#</th><th>Run Name</th><th>Run ID</th><th>Start Time</th><th>Duration</th>"
    for tag in tag_columns:
        header += f"<th>{html_lib.escape(tag)}</th>"
    header += "</tr>"

    rows = ""
    for i, item in enumerate(items):
        rows += (
            f"<tr><td>{i+1}</td>"
            f"<td>{html_lib.escape(item['run_name'])}</td>"
            f"<td><code>{item['run_id'][:8]}</code></td>"
            f"<td>{item['start_time'].strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td>{_duration_str(item)}</td>"
        )
        for tag in tag_columns:
            rows += f"<td>{html_lib.escape(str(item['tags'].get(tag, '–')))}</td>"
        rows += "</tr>"

    return f"<table>{header}{rows}</table>"


def _version_overview_table(items: List[dict], tag_columns: List[str] = None) -> str:
    tag_columns = tag_columns or []

    header = "<tr><th>Version</th><th>Stage/Alias</th><th>Run Name</th><th>Run ID</th><th>Date</th><th>Duration</th>"
    for tag in tag_columns:
        header += f"<th>{html_lib.escape(tag)}</th>"
    header += "</tr>"

    rows = ""
    for item in items:
        stage = item.get('model_stage', '')
        aliases = item.get('model_aliases', [])
        display = ', '.join(aliases) if aliases else (stage or '–')

        css = ''
        if stage.lower() == 'production' or 'champion' in aliases:
            css = ' class="best"'

        rows += (
            f"<tr{css}>"
            f"<td><b>v{item['model_version']}</b></td>"
            f"<td>{html_lib.escape(display)}</td>"
            f"<td>{html_lib.escape(item['run_name'])}</td>"
            f"<td><code>{item['run_id'][:8]}</code></td>"
            f"<td>{item['start_time'].strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td>{_duration_str(item)}</td>"
        )
        for tag in tag_columns:
            rows += f"<td>{html_lib.escape(str(item['tags'].get(tag, '–')))}</td>"
        rows += "</tr>"

    return f"<table>{header}{rows}</table>"


def _metrics_comparison_table(
    items: List[dict],
    metric_keys: List[str],
    lower_is_better_keys: Optional[Set[str]] = None,
    metric_formats: Optional[Dict[str, Any]] = None,
    version_mode: bool = False,
    key_labels: Optional[Dict[str, str]] = None,
) -> str:
    lower_is_better_keys = lower_is_better_keys or set()
    metric_formats = metric_formats or {}

    header = "<tr><th>Metric</th>"
    for item in items:
        if version_mode:
            extra = ''
            if item.get('model_aliases'):
                extra = f"<br><small>({', '.join(item['model_aliases'])})</small>"
            elif item.get('model_stage'):
                extra = f"<br><small>[{item['model_stage']}]</small>"
            header += f"<th>v{item['model_version']}{extra}</th>"
        else:
            label = item['run_name'] if len(item['run_name']) <= 30 else item['run_name'][:27] + '...'
            header += f"<th>{html_lib.escape(label)}<br><small>{item['start_time'].strftime('%m/%d')}</small></th>"
    header += "<th>Trend</th></tr>"

    rows = ""
    for key in metric_keys:
        values = [item['metrics'].get(key) for item in items]
        non_none = [v for v in values if v is not None]
        lower_better = key in lower_is_better_keys

        best_val = None
        worst_val = None
        if len(non_none) > 1:
            best_val = min(non_none) if lower_better else max(non_none)
            worst_val = max(non_none) if lower_better else min(non_none)

        display_name = key_labels.get(key, key) if key_labels else key
        rows += f"<tr><td><b>{html_lib.escape(display_name)}</b></td>"
        for v in values:
            if v is None:
                rows += "<td>–</td>"
            else:
                css = ''
                if best_val is not None:
                    if v == best_val:
                        css = ' class="best"'
                    elif v == worst_val:
                        css = ' class="worst"'
                rows += f"<td{css}>{_format_value(v, key, metric_formats)}</td>"

        trend = ''
        if len(values) >= 2 and values[0] is not None and values[1] is not None:
            trend = _trend_arrow(values[0], values[1], lower_better)
        rows += f"<td>{trend}</td></tr>"

    return f"<table>{header}{rows}</table>"


def _metric_trend_charts(items: List[dict], metric_keys: List[str], version_mode: bool = False) -> str:
    items_chrono = list(reversed(items))
    x_labels = (
        [f"v{i['model_version']}" for i in items_chrono]
        if version_mode
        else [i['run_name'][:20] for i in items_chrono]
    )

    grouped, ungrouped = _group_metrics_by_split(metric_keys)
    charts_html = ""
    colors = {'train': '#636EFA', 'val': '#EF553B', 'validation': '#EF553B', 'test': '#00CC96'}

    for base_metric, split_keys in grouped.items():
        fig = go.Figure()
        for split in ['train', 'val', 'validation', 'test']:
            if split not in split_keys:
                continue
            key = split_keys[split]
            y_vals = [i['metrics'].get(key) for i in items_chrono]
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=y_vals,
                mode='lines+markers',
                name=split,
                marker_color=colors.get(split, '#999'),
                hovertemplate=f"<b>{split}</b><br>%{{x}}<br>{base_metric}: %{{y:.4f}}<extra></extra>",
            ))
        fig.update_layout(
            title=f"{base_metric} across {'model versions' if version_mode else 'runs'}",
            xaxis_title="Model Version" if version_mode else "Run (oldest → newest)",
            yaxis_title=base_metric,
            height=350,
            margin=dict(t=50, b=60),
            legend=dict(orientation='h', y=-0.2),
        )
        charts_html += _fig_to_html(fig)

    for key in ungrouped:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=[i['metrics'].get(key) for i in items_chrono],
            mode='lines+markers',
            name=key,
            marker_color='#636EFA',
        ))
        fig.update_layout(
            title=f"{key} across {'model versions' if version_mode else 'runs'}",
            xaxis_title="Model Version" if version_mode else "Run (oldest → newest)",
            yaxis_title=key,
            height=350,
            margin=dict(t=50, b=60),
        )
        charts_html += _fig_to_html(fig)

    return charts_html


def _radar_chart(
    items: List[dict],
    metric_keys: List[str],
    lower_is_better_keys: Optional[Set[str]] = None,
    version_mode: bool = False,
) -> str:
    lower_is_better_keys = lower_is_better_keys or set()

    if len(metric_keys) < 3:
        return "<p><em>Need at least 3 metrics for radar chart.</em></p>"

    normalized = {}
    for key in metric_keys:
        vals = [i['metrics'].get(key, 0) for i in items]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v if max_v != min_v else 1.0
        if key in lower_is_better_keys:
            normalized[key] = [(max_v - v) / rng for v in vals]
        else:
            normalized[key] = [(v - min_v) / rng for v in vals]

    fig = go.Figure()
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    labels = metric_keys + [metric_keys[0]]
    for idx, item in enumerate(items):
        values = [normalized[k][idx] for k in metric_keys]
        values.append(values[0])

        name = f"v{item['model_version']}" if version_mode else item['run_name'][:20]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=name,
            line_color=palette[idx % len(palette)],
            opacity=0.6,
        ))

    fig.update_layout(
        title="Metrics Comparison (normalized 0–1, higher = better)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        margin=dict(t=60, b=40),
    )
    return _fig_to_html(fig)


def _params_diff_table(items: List[dict]) -> str:
    all_keys = set()
    for item in items:
        all_keys.update(item['params'].keys())

    changed = []
    for key in sorted(all_keys):
        values = [item['params'].get(key, '–') for item in items]
        if len(set(str(v) for v in values)) > 1:
            changed.append((key, values))

    if not changed:
        return "<p><em>All params identical across compared items.</em></p>"

    header = "<tr><th>Param</th>"
    for item in items:
        label = f"v{item['model_version']}" if item.get('source_type') == 'model_version' else item['run_name'][:20]
        header += f"<th>{html_lib.escape(label)}</th>"
    header += "</tr>"

    rows = ""
    for key, values in changed:
        rows += f"<tr><td><b>{html_lib.escape(key)}</b></td>"
        for v in values:
            rows += f"<td>{html_lib.escape(str(v))}</td>"
        rows += "</tr>"

    return f"<table>{header}{rows}</table>"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_html(
    runs: List[dict],
    experiment_name: str,
    metric_keys: List[str] = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
    radar_metric_keys: List[str] = None,
    tag_columns: List[str] = None,
    title: str = None,
) -> str:
    """
    Build HTML report for experiment runs.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("pandas and plotly are required. pip install mlops-wrapper[compare]")

    if metric_keys is None:
        metric_keys = _discover_metric_keys(runs)
    if not metric_keys:
        raise ValueError("no metrics found in runs")

    lower_is_better_keys = lower_is_better_keys or set()
    metric_formats = metric_formats or {}

    if tag_columns is None:
        candidates = ['experiment.phase', 'data.version', 'model.type']
        tag_columns = [t for t in candidates if any(t in r['tags'] for r in runs)]

    if radar_metric_keys is None:
        radar_metric_keys = [k for k in metric_keys if k.startswith('test_')]
        if len(radar_metric_keys) < 3:
            radar_metric_keys = metric_keys[:6]

    display_title = title or f"Run Comparison: {experiment_name}"

    sections = [
        f"<h2>📋 Run Overview ({len(runs)} runs)</h2>\n{_run_overview_table(runs, tag_columns)}",
        f"<h2>📊 Metrics Comparison</h2>\n"
        f"<p><em>🟢 best · 🔴 worst · arrows show latest vs previous</em></p>\n"
        f"{_metrics_comparison_table(runs, metric_keys, lower_is_better_keys, metric_formats, version_mode=False)}",
        f"<h2>📈 Metric Trends</h2>\n{_metric_trend_charts(runs, metric_keys, version_mode=False)}",
    ]

    if len(runs) >= 2 and len(radar_metric_keys) >= 3:
        sections.append(
            f"<h2>🎯 Radar Comparison</h2>\n"
            f"<p><em>Normalized 0–1, higher = better</em></p>\n"
            f"{_radar_chart(runs, radar_metric_keys, lower_is_better_keys, version_mode=False)}"
        )

    sections.append(
        f"<h2>⚙️ Parameter Changes</h2>\n"
        f"<p><em>Only params that differ between runs</em></p>\n"
        f"{_params_diff_table(runs)}"
    )

    body = f"<h1>{html_lib.escape(display_title)}</h1>\n" + "\n".join(sections)
    return _wrap_page(display_title, body)


def build_model_version_comparison_html(
    versions: List[dict],
    model_name: str,
    metric_keys: List[str] = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
    radar_metric_keys: List[str] = None,
    tag_columns: List[str] = None,
    title: str = None,
) -> str:
    """
    Build HTML report for registered model versions.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("pandas and plotly are required. pip install mlops-wrapper[compare]")

    if metric_keys is None:
        metric_keys = _discover_metric_keys(versions)
    if not metric_keys:
        raise ValueError("no metrics found in model versions")

    lower_is_better_keys = lower_is_better_keys or set()
    metric_formats = metric_formats or {}

    if tag_columns is None:
        candidates = ['experiment.phase', 'data.version', 'model.type']
        tag_columns = [t for t in candidates if any(t in v['tags'] for v in versions)]

    if radar_metric_keys is None:
        radar_metric_keys = [k for k in metric_keys if k.startswith('test_')]
        if len(radar_metric_keys) < 3:
            radar_metric_keys = metric_keys[:6]

    display_title = title or f"Model Version Comparison: {model_name}"

    sections = [
        f"<h2>📋 Model Versions ({len(versions)} versions)</h2>\n{_version_overview_table(versions, tag_columns)}",
        f"<h2>📊 Metrics Comparison</h2>\n"
        f"<p><em>🟢 best · 🔴 worst · arrows show latest vs previous version</em></p>\n"
        f"{_metrics_comparison_table(versions, metric_keys, lower_is_better_keys, metric_formats, version_mode=True)}",
        f"<h2>📈 Metric Trends Across Versions</h2>\n{_metric_trend_charts(versions, metric_keys, version_mode=True)}",
    ]

    if len(versions) >= 2 and len(radar_metric_keys) >= 3:
        sections.append(
            f"<h2>🎯 Radar Comparison</h2>\n"
            f"<p><em>Normalized 0–1, higher = better</em></p>\n"
            f"{_radar_chart(versions, radar_metric_keys, lower_is_better_keys, version_mode=True)}"
        )

    sections.append(
        f"<h2>⚙️ Parameter Changes Between Versions</h2>\n"
        f"<p><em>Only params that differ</em></p>\n"
        f"{_params_diff_table(versions)}"
    )

    body = f"<h1>{html_lib.escape(display_title)}</h1>\n" + "\n".join(sections)
    return _wrap_page(display_title, body)


# ═══════════════════════════════════════════════════════════════════════════════
# public entry points
# ═══════════════════════════════════════════════════════════════════════════════

def compare_runs(
    experiment_name: str,
    n_runs: int = 5,
    output_path: str = "comparison_report.html",
    tracking_uri: str = None,
    filter_string: str = "",
    only_successful: bool = True,
    metric_keys: List[str] = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
    radar_metric_keys: List[str] = None,
    tag_columns: List[str] = None,
    title: str = None,
) -> Path:
    """
    Fetch last N runs and write comparison HTML.
    """
    print(f"fetching last {n_runs} runs from '{experiment_name}' ...")
    runs = fetch_recent_runs(
        experiment_name=experiment_name,
        n_runs=n_runs,
        tracking_uri=tracking_uri,
        filter_string=filter_string,
        only_successful=only_successful,
    )

    if not runs:
        raise ValueError(f"no runs found for experiment '{experiment_name}'")

    print(f"  found {len(runs)} runs")
    for i, r in enumerate(runs):
        print(f"    {i+1}. {r['run_name']} ({r['start_time'].strftime('%Y-%m-%d %H:%M')})")

    html_content = build_comparison_html(
        runs=runs,
        experiment_name=experiment_name,
        metric_keys=metric_keys,
        lower_is_better_keys=lower_is_better_keys,
        metric_formats=metric_formats,
        radar_metric_keys=radar_metric_keys,
        tag_columns=tag_columns,
        title=title,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"  ✓ comparison report saved: {output_path}")
    return output_path


def compare_model_versions(
    model_name: str,
    n_versions: int = 5,
    output_path: str = "model_version_comparison.html",
    tracking_uri: str = None,
    stages: List[str] = None,
    aliases: List[str] = None,
    metric_keys: List[str] = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
    radar_metric_keys: List[str] = None,
    tag_columns: List[str] = None,
    title: str = None,
) -> Path:
    """
    Fetch last N registered model versions and write comparison HTML.
    """
    print(f"fetching last {n_versions} versions of model '{model_name}' ...")
    versions = fetch_model_versions(
        model_name=model_name,
        n_versions=n_versions,
        tracking_uri=tracking_uri,
        stages=stages,
        aliases=aliases,
    )

    if not versions:
        raise ValueError(f"no versions found for model '{model_name}'")

    print(f"  found {len(versions)} versions")
    for v in versions:
        stage_info = f" [{v['model_stage']}]" if v.get('model_stage') else ""
        alias_info = f" ({', '.join(v['model_aliases'])})" if v.get('model_aliases') else ""
        print(f"    v{v['model_version']}: {v['run_name']}{stage_info}{alias_info}")

    html_content = build_model_version_comparison_html(
        versions=versions,
        model_name=model_name,
        metric_keys=metric_keys,
        lower_is_better_keys=lower_is_better_keys,
        metric_formats=metric_formats,
        radar_metric_keys=radar_metric_keys,
        tag_columns=tag_columns,
        title=title,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"  ✓ comparison report saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# on-disk experiment run loading  (schema-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

# evaluation split folder name → short metric key prefix
_DISK_SPLIT_PREFIX = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}


def _flatten_dict(d: dict, prefix: str = "") -> Dict[str, float]:
    """Recursively flatten a nested dict with dot-separated keys; keeps only numeric leaves."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, f"{key}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def _infer_lower_is_better(key: str) -> bool:
    """
    Best-effort heuristic: return True when a lower value is better.

    Applies to the full metric key.  Override by passing
    ``lower_is_better_keys`` explicitly to the builder functions.
    """
    low = key.lower()
    lower_patterns = (
        "mae", "mse", "rmse", "mape", "mdape", "ape",
        "loss", "error", "bad", "miss", "false_",
        "fp_", "fn_", "latency", "duration",
    )
    return any(p in low for p in lower_patterns)


def _infer_format(key: str) -> str:
    """
    Best-effort heuristic: return a format alias for this metric key.

    Override by passing ``metric_formats`` explicitly to the builder functions.
    """
    low = key.lower()
    if any(p in low for p in ("_pct", "pct_", "percent", "rate",
                               "accuracy", "ape", "mape", "mdape", "within_")):
        return "pct"
    if any(p in low for p in ("count", "_n.", "_n_", "n_", "num_",
                               "_num", "total_", "support", "rows", "jobs")):
        return "int"
    if any(p in low for p in ("mae", "rmse", "revenue", "cost",
                               "price", "salary", "wage", "dollar")):
        return "dollar"
    return "default"


# ─── loading ──────────────────────────────────────────────────────────────────

def load_run_from_disk(
    run_dir,
    splits: List[str] = None,
) -> dict:
    """
    Load a single on-disk experiment run directory into the standard run-dict
    format.

    Walks ``evaluation/`` for split sub-directories and loads every ``*.json``
    file found inside each split.  Metric keys are produced as::

        {split_prefix}_{json_stem}.{flattened_key}

    e.g. ``evaluation/test/metrics.json`` → ``test_metrics.overall.mdape``.

    CSV files discovered under each split are recorded internally for use by
    ``build_disk_comparison_html``.

    Also reads ``metadata/model_metadata.json`` when present.
    """
    import re

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"run directory not found: {run_dir}")

    dir_name = run_dir.name
    run_name = dir_name
    start_time = datetime.now()

    ts_match = re.search(r"(\d{8}_\d{6})", dir_name)
    if ts_match:
        try:
            start_time = datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    params: dict = {}
    tags: dict = {}
    metadata_path = run_dir / "metadata" / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        params = {
            str(k): str(v)
            for k, v in meta.items()
            if isinstance(v, (str, int, float, bool)) and not str(k).startswith("_")
        }
        if "run_name" in meta:
            run_name = str(meta["run_name"])
        for ts_key in ("timestamp", "run_timestamp", "created_at"):
            if ts_key in meta:
                try:
                    start_time = datetime.fromisoformat(str(meta[ts_key]))
                    break
                except (ValueError, TypeError):
                    pass

    eval_root = run_dir / "evaluation"
    metrics: dict = {}
    segment_csvs: dict = {}  # {split_name: {csv_stem: Path}}

    if eval_root.is_dir():
        split_dirs = sorted(d for d in eval_root.iterdir() if d.is_dir())
        for split_dir in split_dirs:
            split_name = split_dir.name
            if splits and split_name not in splits:
                continue
            prefix = _DISK_SPLIT_PREFIX.get(split_name, split_name) + "_"

            for json_path in sorted(split_dir.glob("*.json")):
                stem = json_path.stem
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    for k, v in _flatten_dict(data).items():
                        metrics[f"{prefix}{stem}.{k}"] = v
                except Exception:
                    pass

            for csv_path in sorted(split_dir.glob("*.csv")):
                segment_csvs.setdefault(split_name, {})[csv_path.stem] = csv_path

    return {
        "run_id":        dir_name,
        "run_name":      run_name,
        "start_time":    start_time,
        "end_time":      None,
        "metrics":       metrics,
        "params":        params,
        "tags":          tags,
        "status":        "FINISHED",
        "source_type":   "disk_run",
        "run_dir":       run_dir,
        "_segment_csvs": segment_csvs,
    }


def load_disk_runs(
    runs_root: str,
    n_runs: int = 5,
    glob_pattern: str = "*",
    splits: List[str] = None,
    newest_first: bool = True,
) -> List[dict]:
    """
    Scan ``runs_root`` for experiment run directories and load them.

    Any directory that contains an ``evaluation/`` subfolder is treated as a
    run.  Use ``glob_pattern`` to restrict which directories are matched
    (e.g. ``"catboost_*"``).

    Parameters
    ----------
    runs_root : str
        Root directory to scan.
    n_runs : int
        Maximum number of runs to load (default 5).
    glob_pattern : str
        Glob applied to direct children of ``runs_root``
        (default ``"*"`` — all directories).
    splits : list of str, optional
        Restrict which evaluation splits to load (default: all found).
    newest_first : bool
        Return newest directories first, sorted by name (default True).
    """
    root = Path(runs_root)
    if not root.is_dir():
        raise ValueError(f"runs_root not found: {runs_root}")

    candidates = sorted(
        d for d in root.glob(glob_pattern)
        if d.is_dir() and (d / "evaluation").is_dir()
    )

    if newest_first:
        candidates = list(reversed(candidates))

    candidates = candidates[:n_runs]

    runs = []
    for run_dir in candidates:
        try:
            run = load_run_from_disk(run_dir, splits=splits)
            runs.append(run)
        except Exception as exc:
            print(f"  warning: skipping {run_dir.name}: {exc}")

    return runs


# ─── auto-inference ───────────────────────────────────────────────────────────

def _auto_lower_is_better(metric_keys: List[str]) -> Set[str]:
    return {k for k in metric_keys if _infer_lower_is_better(k)}


def _auto_metric_formats(metric_keys: List[str]) -> Dict[str, str]:
    return {k: _infer_format(k) for k in metric_keys}


# ─── dynamic section builders ─────────────────────────────────────────────────

def _group_keys_by_stem(metric_keys: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Group flat metric keys by their JSON file stem and split.

    Keys follow ``{split_prefix}_{stem}.{base_key}`` (e.g.
    ``test_metrics.overall.mae``).

    Returns ``{stem: {split: [full_keys]}}``.
    """
    _prefix_to_split = {"train_": "train", "val_": "val", "test_": "test"}
    result: Dict[str, Dict[str, List[str]]] = {}
    for key in metric_keys:
        for prefix, split in _prefix_to_split.items():
            if key.startswith(prefix):
                rest = key[len(prefix):]          # "{stem}.{base_key}"
                dot_idx = rest.find(".")
                if dot_idx > 0:
                    stem = rest[:dot_idx]
                    result.setdefault(stem, {}).setdefault(split, []).append(key)
                break
    return result


def _remap_stem_keys(
    runs: List[dict],
    stem: str,
    stem_splits: Dict[str, List[str]],
) -> tuple:
    """
    Return (remapped_runs, remapped_keys) with the stem prefix stripped so
    chart axis/legend labels are clean.

    ``test_metrics.overall.mae`` → ``test_overall.mae``
    """
    _split_prefixes = {"train": "train_", "val": "val_", "test": "test_"}
    key_map: Dict[str, str] = {}
    for split, keys in stem_splits.items():
        sp = _split_prefixes.get(split, split + "_")
        full_prefix = f"{sp}{stem}."
        for key in keys:
            if key.startswith(full_prefix):
                base = key[len(full_prefix):]
                key_map[key] = f"{sp}{base}"

    if not key_map:
        return runs, [k for keys in stem_splits.values() for k in keys]

    remapped = []
    for run in runs:
        new_run = dict(run)
        new_run["metrics"] = {key_map.get(k, k): v for k, v in run["metrics"].items()}
        remapped.append(new_run)

    # preserve insertion order, dedupe
    return remapped, list(dict.fromkeys(key_map.values()))


def _build_json_section(
    runs: List[dict],
    stem: str,
    stem_splits: Dict[str, List[str]],
    lower_is_better_keys: Set[str],
    metric_formats: Dict[str, Any],
) -> str:
    """
    Build the HTML section for one JSON file (identified by ``stem``).

    Renders one comparison table per split that has this file, using the
    clean base metric name as the row label (e.g. ``overall.mae`` rather than
    ``test_metrics.overall.mae``).  Appends a trend chart with all splits
    overlaid.
    """
    _split_prefixes = {"train": "train_", "val": "val_", "test": "test_"}
    split_order = ["test", "val", "train"]

    html = (
        f'<h2>📄 <code>{html_lib.escape(stem)}.json</code></h2>\n'
        f'<p><em>🟢 best · 🔴 worst within each split · '
        f'arrow = latest vs previous run</em></p>\n'
    )

    for split in split_order:
        if split not in stem_splits:
            continue

        keys = stem_splits[split]
        sp = _split_prefixes.get(split, split + "_")
        full_prefix = f"{sp}{stem}."

        # strip "{split_prefix}_{stem}." → display just the base metric name
        key_labels = {
            k: k[len(full_prefix):] if k.startswith(full_prefix) else k
            for k in keys
        }

        html += f'<h3>{html_lib.escape(split.capitalize())} split</h3>\n'
        html += _metrics_comparison_table(
            runs, keys, lower_is_better_keys, metric_formats,
            version_mode=False, key_labels=key_labels,
        )

    # trend chart with clean remapped keys
    remapped_runs, remapped_keys = _remap_stem_keys(runs, stem, stem_splits)
    if remapped_keys:
        html += _metric_trend_charts(remapped_runs, remapped_keys, version_mode=False)

    return html


def _discover_all_segment_csvs(runs: List[dict]) -> List[tuple]:
    """
    Return ``(split, stem, segment_col, metric_cols)`` for every CSV found in
    the evaluation tree across all runs and all splits.

    ``segment_col`` — first non-numeric column (fallback: first column).
    ``metric_cols`` — all remaining numeric columns.
    """
    if pd is None:
        return []

    pairs: Set[tuple] = set()
    for run in runs:
        for split, stems in run.get("_segment_csvs", {}).items():
            for stem in stems:
                pairs.add((split, stem))

    split_order = ["test", "val", "train"]
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (
            split_order.index(p[0]) if p[0] in split_order else len(split_order),
            p[1],
        ),
    )

    result = []
    for split, stem in sorted_pairs:
        sample_df = None
        for run in runs:
            csv_path = run.get("_segment_csvs", {}).get(split, {}).get(stem)
            if csv_path and Path(csv_path).exists():
                try:
                    sample_df = pd.read_csv(csv_path)
                    break
                except Exception:
                    continue
        if sample_df is None or sample_df.empty:
            continue

        numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
        non_numeric = [c for c in sample_df.columns if c not in numeric_cols]
        segment_col = non_numeric[0] if non_numeric else sample_df.columns[0]
        metric_cols = [c for c in numeric_cols if c != segment_col]

        if not metric_cols:
            continue
        result.append((split, stem, segment_col, metric_cols))

    return result


def _segment_comparison_section(
    runs: List[dict],
    stem: str,
    split: str,
    segment_col: str,
    metric_cols: List[str],
    lower_is_better_cols: Optional[Set[str]] = None,
    metric_formats: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build an HTML section comparing per-segment metrics across disk runs.

    Loads ``evaluation/{split}/{stem}.csv`` for each run, pivots on
    ``segment_col``, and renders a highlighted table per metric column.

    Returns an empty string when pandas is unavailable or no CSVs are found.
    """
    if pd is None:
        return ""

    lower_is_better_cols = lower_is_better_cols or set()
    metric_formats = metric_formats or {}

    run_dfs: List[tuple] = []
    for run in runs:
        csv_path = run.get("_segment_csvs", {}).get(split, {}).get(stem)
        if not csv_path or not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            run_dfs.append((run["run_name"], df))
        except Exception:
            continue

    if not run_dfs:
        return ""

    html = (
        f'<h2>📋 <code>{html_lib.escape(stem)}.csv</code>'
        f'<small style="font-weight:normal;color:#666"> — '
        f'{html_lib.escape(split)} split</small></h2>\n'
    )

    run_names = [r["run_name"] for r in runs]

    for metric in metric_cols:
        pivot: Dict[str, Dict[str, Any]] = {}
        seg_order: List[str] = []
        for rn, df in run_dfs:
            if segment_col not in df.columns or metric not in df.columns:
                continue
            for _, row in df.iterrows():
                seg = str(row[segment_col])
                if seg not in pivot:
                    pivot[seg] = {}
                    seg_order.append(seg)
                pivot[seg][rn] = row[metric]

        if not pivot:
            continue

        lower = metric in lower_is_better_cols

        header = f"<tr><th>{html_lib.escape(segment_col)}</th>"
        for rn in run_names:
            label = rn[:22] if len(rn) > 22 else rn
            header += f"<th>{html_lib.escape(label)}</th>"
        header += "</tr>"

        rows_html = ""
        for seg in seg_order:
            cell_vals = {rn: pivot.get(seg, {}).get(rn) for rn in run_names}
            numeric = [v for v in cell_vals.values() if isinstance(v, (int, float))]
            best_val = worst_val = None
            if len(numeric) > 1:
                best_val = min(numeric) if lower else max(numeric)
                worst_val = max(numeric) if lower else min(numeric)

            rows_html += f"<tr><td><b>{html_lib.escape(seg)}</b></td>"
            for rn in run_names:
                v = cell_vals[rn]
                css = ""
                if isinstance(v, (int, float)) and best_val is not None:
                    if v == best_val:
                        css = ' class="best"'
                    elif v == worst_val:
                        css = ' class="worst"'
                display = _format_value(
                    v if isinstance(v, (int, float)) else None, metric, metric_formats
                )
                rows_html += f"<td{css}>{display}</td>"
            rows_html += "</tr>"

        html += f"<h3>{html_lib.escape(metric)}</h3><table>{header}{rows_html}</table>\n"

    return html


# ─── HTML builder ─────────────────────────────────────────────────────────────

def build_disk_comparison_html(
    runs: List[dict],
    title: str = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
) -> str:
    """
    Build an HTML comparison report from on-disk experiment run dicts.

    The report structure is driven entirely by what files exist in the
    ``evaluation/`` tree — no schema or experiment-type configuration needed:

    - One section per ``*.json`` file, with a per-split comparison table
      (rows = base metric names, columns = runs) and a trend chart.
    - One section per ``*.csv`` file per split, with per-metric segment
      tables (rows = segment values, columns = runs).
    - A radar chart when ≥ 3 test-split metrics are available.
    - A parameter-diff table at the end.

    Parameters
    ----------
    runs : list of dict
        Run dicts from ``load_run_from_disk`` / ``load_disk_runs``.
    title : str, optional
        Report title (auto-generated when omitted).
    lower_is_better_keys : set of str, optional
        Override the auto-inferred lower-is-better set.
    metric_formats : dict, optional
        Override the auto-inferred metric formats.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("pandas and plotly are required: pip install mlops-wrapper[compare]")

    all_metric_keys = _discover_metric_keys(runs)

    if lower_is_better_keys is None:
        lower_is_better_keys = _auto_lower_is_better(all_metric_keys)

    if metric_formats is None:
        metric_formats = _auto_metric_formats(all_metric_keys)

    # group all metric keys by which JSON file they came from
    stem_groups = _group_keys_by_stem(all_metric_keys)

    display_title = title or f"Run Comparison ({len(runs)} runs)"
    sections: List[str] = [
        f"<h2>📋 Run Overview ({len(runs)} runs)</h2>\n{_run_overview_table(runs)}",
    ]

    # one section per JSON file, driven by what actually exists on disk
    for stem in sorted(stem_groups):
        sections.append(
            _build_json_section(
                runs, stem, stem_groups[stem],
                lower_is_better_keys, metric_formats,
            )
        )

    # radar: use the first stem that has test-split metrics with ≥ 3 keys
    radar_html = ""
    if len(runs) >= 2:
        for stem in sorted(stem_groups):
            if "test" not in stem_groups[stem]:
                continue
            remapped_runs, remapped_keys = _remap_stem_keys(
                runs, stem, stem_groups[stem]
            )
            test_keys = [k for k in remapped_keys if k.startswith("test_")]
            if len(test_keys) >= 3:
                radar_html = (
                    f"<h2>🎯 Radar Comparison"
                    f"<small style='font-weight:normal;color:#666'>"
                    f" — {html_lib.escape(stem)}.json · test split</small></h2>\n"
                    f"<p><em>Normalized 0–1, higher = better</em></p>\n"
                    f"{_radar_chart(remapped_runs, test_keys, lower_is_better_keys, version_mode=False)}"
                )
                break
    if radar_html:
        sections.append(radar_html)

    # one section per CSV file per split — driven by what actually exists on disk
    for split, stem, seg_col, metric_cols in _discover_all_segment_csvs(runs):
        lower_cols = {m for m in metric_cols if _infer_lower_is_better(m)}
        seg_html = _segment_comparison_section(
            runs,
            stem=stem,
            split=split,
            segment_col=seg_col,
            metric_cols=metric_cols,
            lower_is_better_cols=lower_cols,
            metric_formats=metric_formats,
        )
        if seg_html:
            sections.append(seg_html)

    sections.append(
        f"<h2>⚙️ Parameter Changes</h2>\n"
        f"<p><em>Only params that differ between runs</em></p>\n"
        f"{_params_diff_table(runs)}"
    )

    body = f"<h1>{html_lib.escape(display_title)}</h1>\n" + "\n".join(sections)
    return _wrap_page(display_title, body)


def build_disk_summary_html(
    runs: "List[dict]",
    title: str = None,
    lower_is_better_keys: "Set[str]" = None,
    metric_formats: "Dict[str, Any]" = None,
) -> str:
    """
    Build the summary HTML: Run Overview, per-JSON metric sections, Radar
    chart, and Parameter Changes \u2014 CSV segment tables excluded.

    Use together with build_disk_segments_html to produce two focused files,
    or build_disk_comparison_html to get everything in one file.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("pandas and plotly are required: pip install mlops-wrapper[compare]")

    all_metric_keys = _discover_metric_keys(runs)

    if lower_is_better_keys is None:
        lower_is_better_keys = _auto_lower_is_better(all_metric_keys)
    if metric_formats is None:
        metric_formats = _auto_metric_formats(all_metric_keys)

    stem_groups = _group_keys_by_stem(all_metric_keys)

    display_title = title or f"Run Comparison ({len(runs)} runs)"
    sections: List[str] = [
        f"<h2>\U0001f4cb Run Overview ({len(runs)} runs)</h2>\n{_run_overview_table(runs)}",
    ]

    for stem in sorted(stem_groups):
        sections.append(
            _build_json_section(
                runs, stem, stem_groups[stem],
                lower_is_better_keys, metric_formats,
            )
        )

    radar_html = ""
    if len(runs) >= 2:
        for stem in sorted(stem_groups):
            if "test" not in stem_groups[stem]:
                continue
            remapped_runs, remapped_keys = _remap_stem_keys(runs, stem, stem_groups[stem])
            test_keys = [k for k in remapped_keys if k.startswith("test_")]
            if len(test_keys) >= 3:
                radar_html = (
                    f"<h2>\U0001f3af Radar Comparison"
                    f"<small style='font-weight:normal;color:#666'>"
                    f" \u2014 {html_lib.escape(stem)}.json \u00b7 test split</small></h2>\n"
                    f"<p><em>Normalized 0\u20131, higher = better</em></p>\n"
                    f"{_radar_chart(remapped_runs, test_keys, lower_is_better_keys, version_mode=False)}"
                )
                break
    if radar_html:
        sections.append(radar_html)

    sections.append(
        f"<h2>\u2699\ufe0f Parameter Changes</h2>\n"
        f"<p><em>Only params that differ between runs</em></p>\n"
        f"{_params_diff_table(runs)}"
    )

    body = f"<h1>{html_lib.escape(display_title)}</h1>\n" + "\n".join(sections)
    return _wrap_page(display_title, body)


def build_disk_segments_html(
    runs: "List[dict]",
    title: str = None,
    lower_is_better_keys: "Set[str]" = None,
    metric_formats: "Dict[str, Any]" = None,
) -> "Optional[str]":
    """
    Build the segments HTML: per-CSV segment comparison tables only.

    Returns None when no CSV segment files are found across the runs \u2014 the
    caller should treat a None return as \'nothing to write\'.

    Use together with build_disk_summary_html to produce two focused files,
    or build_disk_comparison_html to get everything in one file.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("pandas and plotly are required: pip install mlops-wrapper[compare]")

    all_metric_keys = _discover_metric_keys(runs)

    if lower_is_better_keys is None:
        lower_is_better_keys = _auto_lower_is_better(all_metric_keys)
    if metric_formats is None:
        metric_formats = _auto_metric_formats(all_metric_keys)

    display_title = title or f"Run Comparison \u2014 Segments ({len(runs)} runs)"
    sections: List[str] = []

    for split, stem, seg_col, metric_cols in _discover_all_segment_csvs(runs):
        lower_cols = {m for m in metric_cols if _infer_lower_is_better(m)}
        seg_html = _segment_comparison_section(
            runs,
            stem=stem,
            split=split,
            segment_col=seg_col,
            metric_cols=metric_cols,
            lower_is_better_cols=lower_cols,
            metric_formats=metric_formats,
        )
        if seg_html:
            sections.append(seg_html)

    if not sections:
        return None

    body = f"<h1>{html_lib.escape(display_title)}</h1>\n" + "\n".join(sections)
    return _wrap_page(display_title, body)


# ─── public entry point ───────────────────────────────────────────────────────

def compare_disk_runs(
    runs_root: str,
    n_runs: int = 5,
    glob_pattern: str = "*",
    splits: List[str] = None,
    output_path: str = "disk_comparison_report.html",
    segments_output_path: str = None,
    lower_is_better_keys: Set[str] = None,
    metric_formats: Dict[str, Any] = None,
    title: str = None,
    newest_first: bool = True,
) -> Path:
    """
    Load on-disk experiment runs and write a comparison HTML report.

    Schema-agnostic: any directory under ``runs_root`` that contains an
    ``evaluation/`` subfolder is treated as a run.  The HTML structure is
    driven entirely by what files exist in the evaluation tree — no
    experiment-type or metric-list configuration required.

    Parameters
    ----------
    runs_root : str
        Directory containing the experiment run folders,
        e.g. ``"outputs/models"``.
    n_runs : int
        Maximum number of runs to compare (default 5).
    glob_pattern : str
        Glob pattern to filter run directories (default ``"*"``).
        Use e.g. ``"catboost_*"`` to compare only CatBoost runs.
    splits : list of str, optional
        Evaluation split folders to load (default: all found).
    output_path : str
        Destination path for the HTML report.
    lower_is_better_keys : set of str, optional
        Override the auto-inferred lower-is-better set.
    metric_formats : dict, optional
        Override the auto-inferred metric formats.
    newest_first : bool
        Sort runs newest-first by directory name (default True).

    Returns
    -------
    Path
        Absolute path to the written HTML report.
    """
    print(f"loading disk runs from '{runs_root}' (pattern='{glob_pattern}') ...")
    runs = load_disk_runs(
        runs_root=runs_root,
        n_runs=n_runs,
        glob_pattern=glob_pattern,
        splits=splits,
        newest_first=newest_first,
    )

    if not runs:
        raise ValueError(f"no runs found in '{runs_root}' matching '{glob_pattern}'")

    print(f"  found {len(runs)} runs")
    for i, r in enumerate(runs):
        print(f"    {i+1}. {r['run_name']} ({r['start_time'].strftime('%Y-%m-%d %H:%M')})")

    if segments_output_path is not None:
        html_content = build_disk_summary_html(
            runs=runs,
            title=title,
            lower_is_better_keys=lower_is_better_keys,
            metric_formats=metric_formats,
        )
    else:
        html_content = build_disk_comparison_html(
            runs=runs,
            title=title,
            lower_is_better_keys=lower_is_better_keys,
            metric_formats=metric_formats,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html_content)
    print(f"  ✓ summary report saved: {out}")

    if segments_output_path is not None:
        seg_html = build_disk_segments_html(
            runs=runs,
            title=title,
            lower_is_better_keys=lower_is_better_keys,
            metric_formats=metric_formats,
        )
        if seg_html:
            seg_out = Path(segments_output_path)
            seg_out.parent.mkdir(parents=True, exist_ok=True)
            with open(seg_out, "w") as f:
                f.write(seg_html)
            print(f"  ✓ segments report saved: {seg_out}")
        else:
            print(f"  ⚠ no CSV segments found, segments report not written")

    return out
