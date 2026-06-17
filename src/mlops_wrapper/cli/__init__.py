"""mlops_wrapper CLI entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="mlops",
        description="mlops_wrapper CLI tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── compare ───────────────────────────────────────────────────────────
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare MLflow runs or model versions and generate HTML reports",
    )
    compare_sub = compare_parser.add_subparsers(dest="compare_command")

    # mlops compare runs
    runs_parser = compare_sub.add_parser("runs", help="Compare recent experiment runs")
    runs_parser.add_argument("experiment", help="MLflow experiment name")
    runs_parser.add_argument("-n", "--n-runs", type=int, default=5, help="Number of runs to compare (default: 5)")
    runs_parser.add_argument("-o", "--output", default="comparison_report.html", help="Output HTML path")
    runs_parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    runs_parser.add_argument("--filter", default="", help="MLflow filter string")
    runs_parser.add_argument("--all-runs", action="store_true", help="Include non-successful runs")
    runs_parser.add_argument("--metrics", nargs="*", default=None, help="Specific metric keys to include")
    runs_parser.add_argument("--lower-is-better", nargs="*", default=None, help="Metrics where lower is better")
    runs_parser.add_argument("--title", default=None, help="Report title")

    # mlops compare models
    models_parser = compare_sub.add_parser("models", help="Compare registered model versions")
    models_parser.add_argument("model", help="Registered model name")
    models_parser.add_argument("-n", "--n-versions", type=int, default=5, help="Number of versions to compare (default: 5)")
    models_parser.add_argument("-o", "--output", default="model_version_comparison.html", help="Output HTML path")
    models_parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    models_parser.add_argument("--stages", nargs="*", default=None, help="Filter by stages")
    models_parser.add_argument("--aliases", nargs="*", default=None, help="Filter by aliases")
    models_parser.add_argument("--metrics", nargs="*", default=None, help="Specific metric keys to include")
    models_parser.add_argument("--lower-is-better", nargs="*", default=None, help="Metrics where lower is better")
    models_parser.add_argument("--title", default=None, help="Report title")

    # mlops compare disk
    disk_parser = compare_sub.add_parser("disk", help="Compare on-disk experiment run directories")
    disk_parser.add_argument("runs_root", help="Root directory containing run folders")
    disk_parser.add_argument("-n", "--n-runs", type=int, default=5, help="Number of runs to compare (default: 5)")
    disk_parser.add_argument("-o", "--output", default="disk_comparison_report.html", help="Output HTML path")
    disk_parser.add_argument("--segments-output", default=None, help="Separate output path for segment tables")
    disk_parser.add_argument("--glob", default="*", help="Glob pattern to filter run directories")
    disk_parser.add_argument("--splits", nargs="*", default=None, help="Evaluation splits to load")
    disk_parser.add_argument("--title", default=None, help="Report title")
    disk_parser.add_argument("--oldest-first", action="store_true", help="Sort runs oldest-first")

    # ── parse ─────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compare":
        _handle_compare(args, compare_parser)


def _handle_compare(args, parser):
    if args.compare_command is None:
        parser.print_help()
        sys.exit(1)

    from ..compare import compare_runs, compare_model_versions, compare_disk_runs

    if args.compare_command == "runs":
        lower_keys = set(args.lower_is_better) if args.lower_is_better else None
        compare_runs(
            experiment_name=args.experiment,
            n_runs=args.n_runs,
            output_path=args.output,
            tracking_uri=args.tracking_uri,
            filter_string=args.filter,
            only_successful=not args.all_runs,
            metric_keys=args.metrics,
            lower_is_better_keys=lower_keys,
            title=args.title,
        )

    elif args.compare_command == "models":
        lower_keys = set(args.lower_is_better) if args.lower_is_better else None
        compare_model_versions(
            model_name=args.model,
            n_versions=args.n_versions,
            output_path=args.output,
            tracking_uri=args.tracking_uri,
            stages=args.stages,
            aliases=args.aliases,
            metric_keys=args.metrics,
            lower_is_better_keys=lower_keys,
            title=args.title,
        )

    elif args.compare_command == "disk":
        compare_disk_runs(
            runs_root=args.runs_root,
            n_runs=args.n_runs,
            glob_pattern=args.glob,
            splits=args.splits,
            output_path=args.output,
            segments_output_path=args.segments_output,
            title=args.title,
            newest_first=not args.oldest_first,
        )


if __name__ == "__main__":
    main()
