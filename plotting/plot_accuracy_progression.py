"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib.patches import FancyBboxPatch

DATASETS = {
    "mmlu": {
        "prefix": "lukaemon_mmlu_",
        "display_name": "MMLU",
    },
    "mmlu_pro": {
        "prefix": "mmlu_pro_",
        "display_name": "MMLU Pro",
    },
    "gpqa_diamond": {
        "prefix": "GPQA_diamond",
        "display_name": "GPQA Diamond",
    },
    "math500": {
        "prefix": "math-500",
        "display_name": "Math-500",
    },
}


def plot_accuracy_per_dataset(output_dir, ylim=None, plot_current=False):
    """Create separate plots for each dataset."""
    df = load_summary_data(output_dir)
    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)

    current_models_stats = None
    if plot_current:
        current_models_stats = load_current_models_stats(output_dir)

    for dataset_info in DATASETS.values():
        dataset_df = df[df["dataset"].str.startswith(dataset_info["prefix"])]

        if dataset_df.empty:
            print(f"No data found for {dataset_info['display_name']}")
            continue

        steps, accuracies, stds = compute_step_series(dataset_df)

        plot_accuracy(steps, accuracies, stds, ylim=ylim)

        if plot_current and current_models_stats is not None:
            overlay_current_checkpoints(
                category_info=dataset_info,
                current_models_stats=current_models_stats,
                x_value=steps[0],
            )
            plt.legend(fontsize=11)

        plt.title(
            f"{dataset_info['display_name']} Accuracy",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        filename = (
            f"accuracy_{dataset_info['display_name'].replace(' ', '_').lower()}.png"
        )
        output_path = figures_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path.relative_to(figures_dir.parent)}")


def compute_step_series(dataset_df):
    """Compute step-wise arrays of steps, means, stds for a dataset df."""
    steps = []
    means = []
    stds = []
    for step in sorted(dataset_df["step"].unique()):
        step_data = dataset_df[dataset_df["step"] == step]
        mean_accuracy, std_accuracy = compute_run_aggregate_mean_std(step_data)
        steps.append(step)
        means.append(mean_accuracy)
        stds.append(std_accuracy)
    return steps, means, stds


def compute_run_aggregate_mean_std(df):
    """Compute mean and std over per-run means for an accuracy dataframe.

    Groups by run, averages per run, then returns (mean_of_runs, std_of_runs).
    """
    run_averages = df.groupby("run")["accuracy"].mean()
    mean_accuracy = run_averages.mean()
    std_accuracy = run_averages.std() if len(run_averages) > 1 else 0.0
    return float(mean_accuracy), float(std_accuracy)


def overlay_current_checkpoints(category_info, current_models_stats, x_value):
    """Overlay orange markers and std rectangles for current checkpoints on the plot."""
    prefix = category_info["prefix"]
    entries = current_models_stats.get(prefix, [])
    if not entries:
        return

    ax = plt.gca()
    orange = "#ff7f0e"
    marker_size = 35

    rect_width = 0.6
    rounding_size = rect_width * 0.5

    first = True
    for entry in entries:
        mean = entry["mean"]
        std = entry["std"]

        plt.scatter(
            [x_value],
            [mean],
            s=marker_size,
            color=orange,
            alpha=0.9,
            zorder=3,
            edgecolor="none",
            label="Fine-tuned checkpoints" if first else None,
        )
        first = False

        if std > 0:
            x0 = x_value - rect_width / 2.0
            y0 = mean - std
            rect = FancyBboxPatch(
                (x0, y0),
                rect_width,
                std * 2.0,
                boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                linewidth=0,
                facecolor=orange,
                edgecolor=None,
                alpha=0.2,
                zorder=2,
            )
            ax.add_patch(rect)


def load_current_models_stats(output_dir):
    """Load fine-tuned model stats per category from merge_log and model outputs.

    Returns a dict keyed by category prefix mapping to a list of dicts with
    keys {"mean", "std"} for each model (excluding the base model).
    """
    opencompass_root = Path(output_dir)
    merge_log_path = opencompass_root / "merge_log.csv"
    if not merge_log_path.exists():
        raise RuntimeError(f"merge_log.csv not found at {merge_log_path}")

    df_log = pd.read_csv(merge_log_path)
    if "model_id" not in df_log.columns:
        raise RuntimeError("merge_log.csv must contain a 'model_id' column")

    model_ids = [m for m in df_log["model_id"].tolist() if isinstance(m, str)]
    if len(model_ids) <= 1:
        return {}
    model_ids = model_ids[1:]

    models_root = opencompass_root.parent / "models"

    stats_by_category = {info["prefix"]: [] for info in DATASETS.values()}

    for model_id in model_ids:
        model_dir = models_root / model_id.replace("/", "--")
        df = get_individual_accuracies(model_dir)

        for category_info in DATASETS.values():
            prefix = category_info["prefix"]
            cat_df = df[df["dataset"].str.startswith(prefix)]

            if cat_df.empty:
                continue

            mean_accuracy, std_accuracy = compute_run_aggregate_mean_std(cat_df)
            num_runs = cat_df["run"].nunique()
            print(
                f"[plot-current] model={model_id} dataset={prefix} runs={num_runs} "
                f"mean={mean_accuracy:.3f} std={std_accuracy:.3f}"
            )
            stats_by_category[prefix].append(
                {"mean": mean_accuracy, "std": std_accuracy}
            )

    return stats_by_category


def plot_accuracy(
    steps,
    accuracies,
    stds=None,
    ylim=None,
    show_annotations=True,
):
    """Plot accuracy data for a single category."""
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap("Dark2")
    base_color = cmap(0)
    merged_color = cmap(2)
    marker_size = 50

    plt.scatter(
        [steps[0]],
        [accuracies[0]],
        label="Base Model",
        s=marker_size,
        color=base_color,
        zorder=3,
        edgecolor="none",
    )

    valid_indices = ~np.isnan(accuracies)
    valid_steps = np.array(steps)[valid_indices]
    valid_accuracies = np.array(accuracies)[valid_indices]

    if stds is not None:
        valid_stds = np.array(stds)[valid_indices]
    else:
        valid_stds = np.zeros_like(valid_accuracies)

    plt.plot(
        valid_steps,
        valid_accuracies,
        "-",
        linewidth=2,
        color=merged_color,
        alpha=0.8,
        zorder=2,
    )

    plt.fill_between(
        valid_steps,
        valid_accuracies - valid_stds,
        valid_accuracies + valid_stds,
        color=merged_color,
        alpha=0.2,
        zorder=1,
        linewidth=0,
    )

    if len(valid_steps) > 1:
        plt.scatter(
            valid_steps[1:],
            valid_accuracies[1:],
            label="Merged Model",
            s=marker_size,
            color=merged_color,
            alpha=1.0,
            zorder=3,
            edgecolor="none",
        )

    plt.xlabel("Number of merged models", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks()

    if ylim is not None:
        plt.ylim(ylim)

    if show_annotations:
        annotation_step = max(1, len(valid_steps) // 10) if len(valid_steps) > 20 else 1

        for i, (step, acc) in enumerate(zip(valid_steps, valid_accuracies)):
            if i == 0 or i % annotation_step == 0:
                is_base = i == 0
                plt.annotate(
                    f"{acc:.1f}",
                    (step, acc),
                    textcoords="offset points",
                    xytext=(-10, -15) if is_base else (-5, 8),
                    ha="center",
                    fontsize=8,
                    color=base_color if is_base else merged_color,
                    fontweight="bold" if is_base else "normal",
                )


def load_summary_data(output_dir):
    """Load all summary CSV files and extract accuracies for each dataset.

    Args:
        output_dir: Path to the OpenCompass output directory containing evaluation results
    """
    opencompass_root = Path(output_dir)
    merged_model_dir = opencompass_root / "merged_model"
    step_dirs = sorted(
        [d for d in merged_model_dir.iterdir() if d.name.startswith("step_")]
    )

    all_data = []

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[1])

        try:
            step_df = get_individual_accuracies(step_dir)
            step_df["step"] = step_num
            all_data.append(step_df)
        except Exception as e:
            print(f"Warning: Could not process step {step_num}: {e}")
            continue

    if not all_data:
        raise RuntimeError("No valid evaluation data found")

    return pd.concat(all_data, ignore_index=True)


def get_individual_accuracies(work_dir):
    """Get individual accuracies for each dataset from a model directory.

    Returns raw run data to allow proper statistical aggregation at the category level.
    """
    step_dir = Path(work_dir)
    all_run_data = []

    subdirs = [d for d in step_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        summary_dir = subdir / "summary"
        if not summary_dir.exists():
            continue
        csv_files = list(summary_dir.glob("*.csv"))
        if not csv_files:
            continue
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)
        df = df.rename(columns={"eval_model": "accuracy"})
        df["accuracy"] = pd.to_numeric(df["accuracy"].replace("-", 0), errors="coerce")
        df = df[["dataset", "accuracy"]]
        df["run"] = subdir.name
        all_run_data.append(df)

    if not all_run_data:
        raise RuntimeError(f"No valid evaluation data found in {work_dir}")

    combined_df = pd.concat(all_run_data, ignore_index=True)
    return combined_df[["dataset", "accuracy", "run"]]


def main():
    """Main function to load data and create plot."""
    parser = argparse.ArgumentParser(
        description="Plot accuracy progression across model merging steps."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/opencompass",
        help="Path to the OpenCompass output directory containing evaluation results (default: outputs/opencompass)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Set y-axis limits, e.g. --ylim 0 100",
    )
    parser.add_argument(
        "--plot-current",
        action="store_true",
        help="Also plot individual fine-tuned checkpoints from outputs/models",
    )
    args = parser.parse_args()
    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_accuracy_per_dataset(
        output_dir=args.output_dir, ylim=ylim, plot_current=args.plot_current
    )


if __name__ == "__main__":
    main()
