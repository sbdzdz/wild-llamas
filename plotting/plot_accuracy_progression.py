"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_accuracy_progression(ylim=None, individual=False):
    """Create the accuracy progression plot."""
    if individual:
        create_individual_plot(ylim)
    else:
        create_aggregate_plot(ylim)


def create_individual_plot(ylim=None):
    """Create individual plots for each MMLU subset as separate files."""
    df = load_summary_data()
    subsets = sorted(df["dataset"].unique())

    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)

    print(f"Creating individual plots for {len(subsets)} MMLU subsets...")

    for subset in subsets:
        subset_df = df[df["dataset"] == subset].sort_values("step")
        steps = subset_df["step"].tolist()
        current_accuracies = subset_df["eval_model_current"].tolist()
        merged_accuracies = subset_df["eval_model_merged"].tolist()

        plot_accuracy(
            steps,
            current_accuracies,
            merged_accuracies,
            ylim=ylim,
            show_annotations=True,
        )

        plt.title(
            f"LLaMA-3.1-8B-Instruct Accuracy Across Merging Steps: {format_subset_name(subset)}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        filename = create_filename_from_subset(subset)
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    print(f"All individual plots saved to {figures_dir}")


def create_aggregate_plot(ylim=None):
    """Create aggregate plot averaging across all MMLU subsets."""
    df = load_summary_data()

    avg_df = (
        df.groupby("step")
        .agg({"eval_model_current": "mean", "eval_model_merged": "mean"})
        .reset_index()
    )

    steps = avg_df["step"].tolist()
    current_accuracies = avg_df["eval_model_current"].tolist()
    merged_accuracies = avg_df["eval_model_merged"].tolist()

    plot_accuracy(steps, current_accuracies, merged_accuracies, ylim=ylim)

    plt.title(
        "LLaMA-3.1-8B-Instruct Accuracy Across Merging Steps",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "accuracy_progression.png", dpi=300, bbox_inches="tight")
    plt.show()


def format_subset_name(dataset_name):
    """Format MMLU subset name for display."""
    name = dataset_name.replace("lukaemon_mmlu_", "")
    name = name.replace("_", " ")
    return name.title()


def create_filename_from_subset(dataset_name):
    """Create a filename from MMLU subset name."""
    name = dataset_name.replace("lukaemon_mmlu_", "")
    return f"accuracy_progression_{name}.png"


def plot_accuracy(
    steps,
    current_accuracies,
    merged_accuracies,
    ylim=None,
    show_annotations=True,
):
    """Plot accuracy data in a new figure.

    Args:
        steps: List of step numbers
        current_accuracies: List of current model accuracies
        merged_accuracies: List of merged model accuracies
        ylim: Y-axis limits tuple (min, max)
        show_annotations: Whether to show accuracy value annotations
    """
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap("Dark2")
    base_color = cmap(0)
    current_color = cmap(1)
    merged_color = cmap(2)
    marker_size = 50

    plt.scatter(
        [steps[0]],
        [merged_accuracies[0]],
        label="Base Model",
        s=marker_size,
        color=base_color,
        zorder=3,
        edgecolor="none",
    )

    current_valid_indices = ~np.isnan(current_accuracies)
    current_valid_steps = np.array(steps)[current_valid_indices]
    current_valid_accuracies = np.array(current_accuracies)[current_valid_indices]

    if len(current_valid_steps) > 1:
        plt.scatter(
            current_valid_steps[1:],
            current_valid_accuracies[1:],
            label="Current Model",
            s=marker_size,
            color=current_color,
            zorder=0,
            edgecolor="none",
        )

    valid_indices = ~np.isnan(merged_accuracies)
    valid_steps = np.array(steps)[valid_indices]
    valid_merged = np.array(merged_accuracies)[valid_indices]

    plt.plot(
        valid_steps,
        valid_merged,
        "-",
        linewidth=2,
        color=merged_color,
        alpha=0.4,
        zorder=1,
    )
    plt.scatter(
        valid_steps[1:],
        valid_merged[1:],
        label="Merged Model",
        s=marker_size,
        color=merged_color,
        alpha=1.0,
        zorder=2,
        edgecolor="none",
    )

    plt.xlabel("Number of merged models", fontsize=12)
    plt.ylabel("MMLU Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks()

    if ylim is not None:
        plt.ylim(ylim)

    if show_annotations:
        annotation_step = max(1, len(valid_steps) // 10) if len(valid_steps) > 20 else 1

        for i, (step, merged) in enumerate(zip(valid_steps, valid_merged)):
            if i == 0 or i % annotation_step == 0:
                is_base = i == 0
                plt.annotate(
                    f"{merged:.1f}",
                    (step, merged),
                    textcoords="offset points",
                    xytext=(-10, -15) if is_base else (-5, 8),
                    ha="center",
                    fontsize=8,
                    color=base_color if is_base else merged_color,
                    fontweight="bold" if is_base else "normal",
                )


def load_summary_data():
    """Load all summary CSV files and extract accuracies for each MMLU subset."""
    outputs_dir = (Path(__file__) / "../../outputs").resolve()

    df_log = pd.read_csv(outputs_dir / "merge_log.csv")
    merged_models = df_log["model_id"].tolist()

    # Find which steps have merged model evaluations
    merged_model_dir = outputs_dir / "opencompass/merged_model"
    step_dirs = sorted(
        [d for d in merged_model_dir.iterdir() if d.name.startswith("step_")]
    )

    evaluated_steps = set()
    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[1])
        evaluated_steps.add(step_num)

    all_data = []
    for step, model_id in enumerate(merged_models):
        model_name = model_id.replace("/", "--")

        try:
            current_model_dir = outputs_dir / f"opencompass/{model_name}"
            has_current_eval = current_model_dir.exists()

            has_merged_eval = step in evaluated_steps

            if has_current_eval and has_merged_eval:
                current_df = get_individual_accuracies(current_model_dir)
                merged_df = get_individual_accuracies(
                    outputs_dir / f"opencompass/merged_model/step_{step}"
                )
                step_df = pd.merge(
                    current_df[["dataset", "eval_model"]],
                    merged_df[["dataset", "eval_model"]],
                    on="dataset",
                    suffixes=("_current", "_merged"),
                )
            elif has_current_eval and not has_merged_eval:
                current_df = get_individual_accuracies(current_model_dir)
                step_df = current_df[["dataset", "eval_model"]].copy()
                step_df.rename(
                    columns={"eval_model": "eval_model_current"}, inplace=True
                )
                step_df["eval_model_merged"] = float("nan")
            elif not has_current_eval and has_merged_eval:
                merged_df = get_individual_accuracies(
                    outputs_dir / f"opencompass/merged_model/step_{step}"
                )
                step_df = merged_df[["dataset", "eval_model"]].copy()
                step_df.rename(
                    columns={"eval_model": "eval_model_merged"}, inplace=True
                )
                step_df["eval_model_current"] = float("nan")
            else:
                continue

            step_df["step"] = step
            all_data.append(step_df)

        except Exception as e:
            print(f"Warning: Could not process step {step} for model {model_id}: {e}")
            continue

    if not all_data:
        raise RuntimeError("No valid evaluation data found")

    return pd.concat(all_data, ignore_index=True)


def get_individual_accuracies(work_dir):
    """Get individual accuracies for each MMLU subset from a model directory."""
    step_dir = Path(work_dir)
    subdirs = [d for d in step_dir.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one directory in {work_dir}, found {len(subdirs)}"
        )
    summary_dir = subdirs[0] / "summary"
    csv_file = next(summary_dir.glob("*.csv"))
    df = pd.read_csv(csv_file)
    df["eval_model"] = pd.to_numeric(df["eval_model"].replace("-", 0), errors="coerce")
    return df


def main():
    """Main function to load data and create plot."""
    parser = argparse.ArgumentParser(
        description="Plot accuracy progression across model merging steps."
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
        "--individual",
        action="store_true",
        help="Plot each MMLU subset separately in subplots",
    )
    args = parser.parse_args()
    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_accuracy_progression(ylim=ylim, individual=args.individual)


if __name__ == "__main__":
    main()
