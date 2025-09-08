"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

DATASET_CATEGORIES = {
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


def plot_accuracy_progression(ylim=None):
    """Create the accuracy progression plot."""
    create_category_plots(ylim)


def create_category_plots(ylim=None):
    """Create separate plots for each dataset category."""
    df = load_summary_data()
    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)

    for category_info in DATASET_CATEGORIES.values():
        if category_info["prefix"] in ["GPQA_diamond", "math-500"]:
            category_df = df[df["dataset"] == category_info["prefix"]]
        else:
            category_df = df[df["dataset"].str.startswith(category_info["prefix"])]

        if category_df.empty:
            print(f"No data found for {category_info['display_name']}")
            continue

        avg_df = category_df.groupby("step").agg({"accuracy": "mean"}).reset_index()

        steps = avg_df["step"].tolist()
        accuracies = avg_df["accuracy"].tolist()

        plot_single_category_accuracy(steps, accuracies, ylim=ylim)

        plt.title(
            f"LLaMA-3.1-8B-Instruct {category_info['display_name']} Accuracy Across Merging Steps",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        filename = (
            f"accuracy_{category_info['display_name'].replace(' ', '_').lower()}.png"
        )
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    print(f"All category plots saved to {figures_dir}")


def plot_single_category_accuracy(
    steps,
    accuracies,
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

    plt.plot(
        valid_steps,
        valid_accuracies,
        "-",
        linewidth=2,
        color=merged_color,
        alpha=0.4,
        zorder=1,
    )

    if len(valid_steps) > 1:
        plt.scatter(
            valid_steps[1:],
            valid_accuracies[1:],
            label="Merged Model",
            s=marker_size,
            color=merged_color,
            alpha=1.0,
            zorder=2,
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


def load_summary_data():
    """Load all summary CSV files and extract accuracies for each dataset."""
    outputs_dir = (Path(__file__) / "../../outputs").resolve()
    merged_model_dir = outputs_dir / "opencompass/merged_model"
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
    """Get individual accuracies for each dataset from a model directory."""
    step_dir = Path(work_dir)
    subdirs = [d for d in step_dir.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one directory in {work_dir}, found {len(subdirs)}"
        )
    summary_dir = subdirs[0] / "summary"
    csv_file = next(summary_dir.glob("*.csv"))
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"eval_model": "accuracy"})
    df["accuracy"] = pd.to_numeric(df["accuracy"].replace("-", 0), errors="coerce")
    return df[["dataset", "accuracy"]]


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
    args = parser.parse_args()
    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_accuracy_progression(ylim=ylim)


if __name__ == "__main__":
    main()
