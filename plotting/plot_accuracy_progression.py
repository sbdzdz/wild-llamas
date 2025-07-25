"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def get_avg_acc(step_dir, model_type):
    """Get average accuracy for a specific model type from summary CSV."""
    model_step_dir = step_dir / model_type

    if not model_step_dir.exists():
        return None

    timestamp_dir = sorted(model_step_dir.iterdir(), reverse=True)[0]
    summary_dir = timestamp_dir / "summary"
    csv_files = list(summary_dir.glob("*.csv"))

    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])
    df["eval_model"] = pd.to_numeric(df["eval_model"].replace("-", 0), errors="coerce")
    return df["eval_model"].mean()


def load_summary_data():
    """Load all summary CSV files and extract average accuracies."""
    outputs_dir = Path("outputs")
    step_data = {}

    step_dirs = sorted([d for d in outputs_dir.iterdir() if d.name.startswith("step_")])

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[1])

        current_avg_acc = get_avg_acc(step_dir, "current")
        merged_avg_acc = get_avg_acc(step_dir, "merged")
        if step_num == 0:
            merged_avg_acc = current_avg_acc

        current_avg_acc = current_avg_acc or 0.0
        merged_avg_acc = merged_avg_acc or 0.0

        step_data[step_num] = {
            "current_model": current_avg_acc,
            "merged_model": merged_avg_acc,
        }

    return step_data


def create_plot(step_data, num_steps=None, ylim=None):
    """Create the accuracy progression plot."""
    steps = sorted(step_data.keys())
    if num_steps is not None:
        steps = steps[:num_steps]
    current_accuracies = [step_data[step]["current_model"] for step in steps]
    merged_accuracies = [step_data[step]["merged_model"] for step in steps]

    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap("Dark2")
    base_color = cmap(0)
    current_color = cmap(1)
    merged_color = cmap(2)
    marker_size = 50
    plt.scatter(
        [steps[0]],
        [current_accuracies[0]],
        label="Base Model",
        s=marker_size,
        color=base_color,
        zorder=3,
        edgecolor="none",
    )
    plt.scatter(
        steps[1:],
        current_accuracies[1:],
        label="Current Model",
        s=marker_size,
        color=current_color,
        zorder=0,
        edgecolor="none",
    )
    plt.plot(
        steps,
        merged_accuracies,
        "-",
        linewidth=2,
        color=merged_color,
        alpha=0.4,
        zorder=1,
    )
    plt.scatter(
        steps[1:],
        merged_accuracies[1:],
        label="Merged Model",
        s=marker_size,
        color=merged_color,
        alpha=1.0,
        zorder=2,
        edgecolor="none",
    )

    plt.xlabel("Number of merged models", fontsize=12)
    plt.ylabel("MMLU Accuracy", fontsize=12)
    plt.title(
        "LLaMA-3.1-8B-Instruct Accuracy Across Merging Steps",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", alpha=0.3)

    plt.xticks(steps)

    if ylim is not None:
        plt.ylim(ylim)

    for i, (step, current, merged) in enumerate(
        zip(steps, current_accuracies, merged_accuracies)
    ):
        if i == 0:
            plt.annotate(
                f"{current:.1f}",
                (step, current),
                textcoords="offset points",
                xytext=(-10, -15),
                ha="center",
                fontsize=8,
                color=base_color,
                fontweight="bold",
            )
        else:
            plt.annotate(
                f"{merged:.1f}",
                (step, merged),
                textcoords="offset points",
                xytext=(-5, 10),
                ha="center",
                fontsize=8,
                color=merged_color,
            )

    plt.tight_layout()
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "accuracy_progression.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main function to load data and create plot."""
    parser = argparse.ArgumentParser(
        description="Plot accuracy progression across model merging steps."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to plot (default: all)",
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
    step_data = load_summary_data()
    print(f"\nFound data for {len(step_data)} steps")
    ylim = tuple(args.ylim) if args.ylim is not None else None
    create_plot(step_data, num_steps=args.num_steps, ylim=ylim)


if __name__ == "__main__":
    main()
