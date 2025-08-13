"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def create_plot(num_steps=None, ylim=None):
    """Create the accuracy progression plot."""
    step_data = load_summary_data(num_steps)
    steps = sorted(step_data.keys())
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
                xytext=(-5, 8),
                ha="center",
                fontsize=8,
                color=merged_color,
            )

    plt.tight_layout()
    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "accuracy_progression.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_summary_data(num_steps=None):
    """Load all summary CSV files and extract average accuracies."""
    outputs_dir = (Path(__file__) / "../../outputs").resolve()
    step_data = {}

    df_log = pd.read_csv(outputs_dir / "merge_log.csv")
    merged_models = df_log["model_id"].tolist()

    merged_model_dir = outputs_dir / "opencompass/merged_model"
    step_dirs = sorted(
        [d for d in merged_model_dir.iterdir() if d.name.startswith("step_")]
    )

    if num_steps is not None:
        step_dirs = step_dirs[:num_steps]

    print(f"Found {len(step_dirs)} steps")

    for step in range(len(step_dirs)):
        model_name = merged_models[step].replace("/", "--")
        current_model_dir = outputs_dir / f"opencompass/{model_name}"
        current_avg_acc = get_average_accuracy(current_model_dir)

        merged_model_step_dir = outputs_dir / f"opencompass/merged_model/step_{step}"
        merged_avg_acc = get_average_accuracy(merged_model_step_dir)

        step_data[step] = {
            "current_model": current_avg_acc or 0.0,
            "merged_model": merged_avg_acc or 0.0,
        }

    return step_data


def get_average_accuracy(work_dir):
    """Get average accuracy from a model directory with timestamp subdirectories."""
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
    return df["eval_model"].mean()


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
    ylim = tuple(args.ylim) if args.ylim is not None else None
    create_plot(num_steps=args.num_steps, ylim=ylim)


if __name__ == "__main__":
    main()
