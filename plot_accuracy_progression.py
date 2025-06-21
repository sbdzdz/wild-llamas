"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_summary_data():
    """Load all summary CSV files and extract average accuracies."""
    outputs_dir = Path("outputs")
    step_data = {}

    step_dirs = sorted([d for d in outputs_dir.iterdir() if d.name.startswith("step_")])

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[1])
        timestamp_dir = list(step_dir.iterdir())[0]
        summary_dir = timestamp_dir / "summary"
        csv_file = list(summary_dir.glob("*.csv"))[0]

        df = pd.read_csv(csv_file)

        df["current_model"] = pd.to_numeric(
            df["current_model"].replace("-", 0), errors="coerce"
        )
        df["merged_model"] = pd.to_numeric(
            df["merged_model"].replace("-", 0), errors="coerce"
        )

        current_avg = df["current_model"].mean()
        merged_avg = df["merged_model"].mean()

        step_data[step_num] = {
            "current_model": current_avg,
            "merged_model": merged_avg,
        }

        print(
            f"Step {step_num}: current_model={current_avg:.2f}, merged_model={merged_avg:.2f}"
        )

    return step_data


def create_plot(step_data):
    """Create the accuracy progression plot."""
    steps = sorted(step_data.keys())
    current_accuracies = [step_data[step]["current_model"] for step in steps]
    merged_accuracies = [step_data[step]["merged_model"] for step in steps]

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        current_accuracies,
        "o-",
        label="Current Model",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        steps, merged_accuracies, "s-", label="Merged Model", linewidth=2, markersize=6
    )

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.title(
        "Model Accuracy Progression Across Merging Steps",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.xticks(steps)

    for step, current, merged in zip(steps, current_accuracies, merged_accuracies):
        plt.annotate(
            f"{current:.1f}",
            (step, current),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
        plt.annotate(
            f"{merged:.1f}",
            (step, merged),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("accuracy_progression.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSummary Statistics:")
    print(f"Steps processed: {len(steps)}")
    print(
        f"Current model accuracy range: {min(current_accuracies):.2f} - {max(current_accuracies):.2f}"
    )
    print(
        f"Merged model accuracy range: {min(merged_accuracies):.2f} - {max(merged_accuracies):.2f}"
    )
    print(
        f"Average improvement: {np.mean(merged_accuracies) - np.mean(current_accuracies):.2f}"
    )


def main():
    """Main function to load data and create plot."""
    print("Loading summary data from outputs directory...")
    step_data = load_summary_data()
    print(f"\nFound data for {len(step_data)} steps")
    create_plot(step_data)


if __name__ == "__main__":
    main()
