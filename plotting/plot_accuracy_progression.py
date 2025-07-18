"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
    column_name = f"{model_type}_model"
    
    if df is None or column_name not in df.columns:
        return None
    
    df[column_name] = pd.to_numeric(df[column_name].replace("-", 0), errors="coerce")
    return df[column_name].mean()


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


def create_plot(step_data):
    """Create the accuracy progression plot."""
    steps = sorted(step_data.keys())
    current_accuracies = [step_data[step]["current_model"] for step in steps]
    merged_accuracies = [step_data[step]["merged_model"] for step in steps]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        steps,
        current_accuracies,
        label="Current Model",
        s=50,
        color="#1f77b4",
        zorder=5,
    )
    plt.plot(
        steps,
        merged_accuracies,
        "s-",
        label="Merged Model",
        linewidth=2,
        markersize=6,
        color="#ff7f0e",
    )

    plt.xlabel("Number of merged models", fontsize=12)
    plt.ylabel("Average Accuracy (%) over MMLU and CMMLU", fontsize=12)
    plt.title(
        "LLaMA-3.1-8B-Instruct Accuracy Across Merging Steps",
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
            xytext=(5, -15),
            ha="center",
            fontsize=8,
            color="#1f77b4",
        )
        plt.annotate(
            f"{merged:.1f}",
            (step, merged),
            textcoords="offset points",
            xytext=(-5, 10),
            ha="center",
            fontsize=8,
            color="#ff7f0e",
        )

    plt.tight_layout()
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "accuracy_progression.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main function to load data and create plot."""
    step_data = load_summary_data()
    print(f"\nFound data for {len(step_data)} steps")
    create_plot(step_data)


if __name__ == "__main__":
    main()
