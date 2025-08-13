"""Plot accuracy progression across model merging steps."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def create_plot(num_steps=None, ylim=None, individual=False):
    """Create the accuracy progression plot."""
    if individual:
        create_individual_plot(num_steps, ylim)
        return

    # Load data as DataFrame and compute averages using pandas
    df = load_summary_data(num_steps, return_dataframe=True)

    # Compute averages across all subsets for each step using pandas groupby
    avg_df = (
        df.groupby("step")
        .agg({"eval_model_current": "mean", "eval_model_merged": "mean"})
        .reset_index()
    )

    steps = avg_df["step"].tolist()
    current_accuracies = avg_df["eval_model_current"].tolist()
    merged_accuracies = avg_df["eval_model_merged"].tolist()

    plot_accuracy_data(steps, current_accuracies, merged_accuracies, ylim=ylim)

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


def create_individual_plot(num_steps=None, ylim=None):
    """Create individual plots for each MMLU subset as separate files."""
    subset_data = load_summary_data(num_steps)
    subsets = sorted(subset_data.keys())

    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)

    print(f"Creating individual plots for {len(subsets)} MMLU subsets...")

    for subset in subsets:
        data = subset_data[subset]
        steps = data["steps"]
        current_accuracies = data["current_model"]
        merged_accuracies = data["merged_model"]

        # Create individual plot with identical formatting to averaged plot
        plot_accuracy_data(
            steps,
            current_accuracies,
            merged_accuracies,
            ylim=ylim,
            show_annotations=True,
        )

        # Add title specific to this subset
        plt.title(
            f"LLaMA-3.1-8B-Instruct Accuracy Across Merging Steps: {format_subset_name(subset)}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save to individual file
        filename = create_filename_from_subset(subset)
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

        print(f"Saved: {filename}")

    print(f"All individual plots saved to {figures_dir}")


def format_subset_name(dataset_name):
    """Format MMLU subset name for display."""
    name = dataset_name.replace("lukaemon_mmlu_", "")
    name = name.replace("_", " ")
    return name.title()


def create_filename_from_subset(dataset_name):
    """Create a filename from MMLU subset name."""
    name = dataset_name.replace("lukaemon_mmlu_", "")
    return f"accuracy_progression_{name}.png"


def plot_accuracy_data(
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

    # Base model
    plt.scatter(
        [steps[0]],
        [merged_accuracies[0]],
        label="Base Model",
        s=marker_size,
        color=base_color,
        zorder=3,
        edgecolor="none",
    )

    # Current model
    plt.scatter(
        steps[1:],
        current_accuracies[1:],
        label="Current Model",
        s=marker_size,
        color=current_color,
        zorder=0,
        edgecolor="none",
    )

    # Merged model
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
    plt.legend(fontsize=11)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(steps)

    if ylim is not None:
        plt.ylim(ylim)

    # Add annotations if requested
    if show_annotations:
        for i, (step, current, merged) in enumerate(
            zip(steps, current_accuracies, merged_accuracies)
        ):
            if i == 0:
                plt.annotate(
                    f"{merged:.1f}",
                    (step, merged),
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


def load_summary_data(num_steps=None, return_dataframe=False):
    """Load all summary CSV files and extract accuracies for each MMLU subset using pandas operations."""
    outputs_dir = (Path(__file__) / "../../outputs").resolve()

    df_log = pd.read_csv(outputs_dir / "merge_log.csv")
    merged_models = df_log["model_id"].tolist()

    merged_model_dir = outputs_dir / "opencompass/merged_model"
    step_dirs = sorted(
        [d for d in merged_model_dir.iterdir() if d.name.startswith("step_")]
    )

    if num_steps is not None:
        step_dirs = step_dirs[:num_steps]

    min_length = min(len(step_dirs), len(merged_models))
    step_dirs = step_dirs[:min_length]
    merged_models = merged_models[:min_length]

    print(f"Found {len(step_dirs)} steps")

    all_data = []

    for step in range(len(step_dirs)):
        model_name = merged_models[step].replace("/", "--")
        current_model_dir = outputs_dir / f"opencompass/{model_name}"
        current_df = get_individual_accuracies(current_model_dir)

        merged_model_step_dir = outputs_dir / f"opencompass/merged_model/step_{step}"
        merged_df = get_individual_accuracies(merged_model_step_dir)

        # Merge current and merged dataframes on dataset
        step_df = pd.merge(
            current_df[["dataset", "eval_model"]],
            merged_df[["dataset", "eval_model"]],
            on="dataset",
            suffixes=("_current", "_merged"),
        )
        step_df["step"] = step
        all_data.append(step_df)

    # Combine all steps into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    if return_dataframe:
        return combined_df

    # Convert to the expected nested dictionary format for backward compatibility
    subset_data = {}
    for dataset in combined_df["dataset"].unique():
        dataset_df = combined_df[combined_df["dataset"] == dataset].sort_values("step")
        subset_data[dataset] = {
            "steps": dataset_df["step"].tolist(),
            "current_model": dataset_df["eval_model_current"].tolist(),
            "merged_model": dataset_df["eval_model_merged"].tolist(),
        }

    return subset_data


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
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Plot each MMLU subset separately in subplots",
    )
    args = parser.parse_args()
    ylim = tuple(args.ylim) if args.ylim is not None else None
    create_plot(num_steps=args.num_steps, ylim=ylim, individual=args.individual)


if __name__ == "__main__":
    main()
