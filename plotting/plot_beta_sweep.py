"""Plot beta sweep results across model merging steps.

Shows accuracy progression for different beta values, with each beta
plotted as a separate line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "gsm8k": {
        "prefix": "gsm8k-",
        "display_name": "GSM8K",
    },
}


def plot_beta_sweep_per_dataset(
    base_dir, experiment_name, ylim=None, dataset_type="selection"
):
    """Create separate plots for each dataset showing all beta values.

    Args:
        base_dir: Base directory containing beta sweep results
        experiment_name: Name of the experiment (e.g., "ema_holdout")
        ylim: Optional y-axis limits as tuple (min, max)
        dataset_type: Either "selection" or "validation"
    """
    base_dir = Path(base_dir)
    figures_dir = (Path(__file__) / "../../figures").resolve()
    figures_dir.mkdir(exist_ok=True)

    # Load data for all beta values
    beta_data = load_all_beta_data(base_dir, experiment_name, dataset_type)

    if not beta_data:
        print(f"No data found for experiment: {experiment_name}")
        return

    # Create plot for each dataset
    for dataset_info in DATASETS.values():
        prefix = dataset_info["prefix"]
        display_name = dataset_info["display_name"]

        # Check if any beta has data for this dataset
        has_data = any(
            prefix in beta_dict["data"] for beta_dict in beta_data.values()
        )
        if not has_data:
            print(f"No data found for {display_name}")
            continue

        plot_beta_comparison(
            beta_data=beta_data,
            dataset_prefix=prefix,
            dataset_name=display_name,
            ylim=ylim,
        )

        plt.tight_layout()
        dataset_filename = display_name.replace(" ", "_").lower()
        dataset_suffix = "validation" if dataset_type == "validation" else "selection"
        filename = f"{experiment_name}_{dataset_filename}_{dataset_suffix}.png"
        output_path = figures_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path.relative_to(figures_dir.parent)}")


def load_all_beta_data(base_dir, experiment_name, dataset_type="selection"):
    """Load accuracy data for all beta values.

    Returns:
        dict: Mapping from beta (float) to dict containing:
            - "data": dict mapping dataset prefix to (steps, means, stds)
    """
    base_dir = Path(base_dir)
    beta_dirs = sorted(base_dir.glob(f"{experiment_name}_beta_*"))

    if not beta_dirs:
        return {}

    beta_data = {}

    for beta_dir in beta_dirs:
        try:
            beta_str = beta_dir.name.split("beta_")[-1]
            beta = float(beta_str)

            # Determine which results directory to load
            if dataset_type == "validation":
                results_dir = beta_dir / "results/merged_model_validation"
            else:
                results_dir = beta_dir / "results/merged_model"

            if not results_dir.exists():
                continue

            # Load all step data for this beta
            df = load_summary_data(results_dir)

            # Extract data for each dataset
            dataset_data = {}
            for dataset_info in DATASETS.values():
                prefix = dataset_info["prefix"]
                dataset_df = df[df["dataset"].str.startswith(prefix)]

                if not dataset_df.empty:
                    steps, means, stds = compute_step_series(dataset_df)
                    dataset_data[prefix] = (steps, means, stds)

            if dataset_data:
                beta_data[beta] = {"data": dataset_data}

        except Exception as e:
            print(f"Warning: Could not load beta {beta_str}: {e}")
            continue

    return beta_data


def load_summary_data(results_dir):
    """Load all summary CSV files from step directories.

    Args:
        results_dir: Path to results directory containing step_* subdirectories

    Returns:
        DataFrame with columns: dataset, accuracy, run, step
    """
    results_dir = Path(results_dir)
    step_dirs = sorted(
        [d for d in results_dir.iterdir() if d.name.startswith("step_")]
    )

    all_data = []

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[1])

        try:
            step_df = get_individual_accuracies(step_dir)
            step_df["step"] = step_num
            all_data.append(step_df)
        except Exception as e:
            print(f"Warning: Could not process {step_dir.name}: {e}")
            continue

    if not all_data:
        raise RuntimeError(f"No valid evaluation data found in {results_dir}")

    return pd.concat(all_data, ignore_index=True)


def get_individual_accuracies(work_dir):
    """Get individual accuracies for each dataset from a model directory.

    Returns DataFrame with columns: dataset, accuracy, run
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


def compute_step_series(dataset_df):
    """Compute step-wise arrays of steps, means, stds for a dataset df.

    Returns:
        tuple: (steps, means, stds) as lists
    """
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


def plot_beta_comparison(beta_data, dataset_prefix, dataset_name, ylim=None):
    """Plot accuracy comparison across beta values for a single dataset.

    Args:
        beta_data: dict mapping beta to {"data": {prefix: (steps, means, stds)}}
        dataset_prefix: Prefix to identify the dataset
        dataset_name: Display name for the dataset
        ylim: Optional y-axis limits as tuple
    """
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap("Dark2")
    sorted_betas = sorted(beta_data.keys())

    # Plot each beta value as a separate line
    for idx, beta in enumerate(sorted_betas):
        beta_info = beta_data[beta]
        if dataset_prefix not in beta_info["data"]:
            continue

        steps, means, stds = beta_info["data"][dataset_prefix]
        color = cmap(idx % 8)

        valid_indices = ~np.isnan(means)
        valid_steps = np.array(steps)[valid_indices]
        valid_means = np.array(means)[valid_indices]
        valid_stds = np.array(stds)[valid_indices]

        # Plot line and shaded std region
        plt.plot(
            valid_steps,
            valid_means,
            "-",
            linewidth=2,
            color=color,
            alpha=0.8,
            label=f"β={beta}",
            zorder=2,
        )

        plt.fill_between(
            valid_steps,
            valid_means - valid_stds,
            valid_means + valid_stds,
            color=color,
            alpha=0.2,
            zorder=1,
            linewidth=0,
        )

        # Add markers
        plt.scatter(
            valid_steps,
            valid_means,
            s=50,
            color=color,
            alpha=1.0,
            zorder=3,
            edgecolor="none",
        )

    plt.xlabel("Number of merged models", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"{dataset_name} - Beta Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", alpha=0.3)

    if ylim is not None:
        plt.ylim(ylim)


def main():
    """Main function to load data and create plots."""
    parser = argparse.ArgumentParser(
        description="Plot beta sweep results across model merging steps."
    )

    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of the experiment (e.g., 'ema_holdout')",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/beta_sweep",
        help="Base directory containing beta sweep results (default: outputs/beta_sweep)",
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
        "--selection",
        action="store_true",
        help="Plot selection dataset results instead of validation datasets (default: validation)",
    )

    args = parser.parse_args()
    ylim = tuple(args.ylim) if args.ylim is not None else None
    dataset_type = "selection" if args.selection else "validation"

    plot_beta_sweep_per_dataset(
        base_dir=args.base_dir,
        experiment_name=args.experiment_name,
        ylim=ylim,
        dataset_type=dataset_type,
    )


if __name__ == "__main__":
    main()
