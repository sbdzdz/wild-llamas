"""Evaluation utilities for model merging."""

import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

TOP_DIR = Path(__file__).resolve().parent.parent


def setup_eval_paths(model_id: str, output_root: Path) -> tuple[Path, Path]:
    """Return (model_path, eval_output_dir) for a given model_id."""
    model_name = model_id.replace("/", "--")
    model_path = TOP_DIR / f"models/{model_name}"
    model_eval_dir = output_root / f"results/current/{model_name}"
    return model_path, model_eval_dir


def evaluate(
    model_path,
    output_dir,
    eval_runs=1,
    batch_size=32,
    use_cache=True,
    datasets=None,
    num_eval_samples=None,
):
    """Evaluate a model and return its mean accuracy across multiple runs.

    Args:
        model_path: Path to the model directory
        output_dir: Absolute output directory path (will be modified based on num_eval_samples)
        eval_runs: Number of evaluation runs to perform
        batch_size: Batch size for evaluation
        use_cache: Whether to use cached results if available
        datasets: List of dataset names to evaluate on
        num_eval_samples: Number of samples to use from each dataset during evaluation, None for full dataset

    Returns:
        float: Mean accuracy across all evaluation runs
    """
    if datasets is None:
        datasets = []
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if num_eval_samples is not None:
        parts = output_dir.parts
        if len(parts) >= 2 and parts[-2].startswith("merged_model"):
            output_dir = output_dir.parent.parent / f"{parts[-2]}_partial" / parts[-1]

    model_name = model_path.name

    if use_cache and output_dir.exists():
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if len(subdirs) == eval_runs:
            print(f"Using existing evaluation results at {output_dir}")
            accuracies = [get_accuracy(subdir) for subdir in subdirs]
            mean_accuracy = sum(accuracies) / len(accuracies)
            return mean_accuracy

    if not use_cache and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    cuda_devices = ",".join(str(i) for i in range(num_gpus))

    accuracies = []

    for run_idx in range(eval_runs):
        print(f"Running evaluation {run_idx + 1}/{eval_runs} for {model_name}")

        with setup_unique_config(
            output_dir, batch_size, model_path, datasets, num_eval_samples
        ) as eval_config_path:
            subprocess.run(
                [
                    "opencompass",
                    str(eval_config_path),
                    "--work-dir",
                    output_dir,
                    "--max-num-worker",
                    str(num_gpus),
                ],
                env={"CUDA_VISIBLE_DEVICES": cuda_devices, **os.environ},
                check=True,
            )

        latest_dir = get_latest_subdir(output_dir)
        accuracy = get_accuracy(latest_dir)
        accuracies.append(accuracy)
        print(f"Run {run_idx + 1} accuracy: {accuracy:.2f}")

    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_accuracy


@contextmanager
def setup_unique_config(
    parent_dir: Path,
    batch_size: int,
    model_path: Path,
    datasets: list,
    num_eval_samples: int | None = None,
):
    """Create a unique temporary OpenCompass config and yield its path.

    Args:
        parent_dir: Parent directory for temporary config
        batch_size: Batch size for evaluation
        model_path: Path to model directory
        datasets: List of dataset names to include
        num_eval_samples: Number of samples to use from each dataset, None for full dataset

    The temporary directory and file are deleted when the context exits.
    """
    if num_eval_samples is not None and num_eval_samples <= 0:
        raise ValueError(f"num_eval_samples must be positive, got {num_eval_samples}")

    dataset_mapping = {
        "mmlu": "mmlu_datasets",
        "mmlu_pro": "mmlu_pro_datasets",
        "math500": "math_datasets",
        "gpqa": "gpqa_datasets",
        "gsm8k": "gsm8k_datasets",
    }

    dataset_vars = [
        dataset_mapping[name] for name in datasets if name in dataset_mapping
    ]
    if dataset_vars:
        datasets_line = f"datasets = [{', '.join(f'*{var}' for var in dataset_vars)}]"
    else:
        datasets_line = "datasets = []"

    num_eval_samples_literal = (
        num_eval_samples if num_eval_samples is not None else "None"
    )

    template_path = Path(__file__).resolve().parent / "eval_template.py"
    template_text = template_path.read_text()
    replaced_text = (
        template_text.replace("BATCH_SIZE = None", f"BATCH_SIZE = {batch_size}")
        .replace('path="models/eval_model"', f'path="{model_path}"')
        .replace("datasets = []", datasets_line)
        .replace(
            "NUM_EVAL_SAMPLES = None",
            f"NUM_EVAL_SAMPLES = {num_eval_samples_literal}",
        )
    )

    tmp_dir = Path(tempfile.mkdtemp(dir=str(parent_dir), prefix="merge-"))
    try:
        timestamp = int(time.time() * 1000000)
        eval_config_path = tmp_dir / f"eval_{timestamp}.py"
        eval_config_path.write_text(replaced_text)
        yield eval_config_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_latest_subdir(parent_dir):
    """Get the most recently modified subdirectory from a parent directory."""
    return next(
        d
        for d in sorted(
            parent_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if d.is_dir()
    )


def get_accuracy(timestamp_dir):
    """Get the average accuracy from evaluation results."""
    timestamp_dir = Path(timestamp_dir)
    summary_dir = timestamp_dir / "summary"
    csv_file = next(summary_dir.glob("*.csv"))
    df = pd.read_csv(csv_file)
    df["eval_model"] = pd.to_numeric(df["eval_model"].replace("-", 0), errors="coerce")
    return df["eval_model"].mean()
