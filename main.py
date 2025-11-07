"""Find, download, merge, and evaluate finetunes for a base model.

Find finetunes for the base model on HuggingFace, download them, evaluate each one,
and incrementally merge those that pass checks into a running merged model.
"""

import csv
import gc
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
import tempfile
import time
from contextlib import contextmanager

import hydra
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from wildllamas.merge import create_merge_instance

TOP_DIR = Path(__file__).parent


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    api = HfApi()
    base_model_id = cfg.base_model

    output_dir = Path(cfg.get("output_dir", "outputs/opencompass"))
    output_dir = output_dir if output_dir.is_absolute() else (TOP_DIR / output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    setup_model_directory(cfg)

    skipped_models = load_skipped_models()
    models = fetch_or_load_models(api, base_model_id)

    base_model_path = download(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="cpu", trust_remote_code=True
    )
    base_state_dict = base_model.state_dict()
    merged_model_path = output_dir / "merged_model"

    shutil.copytree(
        base_model_path,
        merged_model_path,
        ignore=shutil.ignore_patterns(
            ".cache",
            "*.lock",
            "*.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            ".gitattributes",
        ),
    )
    base_eval_dir = output_dir / "results/merged_model/step_0"
    current_accuracy = evaluate(
        base_model_path,
        base_eval_dir,
        cfg.eval_runs,
        batch_size=int(cfg["batch_size"]),
    )
    best_merged_accuracy = current_accuracy
    log_merged_model(output_dir, base_model_id, current_accuracy, current_accuracy)
    merged_state_dict = deepcopy(base_model.state_dict())
    merging_step = 0

    merger = create_merge_instance(cfg)
    merger.update(merged_state_dict)

    overwrite_skipped = cfg.get("overwrite_skipped", False)

    for model in models:
        gc.collect()
        torch.cuda.empty_cache()

        if model.id in skipped_models:
            print(f"Skipping {model.id} (previously skipped)")
            continue

        if not is_approx_8b_params(model):
            log_skipped_model(model.id, "not_8b", overwrite_skipped)
            continue

        if not is_bf16(model):
            log_skipped_model(model.id, "not_bf16", overwrite_skipped)
            continue

        if not is_text_generation(model):
            log_skipped_model(model.id, "not_text", overwrite_skipped)
            continue

        try:
            download(model.id)
        except Exception:
            log_skipped_model(model.id, "download_error", overwrite_skipped)
            continue

        try:
            current_model_state_dict = load(model.id)
        except Exception:
            log_skipped_model(model.id, "load_error", overwrite_skipped)
            continue

        if not state_dicts_match(base_state_dict, current_model_state_dict):
            log_skipped_model(model.id, "dtypes_mismatch", overwrite_skipped)
            continue

        if are_nearly_equal(base_state_dict, current_model_state_dict):
            log_skipped_model(model.id, "nearly_equal", overwrite_skipped)
            continue

        if cfg.evaluate_current:
            model_name = model.id.replace("/", "--")
            model_path = TOP_DIR / f"models/{model_name}"
            model_eval_dir = output_dir / f"results/current/{model_name}"
            current_accuracy = evaluate(
                model_path,
                output_dir=model_eval_dir,
                eval_runs=cfg.eval_runs,
                batch_size=int(cfg["batch_size"]),
            )
            min_current_accuracy = float(cfg.get("min_current_accuracy", 0.0))
            if current_accuracy < min_current_accuracy:
                log_skipped_model(model.id, "poor_performance", overwrite_skipped)
                continue
        else:
            current_accuracy = None

        if cfg.greedy:
            previous_merged_state_dict = deepcopy(merged_state_dict)
            previous_step_count = merger.step_count

        merging_step += 1
        print(f"Merging {model.id}...")
        merged_state_dict = merger.update(current_model_state_dict)
        base_model.load_state_dict(merged_state_dict)

        if cfg.greedy:
            save(base_model, merged_model_path)
            merged_eval_dir = (
                output_dir / "results/merged_model" / f"step_{merging_step}"
            )
            merged_accuracy = evaluate(
                merged_model_path,
                output_dir=merged_eval_dir,
                eval_runs=cfg.eval_runs,
                batch_size=int(cfg["batch_size"]),
                use_cache=False,
            )

            if merged_accuracy <= best_merged_accuracy:
                print(
                    f"Merge rejected: accuracy decreased from {best_merged_accuracy:.2f} to {merged_accuracy:.2f}"
                )
                merged_state_dict = previous_merged_state_dict
                merger.current_average = deepcopy(previous_merged_state_dict)
                merger.step_count = previous_step_count
                merging_step -= 1
                base_model.load_state_dict(merged_state_dict)
                save(base_model, merged_model_path)
                log_skipped_model(model.id, "greedy_rejected", overwrite_skipped)
                continue
            else:
                print(
                    f"Merge accepted: accuracy {best_merged_accuracy:.2f} -> {merged_accuracy:.2f}"
                )
                best_merged_accuracy = merged_accuracy
                log_merged_model(
                    output_dir, model.id, current_accuracy, merged_accuracy
                )
        else:
            save(base_model, merged_model_path)
            if (
                merging_step % cfg.eval_every_n_merges == 0
                or merging_step == cfg.model_limit
            ):
                merged_eval_dir = (
                    output_dir / "results/merged_model" / f"step_{merging_step}"
                )
                merged_accuracy = evaluate(
                    merged_model_path,
                    output_dir=merged_eval_dir,
                    eval_runs=cfg.eval_runs,
                    batch_size=int(cfg["batch_size"]),
                )
            else:
                merged_accuracy = None

            log_merged_model(output_dir, model.id, current_accuracy, merged_accuracy)

        if merging_step >= cfg.model_limit:
            break


def setup_model_directory(cfg: DictConfig):
    """Setup model directory and create symlink if model_dir is specified."""
    models_dir = TOP_DIR / "models"

    if cfg.get("model_dir"):
        actual_models_dir = Path(cfg.model_dir)
        actual_models_dir.mkdir(parents=True, exist_ok=True)

        if actual_models_dir.resolve() == models_dir.resolve():
            print(f"Using in-repo models directory at {models_dir.resolve()}")
            return

        if models_dir.exists() or models_dir.is_symlink():
            if models_dir.is_symlink():
                models_dir.unlink()
            else:
                shutil.rmtree(models_dir)

        models_dir.symlink_to(actual_models_dir.resolve())
        print(f"Created symlink: models -> {actual_models_dir.resolve()}")
        print(f"All models will be stored in: {actual_models_dir.resolve()}")
    else:
        models_dir.mkdir(exist_ok=True)


def load_skipped_models():
    """Load list of model IDs that have been skipped from skipped_models.csv."""
    skipped_file = TOP_DIR / "skipped_models.csv"
    if not skipped_file.exists():
        return set()

    df = pd.read_csv(skipped_file)
    return set(df["model_id"].tolist())


def fetch_or_load_models(api, base_model_id):
    """Fetch model list from API or load from all_models.csv for consistent ordering."""
    model_ids = load_all_models()
    models = list(
        api.list_models(
            filter=f"base_model:finetune:{base_model_id}",
            sort="downloads",
            direction=-1,
            gated=False,
            expand=["safetensors", "pipeline_tag"],
        )
    )
    if model_ids is None:
        print("Fetching model list from HuggingFace API...")
        model_ids = [model.id for model in models]
        save_all_models(model_ids)
        print(f"Saved {len(model_ids)} model IDs to all_models.csv")
        return models
    else:
        print(f"Loaded {len(model_ids)} model IDs from all_models.csv")
        id_to_model = {model.id: model for model in models}
        models = []
        for model_id in model_ids:
            if model_id in id_to_model:
                models.append(id_to_model[model_id])
            else:
                print(f"Warning: {model_id} not found in API; skipping")
        return models


def load_all_models():
    """Load list of all model IDs from all_models.csv."""
    all_models_file = TOP_DIR / "all_models.csv"
    if not all_models_file.exists():
        return None

    df = pd.read_csv(all_models_file)
    return df["model_id"].tolist()


def save_all_models(model_ids):
    """Save the list of all model IDs to all_models.csv."""
    all_models_file = TOP_DIR / "all_models.csv"

    df = pd.DataFrame({"model_id": model_ids})
    df.to_csv(all_models_file, index=False)


def is_bf16(model):
    """Check if a model is bf16."""
    return model.safetensors is None or "BF16" in model.safetensors.parameters.keys()


def is_text_generation(model):
    """Check if a model is text generation model."""
    return model.pipeline_tag is None or model.pipeline_tag == "text-generation"


def is_approx_8b_params(model):
    """Return True if model.safetensors.total (param count) is approximately 8B."""
    if model.safetensors is None:
        return True
    if "total" not in model.safetensors.keys():
        return True
    else:
        return 7_000_000_000 <= int(model.safetensors.total) <= 9_000_000_000


def are_nearly_equal(sd1, sd2):
    """Check if two state dictionaries are nearly equal."""
    for key in sd1.keys():
        if not torch.allclose(sd1[key], sd2[key]):
            return False
    return True


def state_dicts_match(sd1, sd2):
    """Check if two state dictionaries have matching keys, dtypes, and shapes."""
    if set(sd1.keys()) != set(sd2.keys()):
        return False
    for key in sd1.keys():
        if sd1[key].dtype != sd2[key].dtype:
            return False
        if sd1[key].shape != sd2[key].shape:
            return False
    return True


def download(model_id):
    """Download a model from HuggingFace Hub to a model-specific directory."""
    model_name = model_id.replace("/", "--")
    model_path = TOP_DIR / f"models/{model_name}"

    if model_path.exists():
        print(f"Model {model_id} already exists at {model_path}, skipping download.")
        return model_path

    print(f"Downloading {model_id} to {model_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        max_workers=1,
        local_dir_use_symlinks=False,
    )
    return model_path


def evaluate(model_path, output_dir, eval_runs=1, batch_size=32, use_cache=True):
    """Evaluate a model and return its mean accuracy across multiple runs.

    Args:
        model_path: Path to the model directory
        output_dir: Absolute output directory path
        eval_runs: Number of evaluation runs to perform
        batch_size: Batch size for evaluation
        use_cache: Whether to use cached results if available

    Returns:
        float: Mean accuracy across all evaluation runs
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    model_name = model_path.name

    if use_cache and output_dir.exists():
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if len(subdirs) == eval_runs:
            print(f"Using existing evaluation results at {output_dir}")
            accuracies = [get_accuracy(subdir) for subdir in subdirs]
            mean_accuracy = sum(accuracies) / len(accuracies)
            return mean_accuracy

    output_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    cuda_devices = ",".join(str(i) for i in range(num_gpus))

    accuracies = []

    for run_idx in range(eval_runs):
        print(f"Running evaluation {run_idx + 1}/{eval_runs} for {model_name}")

        with setup_unique_config(
            output_dir, batch_size, model_path
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
def setup_unique_config(parent_dir: Path, batch_size: int, model_path: Path):
    """Create a unique temporary OpenCompass config and yield its path.

    Args:
        parent_dir: Parent directory for temporary config
        batch_size: Batch size for evaluation
        model_path: Path to model directory

    The temporary directory and file are deleted when the context exits.
    """
    template_path = TOP_DIR / "eval.py"
    template_text = template_path.read_text()
    replaced_text = (
        template_text.replace("max_batch_size=None", f"max_batch_size={batch_size}")
        .replace("batch_size=None", f"batch_size={batch_size}")
        .replace('path="models/eval_model"', f'path="{model_path}"')
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
    """Get the most recently modified subdirectory from a parent directory.

    Args:
        parent_dir: Path to the parent directory

    Returns:
        Path to the most recently modified subdirectory
    """
    return next(
        d
        for d in sorted(
            parent_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if d.is_dir()
    )


def get_accuracy(timestamp_dir):
    """Get the average accuracy from evaluation results.

    Args:
        timestamp_dir: The timestamp directory containing evaluation results
    """
    timestamp_dir = Path(timestamp_dir)
    summary_dir = timestamp_dir / "summary"
    csv_file = next(summary_dir.glob("*.csv"))
    df = pd.read_csv(csv_file)
    df["eval_model"] = pd.to_numeric(df["eval_model"].replace("-", 0), errors="coerce")
    return df["eval_model"].mean()


def load(model_id):
    """Load a model from the model directory. Returns None if loading fails."""
    model_name = model_id.replace("/", "--")
    model = AutoModelForCausalLM.from_pretrained(
        TOP_DIR / f"models/{model_name}", device_map="cpu", trust_remote_code=True
    )
    return model.state_dict()


def save(model, model_path):
    """Save the merged model to the specified path."""
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    print(f"Saved merged model to {model_path}")


def log_merged_model(output_root, model_id, current_accuracy, merged_accuracy):
    """Log a successful merge to the merge log."""
    log_file = output_root / "merge_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["model_id", "current_accuracy", "merged_accuracy"])
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        current_acc_value = current_accuracy if current_accuracy is not None else ""
        merged_acc_value = merged_accuracy if merged_accuracy is not None else ""
        writer.writerow([model_id, current_acc_value, merged_acc_value])


def log_skipped_model(model_id, reason, overwrite_skipped):
    """Save a skipped model to skipped_models.csv if overwrite_skipped is True."""
    if not overwrite_skipped:
        print(
            f"Skipping {model_id} ({reason}) - not writing to skipped_models.csv (read-only mode)"
        )
        return

    skipped_file = TOP_DIR / "skipped_models.csv"

    if not skipped_file.exists():
        skipped_file.write_text("model_id,reason\n")

    with skipped_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_id, reason])


if __name__ == "__main__":
    main()
