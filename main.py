"""Find, download, merge, and evaluate finetunes for a base model.

Find finetunes for the base model on HuggingFace, download them, evaluate each one,
and incrementally merge those that pass checks into a running merged model.
"""

import gc
import subprocess
import shutil
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from huggingface_hub import HfApi, snapshot_download
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
import pandas as pd
import csv

from merge import create_merge_instance


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    api = HfApi()
    base_model_id = cfg.base_model

    setup_model_directory(cfg)

    skipped_models = load_skipped_models()
    models = fetch_or_load_models(api, base_model_id)

    download(base_model_id)
    base_model_name = base_model_id.replace("/", "--")
    shutil.copytree(
        f"models/{base_model_name}", "models/merged_model", dirs_exist_ok=True
    )
    current_accuracy = evaluate(base_model_id, overwrite=cfg.overwrite)
    log_merge(base_model_id, "merged", current_accuracy, current_accuracy)

    base_model = AutoModelForCausalLM.from_pretrained(
        f"models/{base_model_name}", device_map="cpu", trust_remote_code=True
    )
    base_state_dict = deepcopy(base_model.state_dict())
    merged_state_dict = deepcopy(base_model.state_dict())

    merger = create_merge_instance(cfg)
    merger.update(merged_state_dict)

    merging_step = 0
    for model in models:
        if model.id in skipped_models:
            print(f"Skipping {model.id} (previously skipped)")
            continue

        if not is_bf16(model):
            log_merge(model.id, "not_bf16")
            continue

        if not is_text_generation(model):
            log_merge(model.id, "not_text")
            continue

        download(model.id)
        current_model_state_dict = load(model.id)

        if current_model_state_dict is None:
            log_merge(model.id, "load_error")
            continue

        if not tensors_match(base_state_dict, current_model_state_dict):
            log_merge(model.id, "dtypes_mismatch")
            continue

        if are_nearly_equal(base_state_dict, current_model_state_dict):
            log_merge(model.id, "nearly_equal")
            continue

        current_accuracy = evaluate(model.id, overwrite=cfg.overwrite)

        if current_accuracy < 50.0:
            log_merge(model.id, "poor_performance", current_accuracy)
            continue

        merging_step += 1
        print(f"Merging {model.id}...")
        merged_state_dict = merger.update(current_model_state_dict)
        base_model.load_state_dict(merged_state_dict)
        save(base_model, "merged_model")
        merged_accuracy = evaluate(
            "merged_model",
            f"outputs/opencompass/merged_model/step_{merging_step}",
            overwrite=cfg.overwrite,
        )
        log_merge(model.id, "merged", current_accuracy, merged_accuracy)
        save_merged_model(model.id)

        if merging_step > cfg.model_limit:
            break

        gc.collect()
        torch.cuda.empty_cache()


def setup_model_directory(cfg: DictConfig):
    """Setup model directory and create symlink if model_dir is specified."""
    models_dir = Path("models")

    if cfg.get("model_dir"):
        model_dir_path = Path(cfg.model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)

        if models_dir.exists() or models_dir.is_symlink():
            if models_dir.is_symlink():
                models_dir.unlink()
            else:
                shutil.rmtree(models_dir)

        models_dir.symlink_to(model_dir_path.resolve())
        print(f"Created symlink: models -> {model_dir_path.resolve()}")
        print(f"All models will be stored in: {model_dir_path.resolve()}")
    else:
        models_dir.mkdir(exist_ok=True)


def load_skipped_models():
    """Load list of model IDs that have been skipped from skipped_models.csv."""
    skipped_file = Path(__file__).parent / "skipped_models.csv"
    if not skipped_file.exists():
        return set()

    df = pd.read_csv(skipped_file)
    return set(df["model_id"].tolist())


def load_merged_models():
    """Load list of model IDs that have been successfully merged from merged_models.csv."""
    merged_file = Path(__file__).parent / "merged_models.csv"
    if not merged_file.exists():
        return set()

    df = pd.read_csv(merged_file)
    return set(df["model_id"].tolist())


def save_merged_model(model_id):
    """Save a successfully merged model to merged_models.csv."""
    merged_file = Path(__file__).parent / "merged_models.csv"

    if not merged_file.exists():
        merged_file.write_text("model_id\n")

    with merged_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_id])


def fetch_or_load_models(api, base_model_id):
    """Fetch model list from API or load from all_models.csv for consistent ordering."""
    model_ids = load_all_models()
    if model_ids is None:
        print("Fetching model list from HuggingFace API...")
        models = api.list_models(
            filter=f"base_model:finetune:{base_model_id}",
            sort="downloads",
            direction=-1,
            gated=False,
            expand=["downloads", "safetensors", "pipeline_tag"],
        )
        model_ids = [model.id for model in models]
        save_all_models(model_ids)
        print(f"Saved {len(model_ids)} model IDs to all_models.csv")
        return models
    else:
        print(f"Loaded {len(model_ids)} model IDs from all_models.csv")
        models = []
        for model_id in model_ids:
            try:
                model_info = api.model_info(
                    model_id, expand=["downloads", "safetensors", "pipeline_tag"]
                )
                models.append(model_info)
            except Exception as e:
                print(f"Could not fetch info for {model_id}: {e}")
                continue
        return models


def load_all_models():
    """Load list of all model IDs from all_models.csv."""
    all_models_file = Path(__file__).parent / "all_models.csv"
    if not all_models_file.exists():
        return None

    df = pd.read_csv(all_models_file)
    return df["model_id"].tolist()


def save_all_models(model_ids):
    """Save the list of all model IDs to all_models.csv."""
    all_models_file = Path(__file__).parent / "all_models.csv"

    df = pd.DataFrame({"model_id": model_ids})
    df.to_csv(all_models_file, index=False)


def is_bf16(model):
    """Check if a model is bf16."""
    return model.safetensors is None or "BF16" in model.safetensors.parameters.keys()


def is_text_generation(model):
    """Check if a model is text generation model."""
    return model.pipeline_tag is None or model.pipeline_tag == "text-generation"


def are_nearly_equal(sd1, sd2):
    """Check if two state dictionaries are nearly equal."""
    for key in sd1.keys():
        if sd1[key].shape != sd2[key].shape:
            return False
        if not torch.allclose(sd1[key], sd2[key]):
            return False
    return True


def tensors_match(sd1, sd2):
    """Check if two state dictionaries have matching keys, dtypes, and shapes."""
    for key in sd1.keys():
        if key not in sd2:
            return False
        if sd1[key].dtype != sd2[key].dtype:
            return False
        if sd1[key].shape != sd2[key].shape:
            return False
    return True


def download(model_id):
    """Download a model from HuggingFace Hub to a model-specific directory."""
    model_name = model_id.replace("/", "--")
    folder_path = Path(f"models/{model_name}")

    if folder_path.exists():
        print(f"Model {model_id} already exists at {folder_path}, skipping download.")
        return

    print(f"Downloading {model_id} to {folder_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(folder_path),
        max_workers=1,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {model_id}.")


def evaluate(model_id, work_dir=None, overwrite=False):
    """Evaluate a model and return its accuracy.

    Args:
        model_id: The model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct" or path like "models/merged_model")
        work_dir: Optional work directory. If None, derives from model_id
        overwrite: If True, overwrite any existing results in work_dir. If False, reuse if present.
    """
    if work_dir is None:
        model_name = model_id.replace("/", "--")
        model_path = f"models/{model_name}"
        work_dir = Path(f"outputs/opencompass/{model_name}")
    else:
        model_name = Path(work_dir).parent.name
        model_path = f"models/{model_name}"
        work_dir = Path(work_dir)

    if not overwrite and work_dir.exists():
        return get_accuracy(work_dir)

    if overwrite and work_dir.exists():
        shutil.rmtree(work_dir)

    set_eval_model_symlink(model_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "opencompass",
            "eval_llama.py",
            "--work-dir",
            work_dir,
        ],
        check=True,
    )
    return get_accuracy(work_dir)


def set_eval_model_symlink(target):
    symlink_path = Path("models/eval_model")
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    target_abs = Path(target).resolve()
    symlink_path.symlink_to(target_abs)


def get_accuracy(work_dir):
    """Get the average accuracy from evaluation results."""
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


def load(model_id):
    """Load a model from the model directory. Returns None if loading fails."""
    model_name = model_id.replace("/", "--")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"models/{model_name}", device_map="cpu", trust_remote_code=True
        )
        return model.state_dict()
    except ImportError:
        log_merge(model_id, "import_error")
        return None
    except ValueError:
        log_merge(model_id, "value_error")
        return None
    except RuntimeError as e:
        log_merge(model_id, "runtime_error", str(e))
        return None


def save(model, model_name):
    """Save the merged model."""
    model_path = Path(f"models/{model_name}")
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    print(f"Saved merged model to {model_path}")


def log_merge(model_id, status, current_accuracy=None, merged_accuracy=None):
    log_file = Path(__file__).parent / "outputs/merge_log.csv"
    log_file.parent.mkdir(exist_ok=True)
    if not log_file.exists():
        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["model_id", "status", "current_accuracy", "merged_accuracy"]
            )
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model_id, status, current_accuracy, merged_accuracy])

    if status != "merged":
        save_skipped_model(model_id, status)


def save_skipped_model(model_id, reason):
    """Save a skipped model to skipped_models.csv."""
    skipped_file = Path(__file__).parent / "skipped_models.csv"

    if not skipped_file.exists():
        skipped_file.write_text("model_id,reason\n")

    with skipped_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_id, reason])


if __name__ == "__main__":
    main()
