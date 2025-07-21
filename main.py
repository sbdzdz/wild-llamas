"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

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

    skipped_models = load_skipped_models()

    models = api.list_models(
        filter=f"base_model:finetune:{base_model_id}",
        sort="downloads",
        direction=-1,
        gated=False,
        expand=["downloads", "safetensors", "pipeline_tag"],
    )

    download(base_model_id, "current_model")
    shutil.copytree("models/current_model", "models/merged_model", dirs_exist_ok=True)
    current_accuracy = evaluate(
        model_path="models/current_model",
        work_dir="outputs/step_0/current",
    )
    log_merge(base_model_id, "merged", current_accuracy, current_accuracy)

    base_model = AutoModelForCausalLM.from_pretrained(
        "models/current_model", device_map="cpu", trust_remote_code=True
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

        download(model.id, "current_model")
        current_model_state_dict = load(model.id, "current_model")

        if current_model_state_dict is None:
            continue

        if not dtypes_match(base_state_dict, current_model_state_dict):
            log_merge(model.id, "dtypes_mismatch")
            continue

        if are_nearly_equal(base_state_dict, current_model_state_dict):
            log_merge(model.id, "nearly_equal")
            continue

        if Path("outputs/current").exists():
            shutil.rmtree("outputs/current")

        current_accuracy = evaluate(
            model_path="models/current_model",
            work_dir="outputs/current",
        )

        if current_accuracy < 50.0:
            log_merge(model.id, "poor_performance", current_accuracy)
            continue

        merging_step += 1
        print(f"Merging {model.id}...")
        merged_state_dict = merger.update(current_model_state_dict)
        base_model.load_state_dict(merged_state_dict)
        save(base_model, "merged_model")
        merged_accuracy = evaluate(
            model_path="models/merged_model",
            work_dir=f"outputs/step_{merging_step}/merged",
        )
        shutil.copytree("outputs/current", f"outputs/step_{merging_step}/current")
        log_merge(model.id, "merged", current_accuracy, merged_accuracy)

        if merging_step > cfg.model_limit:
            break

        gc.collect()
        torch.cuda.empty_cache()


def load_skipped_models():
    """Load list of model IDs that have been skipped from skipped_models.csv."""
    skipped_file = Path(__file__).parent / "skipped_models.csv"
    if not skipped_file.exists():
        return set()

    df = pd.read_csv(skipped_file)
    return set(df["model_id"].tolist())


def is_bf16(model):
    """Check if a model is bf16."""
    return model.safetensors is not None and set(
        model.safetensors.parameters.keys()
    ) == {"BF16"}


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


def dtypes_match(sd1, sd2):
    """Check if two state dictionaries have the same dtypes."""
    for key in sd1.keys():
        if key not in sd2:
            return False
        if sd1[key].dtype != sd2[key].dtype:
            return False
    return True


def download(model_id, folder):
    """Download a model from HuggingFace Hub to a fixed directory."""
    folder_path = Path(f"models/{folder}")

    if folder_path.exists():
        shutil.rmtree(folder_path)

    print(f"Downloading {model_id} to {folder_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(folder_path),
        max_workers=1,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {model_id}.")


def evaluate(model_path, work_dir):
    set_eval_model_symlink(model_path)
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


def load(model_id, folder):
    """Load a model from the specified folder. Returns None if loading fails."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"models/{folder}", device_map="cpu", trust_remote_code=True
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


def save(model, folder):
    """Save the merged model."""
    model_path = Path(f"models/{folder}")
    if model_path.exists():
        shutil.rmtree(model_path)
    model_path.mkdir(parents=True)
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
