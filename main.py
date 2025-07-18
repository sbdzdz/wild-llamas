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

from merge import create_merge_instance


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    api = HfApi()
    base_model_id = cfg.base_model

    models = api.list_models(
        filter=f"base_model:finetune:{base_model_id}",
        sort="downloads",
        direction=-1,
        limit=cfg.model_limit,
        gated=False,
        expand=["downloads", "safetensors", "pipeline_tag"],
    )
    print(list(models))
    # models = [model for model in models if is_bf16(model)]
    is_bf16(models[0])
    print(f"Found {len(models)} BF16 models to merge.")

    download(base_model_id, "current_model")
    shutil.copytree("models/current_model", "models/merged_model", dirs_exist_ok=True)
    evaluate_current("outputs/step_0/current")

    base_model = AutoModelForCausalLM.from_pretrained(
        "models/current_model", device_map="cpu", trust_remote_code=True
    )
    base_state_dict = deepcopy(base_model.state_dict())
    merged_state_dict = deepcopy(base_model.state_dict())

    merger = create_merge_instance(cfg)
    merger.update(merged_state_dict)

    merging_step = 1
    for model in models:
        download(model.id, "current_model")

        current_model_state_dict = load(model.id, "current_model")
        if current_model_state_dict is None or not is_text_generation(model):
            print(f"Model {model.id} is not a text generation model. Skipping.")
            continue

        if are_nearly_equal(base_state_dict, current_model_state_dict):
            print(f"Model {model.id} is nearly equal to the base model. Skipping.")
            continue

        evaluate_current(f"outputs/step_{merging_step}/current")

        accuracy = get_accuracy(f"outputs/step_{merging_step}/current")
        if accuracy < 50.0:
            print(f"Model {model.id} has poor performance: {accuracy:.1f}. Skipping.")
            continue

        print(f"Merging {model.id}...")
        merged_state_dict = merger.update(current_model_state_dict)
        base_model.load_state_dict(merged_state_dict)
        save(base_model, "merged_model")
        evaluate_merged(f"outputs/step_{merging_step}/merged")
        merging_step += 1

        gc.collect()
        torch.cuda.empty_cache()


def is_bf16(model):
    """Check if a model is bf16."""
    print(model.safetensors.parameters.keys())
    return set(model.safetensors.parameters.keys()) == {"BF16"}


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


def download(model_id, folder):
    """Download a model from HuggingFace Hub to a fixed directory."""
    print(f"Downloading {model_id} to models/{folder}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"models/{folder}",
        max_workers=1,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {model_id}.")


def evaluate_current(work_dir):
    result = subprocess.run(
        [
            "opencompass",
            "eval_current.py",
            "--work-dir",
            work_dir,
        ],
        check=True,
    )
    return result


def evaluate_merged(work_dir):
    result = subprocess.run(
        [
            "opencompass",
            "eval_merged.py",
            "--work-dir",
            work_dir,
        ],
        check=True,
    )
    return result


def load(model_id, folder):
    """Load a model from the specified folder. Returns None if loading fails."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"models/{folder}", device_map="cpu", trust_remote_code=True
        )
        return model.state_dict()
    except ImportError:
        print(f"Model {model_id} requires additional dependencies. Skipping.")
        return None


def save(model, folder):
    """Save the merged model with a numbered name."""
    merged_model_path = Path(f"models/{folder}")
    merged_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_model_path)
    print(f"Saved merged model to {merged_model_path}")


def get_accuracy(work_dir):
    """Get the average accuracy from evaluation results."""
    step_dir = Path(work_dir)
    timestamp_dir = next(d for d in step_dir.iterdir() if d.is_dir())
    summary_dir = timestamp_dir / "summary"
    csv_file = next(summary_dir.glob("*.csv"))
    df = pd.read_csv(csv_file)
    df["current_model"] = pd.to_numeric(
        df["current_model"].replace("-", 0), errors="coerce"
    )
    return df["current_model"].mean()


if __name__ == "__main__":
    main()
