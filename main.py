"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

import gc
import platform
import subprocess
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from huggingface_hub import HfApi, snapshot_download
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

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
        expand=["downloads", "safetensors"],
    )
    models = [model for model in models if is_bf16(model)]
    print(f"Found {len(models)} models to merge.")

    download(base_model_id, "current_model")
    download(base_model_id, "merged_model")
    evaluate("outputs/step_0")

    base_model = AutoModelForCausalLM.from_pretrained(
        "models/current_model", device_map="cpu", trust_remote_code=True
    )
    merged_state_dict = deepcopy(base_model.state_dict())

    merger = create_merge_instance(cfg)
    merger.update(merged_state_dict)

    for i, model in enumerate(models, start=1):
        download(model.id, "current_model")
        current_model_state_dict = load(model.id, "current_model")

        if current_model_state_dict is None:
            continue

        print(f"Merging {model.id}...")
        merged_state_dict = merger.update(current_model_state_dict)
        base_model.load_state_dict(merged_state_dict)
        save(base_model, "merged_model")
        evaluate(f"outputs/step_{i}")

        gc.collect()
        torch.cuda.empty_cache()


def is_bf16(model):
    """Check if a model is bf16."""
    return set(model.safetensors.parameters.keys()) == {"BF16"}


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


def evaluate(work_dir):
    """Call OpenCompass to evaluate the model using eval_llama.py."""
    if platform.system() == "Darwin":
        eval_script = "eval_llama_hf.py"
    else:
        eval_script = "eval_llama.py"

    result = subprocess.run(
        [
            "opencompass",
            eval_script,
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


if __name__ == "__main__":
    main()
