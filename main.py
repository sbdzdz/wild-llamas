"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

import hydra
from omegaconf import DictConfig
from huggingface_hub import HfApi, snapshot_download
from merge import create_merge_instance
from copy import deepcopy
from transformers import AutoModelForCausalLM
import gc
import torch
from pathlib import Path
import subprocess


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

    download(base_model_id)
    print(f"Evaluating {base_model_id}...")
    evaluate_with_opencompass()
    base_model = AutoModelForCausalLM.from_pretrained(
        "models/current_model", device_map="cpu"
    )
    save_merged_model(base_model, 0)

    base_model_state_dict = base_model.state_dict()
    merged_state_dict = deepcopy(base_model_state_dict)

    merger = create_merge_instance(cfg)

    for i, model in enumerate(models, start=1):
        download(model.id)

        print(f"Evaluating {model.id}...")
        evaluate_with_opencompass()
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            "models/current_model", device_map="cpu"
        )
        finetuned_model_state_dict = finetuned_model.state_dict()

        print(f"Merging {model.id}...")
        merged_state_dict = merger.merge(
            [merged_state_dict, finetuned_model_state_dict]
        )
        base_model.load_state_dict(merged_state_dict)

        save_merged_model(base_model, i)

        gc.collect()
        torch.cuda.empty_cache()


def download(model_id):
    """Download a model from HuggingFace Hub to a fixed directory."""
    print(f"Downloading {model_id} to models/current_model ...")
    snapshot_download(
        repo_id=model_id,
        local_dir="models/current_model",
        max_workers=1,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {model_id}.")


def evaluate_with_opencompass():
    """Call OpenCompass to evaluate the model using eval_llama.py."""
    result = subprocess.run(["opencompass", "eval_llama.py"], check=True)
    return result


def save_merged_model(model, merge_index):
    """Save the merged model with a numbered name."""
    merged_model_path = Path(f"models/merged_model_{merge_index}")
    merged_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_model_path)
    print(f"Saved merged model to {merged_model_path}")


def is_bf16(model):
    """Check if a model is bf16."""
    return set(model.safetensors.parameters.keys()) == {"BF16"}


if __name__ == "__main__":
    main()
