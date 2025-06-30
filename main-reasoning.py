"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

import gc
import platform
import subprocess
import shutil
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from merge import create_merge_instance


@hydra.main(version_base=None, config_path="config", config_name="config-reasoning")
def main(cfg: DictConfig):
    base_model_id = cfg.base_model
    models = cfg.models

    current_model, current_tokenizer = load_model_and_tokenizer(base_model_id)
    save_model_and_tokenizer(current_model, current_tokenizer, "current_model")
    save_model_and_tokenizer(current_model, current_tokenizer, "merged_model")
    evaluate_current("outputs-reasoning/step_0/current")
    
    # base_state_dict = deepcopy(base_model.state_dict())
    # merged_state_dict = deepcopy(base_model.state_dict())

    # merger = create_merge_instance(cfg)
    # merger.update(merged_state_dict)

    merging_step = 1
    for model in models:
        current_model, current_tokenizer = load_model_and_tokenizer(model)
        save_model_and_tokenizer(current_model, current_tokenizer, "current_model")
        
        # current_model_state_dict = load(model, "current_model")
        # if current_model_state_dict is None:
        #     continue

        # if are_nearly_equal(base_state_dict, current_model_state_dict):
        #     print(f"Model {model} is nearly equal to the base model. Skipping.")
        #     continue
        # else:
        #     print(f"Model {model} is not nearly equal to the base model. Merging.")

        evaluate_current(f"outputs-reasoning/step_{merging_step}/current")

        # current_avg = get_accuracy(f"outputs-reasoning/step_{merging_step}/current")
        # if current_avg < 50.0:
        #     print(f"Model {model} has poor performance: {current_avg:.1f}. Skipping.")
        #     continue

        # print(f"Merging {model}...")
        # merged_state_dict = merger.update(current_model_state_dict)
        # base_model.load_state_dict(merged_state_dict)
        # evaluate_merged(f"outputs-reasoning/step_{merging_step}/merged")
        merging_step += 1  # Increment merging step only after successful merge

        gc.collect()
        torch.cuda.empty_cache()


def is_bf16(model):
    """Check if a model is bf16."""
    return set(model.safetensors.parameters.keys()) == {"BF16"}


def is_text_generation(model):
    """Check if a model is text generation model."""
    return model.pipeline_tag == 'text-generation'


def are_nearly_equal(sd1, sd2):
    """Check if two state dictionaries are nearly equal."""
    for key in sd1.keys():
        if sd1[key].shape != sd2[key].shape:
            return False
        if not torch.allclose(sd1[key], sd2[key]):
            return False
    return True


def evaluate_current(work_dir):
    if platform.system() == "Darwin":
        eval_script = "eval_current_hf_reasoning.py"
    else:
        eval_script = "eval_current_reasoning.py"
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


def evaluate_merged(work_dir):
    if platform.system() == "Darwin":
        eval_script = "eval_merged_hf_reasoning.py"
    else:
        eval_script = "eval_merged_reasoning.py"
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


def load_model_and_tokenizer(model_id):
    """Load both model and tokenizer from HuggingFace Hub."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    return model, tokenizer


def save_model_and_tokenizer(model, tokenizer, folder):
    """Save both model and tokenizer together."""
    model_path = Path(f"models/{folder}")
    if model_path.exists():  # remove so the files do not mix in unexpected way
        shutil.rmtree(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved model and tokenizer to {model_path}")


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
