"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

import hydra
from omegaconf import DictConfig
from huggingface_hub import HfApi, snapshot_download
from merge import create_merge_instance
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        expand=["downloads"]
    )

    download(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(f"models/{base_model_id}", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(f"models/{base_model_id}", device_map="cpu")
    base_model_state_dict = base_model.state_dict()
    merged_state_dict = deepcopy(base_model_state_dict)
    merger = create_merge_instance(cfg)

    for model in models:
        print("Processing model: ", model.id)
        download(model.id)
        finetuned_model = AutoModelForCausalLM.from_pretrained(f"models/{model.id}", device_map="cpu")
        finetuned_model_state_dict = finetuned_model.state_dict()
        merged_state_dict = merger.merge([merged_state_dict, finetuned_model_state_dict])
        merged_model = AutoModelForCausalLM.from_pretrained(
            f"models/{base_model_id}",
            state_dict=merged_state_dict,
            device_map="cpu"
        )
        if cfg.evaluate:
            evaluate(merged_model, tokenizer)

    print(f"Created merge instance using {cfg.merge.method} method")

def download(model_id):
    """Download a model from HuggingFace Hub."""
    print(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"models/{model_id}",
        max_workers=1
    )
    print(f"Downloaded {model_id}.")

def evaluate(model, tokenizer):
    """Evaluate a model on a simple prompt."""
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()