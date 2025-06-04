"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

import hydra
from omegaconf import DictConfig
from huggingface_hub import HfApi, snapshot_download
from merge import create_merge_instance

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    api = HfApi()
    base_model = cfg.base_model

    models = api.list_models(
        filter=f"base_model:{base_model}",
        sort="downloads",
        direction=-1,
        limit=cfg.model_limit
    )

    snapshot_download(
        repo_id=base_model,
        local_dir=f"models/{base_model}",
        local_dir_use_symlinks=False
    )

    for model in models:
        print(f"Downloading {model.modelId}...")
        try:
            snapshot_download(
                repo_id=model.modelId,
                local_dir=f"models/{model.modelId}",
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {model.modelId}")
        except Exception as e:
            print(f"Failed to download {model.modelId}: {str(e)}")

    merger = create_merge_instance(cfg)
    print(f"Created merge instance using {cfg.merge.method} method")


if __name__ == "__main__":
    main()