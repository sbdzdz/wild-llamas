"""Find, download, merge, and evaluate Llama-3.1-8B-Instruct finetunes."""

from huggingface_hub import HfApi, snapshot_download

def main():
    api = HfApi()
    base_model = "meta-llama/Llama-3.1-8B-Instruct"

    models = api.list_models(
        filter=f"base_model:{base_model}",
        sort="downloads",
        direction=-1,
        limit=10
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


if __name__ == "__main__":
    main()