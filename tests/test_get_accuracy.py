from pathlib import Path
import pandas as pd


def get_accuracy(work_dir):
    """Get the average accuracy from evaluation results."""
    work_dir = Path(work_dir)
    subdirs = [d for d in work_dir.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one directory in {work_dir}, found {len(subdirs)}"
        )
    summary_dir = subdirs[0] / "summary"
    csv_files = list(summary_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {summary_dir}")
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df["eval_model"] = pd.to_numeric(df["eval_model"].replace("-", 0), errors="coerce")
    return df["eval_model"].mean()


def test_get_accuracy():
    """Test the get_accuracy function on all models in outputs/opencompass/."""
    opencompass_dir = Path("outputs/opencompass")

    if not opencompass_dir.exists():
        print(f"Directory {opencompass_dir} does not exist")
        return

    print("Model -> Average Accuracy")
    print("-" * 40)

    # Get all model directories
    model_dirs = []
    merged_model_dir = None

    for item in opencompass_dir.iterdir():
        if item.is_dir():
            if item.name == "merged_model":
                merged_model_dir = item
            else:
                model_dirs.append(item)

    if not model_dirs and merged_model_dir is None:
        print("No model directories found")
        return

    model_dirs.sort(key=lambda x: x.name)

    for model_dir in model_dirs:
        model_name = model_dir.name
        try:
            accuracy = get_accuracy(model_dir)
            print(f"{model_name:25} -> {accuracy:.4f}")
        except Exception as e:
            print(f"{model_name:25} -> ERROR: {e}")

    if merged_model_dir is not None:
        print("\nmerged_model:")
        print("-" * 20)

        step_dirs = []
        for item in merged_model_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                step_dirs.append(item)

        if not step_dirs:
            print("  No step directories found")
        else:
            step_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

            for step_dir in step_dirs:
                try:
                    accuracy = get_accuracy(step_dir)
                    print(f"  {step_dir.name:15} -> {accuracy:.4f}")
                except Exception as e:
                    print(f"  {step_dir.name:15} -> ERROR: {e}")


if __name__ == "__main__":
    test_get_accuracy()
