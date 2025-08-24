"""
Script to safely remove model directories listed in skipped_models.csv
"""

import csv
import os
import shutil
import sys
from pathlib import Path

WORK_DIR = Path(os.environ.get("WORK"))
MODELS_DIR = WORK_DIR / "wild-llamas" / "models"
REPO_DIR = Path(__file__).parent.parent.resolve()
SKIPPED_CSV = REPO_DIR / "skipped_models.csv"


def read_skipped_models(csv_path):
    """Read model IDs from the CSV file."""
    skipped_models = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model_id = row.get("model_id", "").strip()
                if model_id:  # Skip empty lines
                    skipped_models.append(model_id)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    return skipped_models


def model_id_to_directory_name(model_id):
    """Convert model ID format (org/model) to directory format (org--model)."""
    return model_id.replace("/", "--")


def find_existing_model_directories(models_dir, skipped_models):
    """Find which skipped models actually exist as directories."""
    existing_dirs = []
    missing_dirs = []

    if not models_dir.exists():
        print(f"Error: Models directory {models_dir} does not exist")
        sys.exit(1)

    for model_id in skipped_models:
        dir_name = model_id_to_directory_name(model_id)
        full_path = models_dir / dir_name

        if full_path.exists() and full_path.is_dir():
            existing_dirs.append((model_id, dir_name, full_path))
        else:
            missing_dirs.append((model_id, dir_name))

    return existing_dirs, missing_dirs


def get_directory_size(directory):
    """Get the total size of a directory in bytes."""
    total_size = 0
    try:
        for filepath in directory.rglob("*"):
            if filepath.is_file():
                try:
                    total_size += filepath.stat().st_size
                except (OSError, IOError):
                    pass
    except (OSError, IOError):
        return 0
    return total_size


def format_size(size_bytes):
    """Format size in human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def main():
    print("=== Skipped Models Removal Script ===")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Reading from: {SKIPPED_CSV}")
    print()

    print("Reading skipped models from CSV...")
    skipped_models = read_skipped_models(SKIPPED_CSV)
    print(f"Found {len(skipped_models)} models in CSV file")

    print("\nChecking which models exist as directories...")
    existing_dirs, missing_dirs = find_existing_model_directories(
        MODELS_DIR, skipped_models
    )

    print("\nSummary:")
    print(f"  Models in CSV: {len(skipped_models)}")
    print(f"  Existing directories: {len(existing_dirs)}")
    print(f"  Missing directories: {len(missing_dirs)}")

    if missing_dirs:
        print(f"\nModels not found as directories ({len(missing_dirs)}):")
        for model_id, dir_name in missing_dirs[:10]:  # Show first 10
            print(f"  - {model_id} (would be: {dir_name})")
        if len(missing_dirs) > 10:
            print(f"  ... and {len(missing_dirs) - 10} more")

    if not existing_dirs:
        print("\nNo model directories found to delete. Exiting.")
        return

    print(f"\nDirectories that will be DELETED ({len(existing_dirs)}):")
    total_size = 0
    for model_id, dir_name, full_path in existing_dirs:
        dir_size = get_directory_size(full_path)
        total_size += dir_size
        print(f"  - {dir_name} ({format_size(dir_size)})")

    print(f"\nTotal size to be freed: {format_size(total_size)}")

    print("\n" + "=" * 60)
    print("WARNING: This will permanently delete the directories above!")
    print("=" * 60)

    response = (
        input("\nDo you want to proceed with deletion? (yes/no): ").strip().lower()
    )

    if response not in ["yes", "y"]:
        print("Operation cancelled.")
        return

    response2 = input(
        f"Are you absolutely sure you want to delete {len(existing_dirs)} directories? (DELETE/cancel): "
    ).strip()

    if response2 != "DELETE":
        print("Operation cancelled.")
        return

    print("\nDeleting directories...")
    deleted_count = 0
    failed_deletions = []

    for model_id, dir_name, full_path in existing_dirs:
        try:
            print(f"  Deleting: {dir_name}")
            shutil.rmtree(full_path)
            deleted_count += 1
        except Exception as e:
            print(f"  ERROR deleting {dir_name}: {e}")
            failed_deletions.append((dir_name, str(e)))

    print("\n=== Deletion Summary ===")
    print(f"Successfully deleted: {deleted_count} directories")
    print(f"Failed deletions: {len(failed_deletions)}")

    if failed_deletions:
        print("\nFailed deletions:")
        for dir_name, error in failed_deletions:
            print(f"  - {dir_name}: {error}")

    print(f"\nApproximate space freed: {format_size(total_size)}")
    print("Done!")


if __name__ == "__main__":
    main()
