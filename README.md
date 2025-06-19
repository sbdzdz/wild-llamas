# Wild Llamas 🦙

Merging finetuned language models in the wild.

## Installation (with uv)

To install the dependencies for this project using uv:

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation)
2. Install dependencies:

   ```bash
   uv sync
   ```

This will set up your virtual environment with all required packages.

## Usage

```bash
uv run python main.py
```

## Configuration

The configuration is stored in the `config` directory.

## Slurm

To run the script on the cluster, use the `slurm/run.sh` script.

```bash
sbatch slurm/run.sh
```

## Requirements

- Python 3.10
