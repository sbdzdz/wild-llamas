sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true output_dir=outputs/ema_greedy

# math
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema_holdout 'selection_datasets=[math500]' 'validation_datasets=[gsm8k]' output_dir=outputs/ema_holdout_math

# knowledge
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema_holdout 'selection_datasets=[mmlu]' 'validation_datasets=[mmlu_pro]' output_dir=outputs/ema_holdout_knowledge