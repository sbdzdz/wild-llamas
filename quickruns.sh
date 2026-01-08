sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true output_dir=outputs/ema_greedy

sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true 'datasets=[math500]' output_dir=outputs/ema_greedy_math500
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true 'datasets=[mmlu]' output_dir=outputs/ema_greedy_mmlu
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true 'datasets=[gpqa]' output_dir=outputs/ema_greedy_gpqa
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true 'datasets=[mmlu_pro]' output_dir=outputs/ema_greedy_mmlu_pro