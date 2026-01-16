sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true output_dir=outputs/ema_greedy

sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=10 'datasets=[math500]' output_dir=outputs/ema_greedy_math500
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=10 'datasets=[mmlu]' output_dir=outputs/ema_greedy_mmlu
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=10 'datasets=[gpqa]' output_dir=outputs/ema_greedy_gpqa
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=10 'datasets=[mmlu_pro]' output_dir=outputs/ema_greedy_mmlu_pro

sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=5 evaluate_current=true 'datasets=[math500]' output_dir=outputs/ema_greedy_5_math500
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=10 evaluate_current=true 'datasets=[math500]' output_dir=outputs/ema_greedy_10_math500
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=20 evaluate_current=true 'datasets=[math500]' output_dir=outputs/ema_greedy_20_math500
sbatch slurm/run_ferranti_multi_gpu.sh experiment=ema greedy=true eval_runs=2 greedy_eval_samples=50 evaluate_current=true 'datasets=[math500]' output_dir=outputs/ema_greedy_50_math500