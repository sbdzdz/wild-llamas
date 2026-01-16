# Proposed Fixes for Greedy Merging

## Fix 1: Held-Out Validation Set

**Problem**: Greedy selection may overfit to the evaluation benchmarks.

**Fix**: Split available benchmarks into two groups:

- **Selection set**: Used for accept/reject decisions during merging (e.g., MMLU, Math-500)
- **Validation set**: Never seen during selection, only used for periodic checkpointing (e.g., MMLU-Pro, GPQA, GSM8K)

## Fix 2: Multi-Ordering Ensemble

**Problem**: Results depend heavily on the arbitrary model ordering.

**Fix**: Run the greedy pipeline with K different random orderings (e.g., K=3-5). Then either:

- **Option A**: Take the single best final model across all runs
- **Option B**: Merge the K final models together (meta-merge) using simple weight averaging
- **Option C**: Accept a model into the final merge only if it was accepted in at least M out of K runs (consensus filtering)

## Fix 3: Escaping Local Optima

**Problem**: Strict greedy rejection prevents discovering synergistic combinations.

**Fix**: Two alternative approaches:

### Option A: Epsilon-Greedy with Periodic Pruning

1. **Epsilon-greedy acceptance**: Accept any merge that doesn't decrease accuracy by more than ε (e.g., 1%). This allows marginally-worse models through on the chance they synergize with future models.
2. **Periodic pruning**: Every N merges, rebuild the merged model from scratch using only the top-performing subset of accepted models (based on ablation — measure accuracy drop when each model is removed).

### Option B: Beam Search

1. **Maintain K parallel trajectories**: Instead of a single merged model, keep the top-K merge trajectories (e.g., K=3) ranked by accuracy.
2. **Branch on each candidate**: For each new model, try merging it into all K trajectories. This produces up to 2K candidates (K with the new model, K without).
3. **Prune to top-K**: Evaluate all candidates and keep only the top-K for the next iteration.