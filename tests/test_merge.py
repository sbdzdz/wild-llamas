import torch
import pytest
from wildllamas.merge import WeightAveraging, WeightAveragingIncremental


def create_random_state_dict(num_params=3, param_size=(10, 10), seed=None):
    """Create a random state dictionary for testing.

    Args:
        num_params: Number of parameters in the state dict
        param_size: Size of each parameter tensor
        seed: Random seed for reproducibility

    Returns:
        A state dictionary with random tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    state_dict = {}
    for i in range(num_params):
        state_dict[f"layer_{i}.weight"] = torch.randn(param_size)

    return state_dict


def test_weight_averaging_incremental_vs_batch():
    """Test that WeightAveragingIncremental produces the same result as WeightAveraging.

    This test creates multiple state dicts, merges them using WeightAveraging (batch),
    and compares the result to incrementally merging them one-by-one using
    WeightAveragingIncremental.
    """
    num_models = 5
    state_dicts = [
        create_random_state_dict(num_params=3, param_size=(10, 10), seed=i)
        for i in range(num_models)
    ]

    batch_merger = WeightAveraging()
    batch_result = batch_merger.merge(state_dicts)

    incremental_merger = WeightAveragingIncremental()
    for state_dict in state_dicts:
        incremental_result = incremental_merger.update(state_dict)

    assert incremental_merger.step_count == num_models, \
        f"Expected step_count to be {num_models}, got {incremental_merger.step_count}"

    for key in batch_result.keys():
        assert key in incremental_result, f"Key {key} missing in incremental result"

        batch_tensor = batch_result[key]
        incremental_tensor = incremental_result[key]

        assert torch.allclose(batch_tensor, incremental_tensor, rtol=1e-5, atol=1e-7), \
            f"Mismatch in parameter {key}: max diff = {(batch_tensor - incremental_tensor).abs().max()}"

    print(f"✓ Batch and incremental merging produce identical results for {num_models} models")


def test_weight_averaging_incremental_different_sizes():
    """Test incremental averaging with different numbers of models."""
    for num_models in [2, 3, 10]:
        state_dicts = [
            create_random_state_dict(num_params=2, param_size=(5, 5), seed=i)
            for i in range(num_models)
        ]

        batch_merger = WeightAveraging()
        batch_result = batch_merger.merge(state_dicts)

        incremental_merger = WeightAveragingIncremental()
        for state_dict in state_dicts:
            incremental_result = incremental_merger.update(state_dict)

        for key in batch_result.keys():
            assert torch.allclose(batch_result[key], incremental_result[key], rtol=1e-5, atol=1e-7), \
                f"Mismatch for {num_models} models at parameter {key}"

        print(f"✓ Test passed for {num_models} models")


def test_weight_averaging_incremental_step_count():
    """Test that step_count is correctly tracked."""
    incremental_merger = WeightAveragingIncremental()

    assert incremental_merger.step_count == 0, "Initial step_count should be 0"

    num_updates = 7
    for i in range(num_updates):
        state_dict = create_random_state_dict(seed=i)
        incremental_merger.update(state_dict)
        assert incremental_merger.step_count == i + 1, \
            f"After update {i+1}, step_count should be {i+1}, got {incremental_merger.step_count}"

    print(f"✓ Step count correctly tracked through {num_updates} updates")


def test_weight_averaging_incremental_validation():
    """Test that incremental merger validates state dict keys."""
    incremental_merger = WeightAveragingIncremental()

    state_dict_1 = create_random_state_dict(num_params=3, seed=0)
    incremental_merger.update(state_dict_1)

    state_dict_2 = create_random_state_dict(num_params=2, seed=1)

    with pytest.raises(ValueError, match="Cannot merge state dictionaries with unequal keys"):
        incremental_merger.update(state_dict_2)

    print("✓ Validation correctly rejects mismatched keys")


if __name__ == "__main__":
    test_weight_averaging_incremental_vs_batch()
    test_weight_averaging_incremental_different_sizes()
    test_weight_averaging_incremental_step_count()
    test_weight_averaging_incremental_validation()
    print("\n✓ All tests passed!")
