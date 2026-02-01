import torch
import pytest
from wildllamas.merge import WeightAveraging, WeightAveragingIncremental, EMAIncremental


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


def test_ema_incremental_basic():
    """Test basic EMA incremental merging functionality."""
    beta = 0.9
    ema_merger = EMAIncremental(beta=beta)

    state_dict_1 = create_random_state_dict(num_params=2, param_size=(5, 5), seed=0)
    result_1 = ema_merger.update(state_dict_1)

    for key in state_dict_1.keys():
        assert torch.allclose(result_1[key], state_dict_1[key]), \
            f"First update should be identical to input for key {key}"

    assert ema_merger.step_count == 1

    state_dict_2 = create_random_state_dict(num_params=2, param_size=(5, 5), seed=1)
    result_2 = ema_merger.update(state_dict_2)

    for key in state_dict_1.keys():
        expected = beta * state_dict_1[key] + (1 - beta) * state_dict_2[key]
        assert torch.allclose(result_2[key], expected, rtol=1e-5, atol=1e-7), \
            f"EMA formula not applied correctly for key {key}"

    assert ema_merger.step_count == 2

    print(f"✓ EMA incremental basic functionality works correctly with beta={beta}")


def test_ema_incremental_multiple_updates():
    """Test EMA with multiple sequential updates."""
    beta = 0.8
    ema_merger = EMAIncremental(beta=beta)

    state_dicts = [
        create_random_state_dict(num_params=2, param_size=(3, 3), seed=i)
        for i in range(5)
    ]

    current = None
    for i, state_dict in enumerate(state_dicts):
        result = ema_merger.update(state_dict)

        if i == 0:
            current = {k: v.clone() for k, v in result.items()}
        else:
            expected = {k: beta * current[k] + (1 - beta) * state_dict[k] for k in state_dict.keys()}
            for key in state_dict.keys():
                assert torch.allclose(result[key], expected[key], rtol=1e-5, atol=1e-7), \
                    f"EMA formula mismatch at update {i+1} for key {key}"
            current = expected

        assert ema_merger.step_count == i + 1

    print(f"✓ EMA incremental with {len(state_dicts)} updates works correctly")


def test_ema_incremental_different_betas():
    """Test EMA with different beta values."""
    state_dicts = [
        create_random_state_dict(num_params=2, param_size=(3, 3), seed=i)
        for i in range(3)
    ]

    for beta in [0.5, 0.9, 0.99]:
        ema_merger = EMAIncremental(beta=beta)

        for state_dict in state_dicts:
            ema_merger.update(state_dict)

        assert ema_merger.step_count == len(state_dicts)

    print("✓ EMA incremental works with different beta values")


def test_ema_incremental_validation():
    """Test that EMA merger validates state dict keys."""
    ema_merger = EMAIncremental(beta=0.9)

    state_dict_1 = create_random_state_dict(num_params=3, seed=0)
    ema_merger.update(state_dict_1)

    state_dict_2 = create_random_state_dict(num_params=2, seed=1)

    with pytest.raises(ValueError, match="Cannot merge state dictionaries with unequal keys"):
        ema_merger.update(state_dict_2)

    print("✓ EMA validation correctly rejects mismatched keys")


def test_unmerge_weight_averaging():
    """Test unmerge logic: after rejecting a merge, result equals merging without that model.

    Simulates main.py greedy rejection flow for WeightAveragingIncremental.
    """
    from copy import deepcopy

    base = create_random_state_dict(num_params=3, param_size=(10, 10), seed=0)
    model1 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=1)
    model2 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=2)
    model3 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=3)

    merger = WeightAveragingIncremental()
    merger.update(base)

    merged_state_dict = merger.update(model1)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merger.update(model2)

    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    merged_state_dict = merger.update(model3)

    expected = WeightAveraging().merge([base, model1, model3])
    for key in expected.keys():
        assert torch.allclose(
            merged_state_dict[key], expected[key], rtol=1e-5, atol=1e-7
        ), f"Unmerge mismatch at {key}"


def test_unmerge_ema():
    """Test unmerge logic for EMA: after rejecting a merge, subsequent merges are correct.

    Simulates main.py greedy rejection flow for EMAIncremental.
    """
    from copy import deepcopy

    beta = 0.9
    base = create_random_state_dict(num_params=3, param_size=(10, 10), seed=0)
    model1 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=1)
    model2 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=2)
    model3 = create_random_state_dict(num_params=3, param_size=(10, 10), seed=3)

    merger = EMAIncremental(beta=beta)
    merger.update(base)

    merged_state_dict = merger.update(model1)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merger.update(model2)

    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    merged_state_dict = merger.update(model3)

    ema_12 = {k: beta * base[k] + (1 - beta) * model1[k] for k in base.keys()}
    expected = {k: beta * ema_12[k] + (1 - beta) * model3[k] for k in base.keys()}
    for key in expected.keys():
        assert torch.allclose(
            merged_state_dict[key], expected[key], rtol=1e-5, atol=1e-7
        ), f"EMA unmerge mismatch at {key}"


if __name__ == "__main__":
    test_weight_averaging_incremental_vs_batch()
    test_weight_averaging_incremental_different_sizes()
    test_weight_averaging_incremental_step_count()
    test_weight_averaging_incremental_validation()
    test_ema_incremental_basic()
    test_ema_incremental_multiple_updates()
    test_ema_incremental_different_betas()
    test_ema_incremental_validation()
    test_unmerge_weight_averaging()
    test_unmerge_ema()
    print("\n✓ All tests passed!")
