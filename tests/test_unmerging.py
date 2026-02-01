"""Tests for unmerging/rollback logic used in greedy mode.

This module tests the scenario where a merge is accepted, then another merge
is attempted but rejected, requiring rollback to the previous state.
"""

import torch
import pytest
from copy import deepcopy
from wildllamas.merge import WeightAveragingIncremental, EMAIncremental


def create_simple_state_dict(value, num_params=2, param_size=(3, 3)):
    """Create a state dictionary with constant values for easy verification.

    Args:
        value: Constant value to fill all tensors with
        num_params: Number of parameters in the state dict
        param_size: Size of each parameter tensor

    Returns:
        A state dictionary with constant-valued tensors
    """
    state_dict = {}
    for i in range(num_params):
        state_dict[f"layer_{i}.weight"] = torch.full(param_size, float(value))
    return state_dict


def test_weight_averaging_rollback_simple():
    """Test that WeightAveragingIncremental can rollback after a merge.

    Simulates the greedy mode scenario:
    1. Initialize with base model
    2. Accept first merge
    3. Attempt second merge
    4. Reject second merge and rollback
    5. Accept third merge

    Verifies that after rollback, the state is as if the rejected merge never happened.
    """
    merger = WeightAveragingIncremental()

    # Initialize with base model (all weights = 1.0)
    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.0))

    # Accept first model (all weights = 2.0)
    # Expected average: (1.0 + 2.0) / 2 = 1.5
    model_a = create_simple_state_dict(2.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_a)

    assert merger.step_count == 2
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.5))

    # Attempt second model (all weights = 10.0) - WILL BE REJECTED
    # Expected average: (1.0 + 2.0 + 10.0) / 3 = 4.333...
    model_b = create_simple_state_dict(10.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_b)

    assert merger.step_count == 3
    for key in base_model.keys():
        expected_value = (1.0 + 2.0 + 10.0) / 3
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value))

    # REJECT the merge - rollback to previous state
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    # Verify rollback restored correct state
    assert merger.step_count == 2
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.5))
        assert torch.allclose(merger.current_average[key], torch.full((3, 3), 1.5))

    # Accept third model (all weights = 3.0)
    # Expected average: (1.0 + 2.0 + 3.0) / 3 = 2.0
    model_c = create_simple_state_dict(3.0)
    merged_state_dict = merger.update(model_c)

    assert merger.step_count == 3
    for key in base_model.keys():
        expected_value = (1.0 + 2.0 + 3.0) / 3
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value),
                            rtol=1e-5, atol=1e-7)

    print("✓ WeightAveragingIncremental rollback works correctly")


def test_ema_rollback_simple():
    """Test that EMAIncremental can rollback after a merge.

    Simulates the greedy mode scenario with EMA merging.
    """
    beta = 0.9
    merger = EMAIncremental(beta=beta)

    # Initialize with base model (all weights = 1.0)
    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.0))

    # Accept first model (all weights = 2.0)
    # Expected EMA: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    model_a = create_simple_state_dict(2.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_a)

    assert merger.step_count == 2
    for key in base_model.keys():
        expected_value = beta * 1.0 + (1 - beta) * 2.0
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value))

    # Attempt second model (all weights = 10.0) - WILL BE REJECTED
    # Expected EMA: 0.9 * 1.1 + 0.1 * 10.0 = 1.99
    model_b = create_simple_state_dict(10.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_b)

    assert merger.step_count == 3
    for key in base_model.keys():
        expected_value = beta * 1.1 + (1 - beta) * 10.0
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value))

    # REJECT the merge - rollback to previous state
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    # Verify rollback restored correct state
    assert merger.step_count == 2
    for key in base_model.keys():
        expected_value = beta * 1.0 + (1 - beta) * 2.0
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value))
        assert torch.allclose(merger.current_average[key], torch.full((3, 3), expected_value))

    # Accept third model (all weights = 3.0)
    # Expected EMA: 0.9 * 1.1 + 0.1 * 3.0 = 1.29
    model_c = create_simple_state_dict(3.0)
    merged_state_dict = merger.update(model_c)

    assert merger.step_count == 3
    for key in base_model.keys():
        expected_value = beta * 1.1 + (1 - beta) * 3.0
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected_value),
                            rtol=1e-5, atol=1e-7)

    print("✓ EMAIncremental rollback works correctly")


def test_weight_averaging_multiple_rollbacks():
    """Test multiple consecutive rollbacks."""
    merger = WeightAveragingIncremental()

    # Initialize
    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    # Accept model A (weight=2.0) -> avg = 1.5
    model_a = create_simple_state_dict(2.0)
    merged_state_dict = merger.update(model_a)
    assert merger.step_count == 2

    # Try to merge and reject B (weight=10.0)
    for _ in range(3):
        previous_merged_state_dict = deepcopy(merged_state_dict)
        previous_step_count = merger.step_count

        model_b = create_simple_state_dict(10.0)
        merged_state_dict = merger.update(model_b)

        # Reject
        merged_state_dict = previous_merged_state_dict
        merger.current_average = deepcopy(previous_merged_state_dict)
        merger.step_count = previous_step_count

        # Verify still at step 2 with avg = 1.5
        assert merger.step_count == 2
        for key in base_model.keys():
            assert torch.allclose(merger.current_average[key], torch.full((3, 3), 1.5))

    # Now accept model C (weight=3.0) -> avg = 2.0
    model_c = create_simple_state_dict(3.0)
    merged_state_dict = merger.update(model_c)

    assert merger.step_count == 3
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 2.0))

    print("✓ Multiple consecutive rollbacks work correctly")


def test_rollback_no_aliasing_bug():
    """Test that rollback doesn't have aliasing bugs between merged_state_dict and merger.current_average.

    This is critical because merger.update() returns self.current_average, which creates
    an alias. We need to ensure the rollback properly breaks this alias.
    """
    merger = WeightAveragingIncremental()

    # Initialize
    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    # merged_state_dict should be aliased to merger.current_average
    assert merged_state_dict is merger.current_average, \
        "merged_state_dict should be aliased to merger.current_average after update"

    # Accept first model
    model_a = create_simple_state_dict(2.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_a)

    # Still aliased
    assert merged_state_dict is merger.current_average

    # Attempt and reject second model
    model_b = create_simple_state_dict(10.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_b)

    # Rollback
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    # After rollback, they should NOT be aliased
    assert merged_state_dict is not merger.current_average, \
        "After rollback, merged_state_dict and merger.current_average should not be aliased"

    # But they should have the same values
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], merger.current_average[key])

    # Modifying one should not affect the other
    for key in merged_state_dict.keys():
        merged_state_dict[key] += 100.0

    for key in base_model.keys():
        assert not torch.allclose(merged_state_dict[key], merger.current_average[key]), \
            "Modifying merged_state_dict should not affect merger.current_average after rollback"

    print("✓ No aliasing bugs in rollback logic")


def test_weight_averaging_correctness_after_rejection():
    """Verify mathematical correctness: averaging with rejection = averaging without rejected model."""
    # Scenario: merge base, A, B (rejected), C
    # Should equal: merge base, A, C

    # With rejection
    merger1 = WeightAveragingIncremental()
    base = create_simple_state_dict(1.0)
    merger1.update(base)

    model_a = create_simple_state_dict(2.0)
    merger1.update(model_a)

    model_b = create_simple_state_dict(10.0)
    prev_state = deepcopy(merger1.current_average)
    prev_count = merger1.step_count
    merger1.update(model_b)

    # Reject
    merger1.current_average = deepcopy(prev_state)
    merger1.step_count = prev_count

    model_c = create_simple_state_dict(3.0)
    result1 = merger1.update(model_c)

    # Without rejected model
    merger2 = WeightAveragingIncremental()
    merger2.update(base)
    merger2.update(model_a)
    result2 = merger2.update(model_c)

    # Should be identical
    assert merger1.step_count == merger2.step_count == 3
    for key in base.keys():
        assert torch.allclose(result1[key], result2[key], rtol=1e-6, atol=1e-8), \
            f"Results differ for key {key}: {result1[key][0,0]} vs {result2[key][0,0]}"

    print("✓ Mathematical correctness verified: rejection properly excludes model from average")


if __name__ == "__main__":
    test_weight_averaging_rollback_simple()
    test_ema_rollback_simple()
    test_weight_averaging_multiple_rollbacks()
    test_rollback_no_aliasing_bug()
    test_weight_averaging_correctness_after_rejection()
    print("\n✓ All unmerging tests passed!")
