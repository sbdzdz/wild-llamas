"""Edge case tests for merge operations."""

import torch
from copy import deepcopy
from wildllamas.merge import WeightAveragingIncremental, EMAIncremental


def create_simple_state_dict(value, num_params=2, param_size=(3, 3)):
    """Create a state dictionary with constant values."""
    state_dict = {}
    for i in range(num_params):
        state_dict[f"layer_{i}.weight"] = torch.full(param_size, float(value))
    return state_dict


def test_reject_first_model_after_base():
    """Test rejecting the very first model after base initialization.

    This is a critical edge case: after initializing with base model,
    we attempt to merge the first finetuned model but reject it.
    We should be back to just the base model.
    """
    merger = WeightAveragingIncremental()

    # Initialize with base model (value = 1.0)
    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.0))

    # Attempt to merge first model (value = 10.0) - WILL BE REJECTED
    model_a = create_simple_state_dict(10.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_a)

    # At this point, average should be (1.0 + 10.0) / 2 = 5.5
    assert merger.step_count == 2
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 5.5))

    # REJECT - rollback to base model only
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    # Verify we're back to base model
    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.0))
        assert torch.allclose(merger.current_average[key], torch.full((3, 3), 1.0))

    # Now accept a different model (value = 2.0)
    # Average should be (1.0 + 2.0) / 2 = 1.5
    model_b = create_simple_state_dict(2.0)
    merged_state_dict = merger.update(model_b)

    assert merger.step_count == 2
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.5))

    print("✓ Rejecting first model after base works correctly")


def test_reject_all_models():
    """Test rejecting all attempted merges - should stay at base model."""
    merger = WeightAveragingIncremental()

    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    # Try to merge and reject 5 different models
    for i in range(5):
        previous_merged_state_dict = deepcopy(merged_state_dict)
        previous_step_count = merger.step_count

        model = create_simple_state_dict(float(i + 10))
        merged_state_dict = merger.update(model)

        # Reject
        merged_state_dict = previous_merged_state_dict
        merger.current_average = deepcopy(previous_merged_state_dict)
        merger.step_count = previous_step_count

    # Should still be at base model
    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), 1.0))
        assert torch.allclose(merger.current_average[key], torch.full((3, 3), 1.0))

    print("✓ Rejecting all models keeps base model intact")


def test_ema_edge_case_first_rejection():
    """Test EMA when rejecting the first model after base."""
    beta = 0.9
    merger = EMAIncremental(beta=beta)

    base_model = create_simple_state_dict(1.0)
    merged_state_dict = merger.update(base_model)

    # Attempt and reject first model
    model_a = create_simple_state_dict(10.0)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_a)

    # Reject
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    # Should be back to base
    assert merger.step_count == 1
    for key in base_model.keys():
        assert torch.allclose(merger.current_average[key], torch.full((3, 3), 1.0))

    # Accept second model
    model_b = create_simple_state_dict(2.0)
    merged_state_dict = merger.update(model_b)

    # Should be EMA of base and model_b
    expected = beta * 1.0 + (1 - beta) * 2.0
    assert merger.step_count == 2
    for key in base_model.keys():
        assert torch.allclose(merged_state_dict[key], torch.full((3, 3), expected))

    print("✓ EMA: Rejecting first model after base works correctly")


def test_initialization_matches_first_update():
    """Verify that initializing with a model sets it as the current average."""
    merger = WeightAveragingIncremental()

    model = create_simple_state_dict(5.0)
    result = merger.update(model)

    # First update should return exact copy of input
    assert merger.step_count == 1
    for key in model.keys():
        assert torch.allclose(result[key], model[key])
        assert torch.allclose(merger.current_average[key], model[key])

    # Verify it's a deepcopy, not a reference
    assert result is not model
    assert merger.current_average is not model

    print("✓ Initialization correctly copies first model")


def test_step_count_consistency():
    """Verify step_count is always consistent with number of merged models."""
    merger = WeightAveragingIncremental()

    expected_count = 0
    assert merger.step_count == expected_count

    # Merge base
    base = create_simple_state_dict(1.0)
    merger.update(base)
    expected_count = 1
    assert merger.step_count == expected_count

    # Merge 3 models
    for i in range(3):
        model = create_simple_state_dict(float(i + 2))
        merger.update(model)
        expected_count += 1
        assert merger.step_count == expected_count

    # Attempt and reject a merge
    prev_count = merger.step_count
    prev_state = deepcopy(merger.current_average)

    model = create_simple_state_dict(100.0)
    merger.update(model)

    # Rollback
    merger.current_average = deepcopy(prev_state)
    merger.step_count = prev_count

    assert merger.step_count == expected_count

    # Merge another model
    model = create_simple_state_dict(50.0)
    merger.update(model)
    expected_count += 1
    assert merger.step_count == expected_count

    print("✓ step_count remains consistent through merges and rollbacks")


if __name__ == "__main__":
    test_reject_first_model_after_base()
    test_reject_all_models()
    test_ema_edge_case_first_rejection()
    test_initialization_matches_first_update()
    test_step_count_consistency()
    print("\n✓ All edge case tests passed!")
