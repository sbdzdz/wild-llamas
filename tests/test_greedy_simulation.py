"""Simulate greedy merging as done in main.py to verify no bugs in realistic scenarios."""

import torch
from copy import deepcopy
from wildllamas.merge import EMAIncremental, WeightAveragingIncremental


def create_state_dict_with_seed(seed, num_params=2, param_size=(3, 3)):
    """Create a random state dictionary with a given seed."""
    torch.manual_seed(seed)
    state_dict = {}
    for i in range(num_params):
        state_dict[f"layer_{i}.weight"] = torch.randn(param_size)
    return state_dict


def compute_average_value(state_dict):
    """Compute the average value across all parameters (for testing purposes)."""
    total = 0.0
    count = 0
    for tensor in state_dict.values():
        total += tensor.sum().item()
        count += tensor.numel()
    return total / count


def test_ema_greedy_simulation():
    """Simulate the greedy merging flow from main.py with EMA.

    This test verifies that:
    1. Accepting models works correctly
    2. Rejecting models properly rolls back state
    3. After rejection, subsequent merges are correct
    4. The final state is mathematically consistent
    """
    beta = 0.5
    merger = EMAIncremental(beta=beta)

    # Initialize with base model (seed 0)
    base_model = create_state_dict_with_seed(0)
    merged_state_dict = merger.update(base_model)

    base_avg = compute_average_value(merged_state_dict)
    print(f"Step 0 (base): avg={base_avg:.4f}, step_count={merger.step_count}")

    assert merger.step_count == 1

    # Try to merge model 1 (seed 1) - ACCEPT
    model_1 = create_state_dict_with_seed(1)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_1)
    step1_avg = compute_average_value(merged_state_dict)
    print(f"Step 1 (accepted): avg={step1_avg:.4f}, step_count={merger.step_count}")

    assert merger.step_count == 2

    # Try to merge model 2 (seed 2) - REJECT
    model_2 = create_state_dict_with_seed(2)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_2)
    rejected_avg = compute_average_value(merged_state_dict)
    print(f"Step 2 (rejected): avg={rejected_avg:.4f}, step_count={merger.step_count}")

    # REJECT - rollback
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    rollback_avg = compute_average_value(merged_state_dict)
    print(f"After rollback: avg={rollback_avg:.4f}, step_count={merger.step_count}")

    assert merger.step_count == 2
    assert torch.allclose(
        torch.tensor(rollback_avg), torch.tensor(step1_avg), rtol=1e-5
    ), f"Rollback didn't restore correct state: {rollback_avg} vs {step1_avg}"

    # Try to merge model 3 (seed 3) - ACCEPT
    model_3 = create_state_dict_with_seed(3)
    merged_state_dict = merger.update(model_3)
    step2_avg = compute_average_value(merged_state_dict)
    print(f"Step 2 (accepted): avg={step2_avg:.4f}, step_count={merger.step_count}")

    assert merger.step_count == 3

    # Verify mathematical correctness: step2 should be EMA of step1 and model_3
    model_3_avg = compute_average_value(model_3)
    expected_step2_avg = beta * step1_avg + (1 - beta) * model_3_avg

    assert torch.allclose(
        torch.tensor(step2_avg), torch.tensor(expected_step2_avg), rtol=1e-5
    ), f"Step 2 average {step2_avg} doesn't match expected {expected_step2_avg}"

    # Try to merge model 4 (seed 4) - ACCEPT
    model_4 = create_state_dict_with_seed(4)
    merged_state_dict = merger.update(model_4)
    step3_avg = compute_average_value(merged_state_dict)
    print(f"Step 3 (accepted): avg={step3_avg:.4f}, step_count={merger.step_count}")

    # Verify mathematical correctness: step3 should be EMA of step2 and model_4
    model_4_avg = compute_average_value(model_4)
    expected_step3_avg = beta * step2_avg + (1 - beta) * model_4_avg

    assert torch.allclose(
        torch.tensor(step3_avg), torch.tensor(expected_step3_avg), rtol=1e-5
    ), f"Step 3 average {step3_avg} doesn't match expected {expected_step3_avg}"

    print("\n✓ EMA greedy simulation passed all checks")


def test_weight_averaging_greedy_simulation():
    """Simulate the greedy merging flow with WeightAveragingIncremental."""
    merger = WeightAveragingIncremental()

    # Initialize with base model
    base_model = create_state_dict_with_seed(10)
    merged_state_dict = merger.update(base_model)

    assert merger.step_count == 1

    # Accept model 1
    model_1 = create_state_dict_with_seed(11)
    merged_state_dict = merger.update(model_1)
    step1_avg = compute_average_value(merged_state_dict)

    assert merger.step_count == 2

    # Try model 2 - REJECT
    model_2 = create_state_dict_with_seed(12)
    previous_merged_state_dict = deepcopy(merged_state_dict)
    previous_step_count = merger.step_count

    merged_state_dict = merger.update(model_2)

    # Rollback
    merged_state_dict = previous_merged_state_dict
    merger.current_average = deepcopy(previous_merged_state_dict)
    merger.step_count = previous_step_count

    rollback_avg = compute_average_value(merged_state_dict)

    assert merger.step_count == 2
    assert torch.allclose(torch.tensor(rollback_avg), torch.tensor(step1_avg), rtol=1e-5)

    # Accept model 3
    model_3 = create_state_dict_with_seed(13)
    merged_state_dict = merger.update(model_3)

    assert merger.step_count == 3

    # Verify mathematical correctness
    # Should equal (base + model_1 + model_3) / 3
    base_avg = compute_average_value(base_model)
    model_1_avg = compute_average_value(model_1)
    model_3_avg = compute_average_value(model_3)

    expected_avg = (base_avg + model_1_avg + model_3_avg) / 3
    actual_avg = compute_average_value(merged_state_dict)

    assert torch.allclose(
        torch.tensor(actual_avg), torch.tensor(expected_avg), rtol=1e-5
    ), f"Final average {actual_avg} doesn't match expected {expected_avg}"

    print("✓ WeightAveragingIncremental greedy simulation passed all checks")


def test_multiple_consecutive_rejections():
    """Test rejecting multiple models in a row (like in greedy mode)."""
    beta = 0.9
    merger = EMAIncremental(beta=beta)

    # Initialize
    base_model = create_state_dict_with_seed(20)
    merged_state_dict = merger.update(base_model)
    base_avg = compute_average_value(merged_state_dict)

    # Accept first model
    model_1 = create_state_dict_with_seed(21)
    merged_state_dict = merger.update(model_1)
    step1_avg = compute_average_value(merged_state_dict)

    # Reject 5 models in a row
    for i in range(5):
        previous_merged_state_dict = deepcopy(merged_state_dict)
        previous_step_count = merger.step_count

        bad_model = create_state_dict_with_seed(100 + i)
        merged_state_dict = merger.update(bad_model)

        # Rollback
        merged_state_dict = previous_merged_state_dict
        merger.current_average = deepcopy(previous_merged_state_dict)
        merger.step_count = previous_step_count

        # Verify we're still at step 2
        assert merger.step_count == 2
        current_avg = compute_average_value(merged_state_dict)
        assert torch.allclose(
            torch.tensor(current_avg), torch.tensor(step1_avg), rtol=1e-5
        ), f"After rejection {i+1}, state changed unexpectedly"

    # Now accept another model
    model_2 = create_state_dict_with_seed(22)
    merged_state_dict = merger.update(model_2)

    assert merger.step_count == 3

    # Verify correctness: should be EMA of step1 and model_2
    model_2_avg = compute_average_value(model_2)
    expected = beta * step1_avg + (1 - beta) * model_2_avg
    actual = compute_average_value(merged_state_dict)

    assert torch.allclose(torch.tensor(actual), torch.tensor(expected), rtol=1e-5)

    print(f"✓ Survived {5} consecutive rejections with correct state")


def test_edge_case_all_rejections():
    """Test case where all models after base are rejected."""
    merger = EMAIncremental(beta=0.5)

    base_model = create_state_dict_with_seed(30)
    merged_state_dict = merger.update(base_model)
    base_avg = compute_average_value(merged_state_dict)

    # Try to merge 10 models, reject all
    for i in range(10):
        previous_merged_state_dict = deepcopy(merged_state_dict)
        previous_step_count = merger.step_count

        model = create_state_dict_with_seed(31 + i)
        merged_state_dict = merger.update(model)

        # Reject
        merged_state_dict = previous_merged_state_dict
        merger.current_average = deepcopy(previous_merged_state_dict)
        merger.step_count = previous_step_count

    # Should still be at base
    assert merger.step_count == 1
    final_avg = compute_average_value(merged_state_dict)
    assert torch.allclose(torch.tensor(final_avg), torch.tensor(base_avg), rtol=1e-5)

    print("✓ All-rejection scenario works correctly")


if __name__ == "__main__":
    test_ema_greedy_simulation()
    test_weight_averaging_greedy_simulation()
    test_multiple_consecutive_rejections()
    test_edge_case_all_rejections()
    print("\n✓ All greedy simulation tests passed!")
