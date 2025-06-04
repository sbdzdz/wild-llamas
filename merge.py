from abc import ABC, abstractmethod
import torch
from omegaconf import DictConfig

def create_merge_instance(cfg: DictConfig):
    """Creates and returns an instance of the merge class based on the provided configuration."""
    method_to_class = {
        "weight_averaging": WeightAveraging,
        "task_arithmetic": TaskArithmetic,
        "ties": TIES,
    }

    method = cfg.merge.method
    if method not in method_to_class:
        raise KeyError(f"Unknown merge method: {method}")

    method_cfg = cfg.merge.get(method, {})

    return method_to_class[method](**method_cfg)


class BaseMerge(ABC):
    """Abstract base class for merging multiple model state dictionaries."""

    def __init__(self):
        pass

    def __call__(self, state_dicts, zero_shot=None):
        """Validate and merge a list of state dictionaries."""
        self.validate_state_dicts(state_dicts)
        return self.merge(state_dicts, zero_shot)

    @abstractmethod
    def merge(self, state_dicts, zero_shot=None):
        raise NotImplementedError("Merge method not implemented.")

    def validate_state_dicts(self, state_dicts):
        """Check if all dictionaries have the same keys."""
        keys = set(state_dicts[0].keys())
        for state_dict in state_dicts:
            if set(state_dict.keys()) != keys:
                raise ValueError("Cannot merge state dictionaries with unequal keys.")

    def move_state_dicts_to_cpu(self, state_dicts):
        """Move the state dictionaries to the CPU."""
        return [
            {k: v.cpu() for k, v in state_dict.items()} for state_dict in state_dicts
        ]


class BaseTaskVectorMerge(BaseMerge):
    """Base class for task vector based merging methods."""

    REMOVE_KEYS = [
        "transformer.encoder.embed_tokens.weight",
        "transformer.decoder.embed_tokens.weight",
    ]

    def __init__(self, scaling_factor=1):
        super().__init__()
        self.scaling_factor = scaling_factor

    def state_dict_to_vector(self, state_dict, remove_keys):
        """Convert a state dictionary to a flattened vector."""
        shared_state_dict = {
            k: v for k, v in state_dict.items() if k not in remove_keys
        }
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for value in shared_state_dict.values()]
        )

    def vector_to_state_dict(self, vector, state_dict, remove_keys):
        """Convert a flattened vector back to a state dictionary."""
        reference_dict = {k: v for k, v in state_dict.items() if k not in remove_keys}
        torch.nn.utils.vector_to_parameters(vector, reference_dict.values())

        if "transformer.shared.weight" in reference_dict:
            shared_weight = reference_dict["transformer.shared.weight"]
            for key in remove_keys:
                reference_dict[key] = shared_weight

        return reference_dict

    def compute_task_vectors(self, state_dicts, zero_shot):
        """Compute task vectors by subtracting zero-shot from fine-tuned vectors."""
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, self.REMOVE_KEYS)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, self.REMOVE_KEYS)
        task_vectors = ft_vectors - zs_vector

        return task_vectors, zs_vector


class WeightAveraging(BaseMerge):
    """Weight averaging merging technique for multiple model state dictionaries."""

    def merge(self, state_dicts, zero_shot=None):
        """Average the parameters of multiple state dictionaries."""
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)
        weight = 1.0 / len(state_dicts)

        return {
            key: sum(weight * state_dict[key] for state_dict in state_dicts)
            for key in state_dicts[0].keys()
        }


class TaskArithmetic(BaseTaskVectorMerge):
    """Task-Arithmetic merging technique."""

    def __init__(self, scaling_factor=1):
        super().__init__(scaling_factor)

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries using the Task-Arithmetic technique."""
        task_vectors, zs_vector = self.compute_task_vectors(state_dicts, zero_shot)
        merged_task_vectors = torch.sum(task_vectors, dim=0)

        merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, self.REMOVE_KEYS)


class TIES(BaseTaskVectorMerge):
    """TIES merging technique."""

    def __init__(
        self,
        scaling_factor=1,
        prune_percentile=0.2,
    ):
        super().__init__(scaling_factor)
        self.prune_percentile = prune_percentile

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries using the TIES technique."""
        task_vectors, zs_vector = self.compute_task_vectors(state_dicts, zero_shot)
        merged_task_vectors = self.ties_merging(task_vectors)

        merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, self.REMOVE_KEYS)

    def ties_merging(self, task_vectors):
        """Perform TIES merging on flattened task checkpoints."""
        task_vectors = self.sparsify(task_vectors)
        signs = self.resolve_sign(task_vectors)
        return self.disjoint_merge(task_vectors, signs)

    def sparsify(self, task_vectors):
        """Apply a top-k mask to the input tensor."""
        original_shape = task_vectors.shape

        if task_vectors.dim() == 1:
            task_vectors = task_vectors.unsqueeze(0)

        num_elements = task_vectors.shape[1]
        k = int(num_elements * self.prune_percentile)
        kth_values, _ = task_vectors.abs().kthvalue(k, dim=1, keepdim=True)
        mask = task_vectors.abs() >= kth_values
        mask = (
            mask.squeeze() if original_shape == task_vectors.squeeze().shape else mask
        )
        return task_vectors * mask

    def resolve_sign(self, tensor):
        """Resolve the sign of the input tensor."""
        sign_to_mult = torch.sign(tensor.sum(dim=0))
        sign_to_mult[sign_to_mult == 0] = torch.sign(sign_to_mult.sum())
        return sign_to_mult

    def disjoint_merge(self, tensor, signs):
        """Perform disjoint merging on the input tensor."""
        rows_to_keep = torch.where(signs.unsqueeze(0) > 0, tensor > 0, tensor < 0)
        selected_entries = tensor * rows_to_keep

        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )

        return disjoint_aggs
