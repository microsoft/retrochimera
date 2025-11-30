import torch
from torch.optim.lr_scheduler import LambdaLR


def count_occurrences(indices: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of indices into counts of how many each value occurs.

    >>> count_occurrences(torch.as_tensor([0, 1, 1, 3]))
    tensor([1, 2, 0, 1])
    """
    out = torch.zeros(indices.max() + 1, device=indices.device, dtype=indices.dtype)
    return torch.scatter_add(out, 0, indices, torch.ones_like(indices))


def split_select(source: torch.Tensor, indices: torch.Tensor) -> list[torch.Tensor]:
    """Split a given tensor into chunks denoted by a separate set of indices.

    >>> split_select(torch.arange(6), torch.as_tensor([0, 0, 1, 2, 2, 2]))
    (tensor([0, 1]), tensor([2]), tensor([3, 4, 5]))
    """
    return torch.split(source, tuple(count_occurrences(indices).tolist()))


def tensor_to_list(t: torch.Tensor):
    return t.cpu().detach().numpy().tolist()


def get_sorted_ids_and_probs(logits: torch.Tensor, k: int) -> list[tuple[list[int], list[float]]]:
    """Given raw batched logits return `k` most likely indices with their probabilities."""
    # Compute probabilities and extract top indices.
    batch_rule_probs = torch.nn.functional.softmax(logits, dim=1)
    sorted_batch_rule_ids = batch_rule_probs.argsort(dim=1, descending=True)

    # Limit to top rules before moving to CPU and converting to `RulePrediction` objects.
    sorted_batch_rule_ids = sorted_batch_rule_ids[:, :k]
    sorted_batch_rule_probs = batch_rule_probs.gather(index=sorted_batch_rule_ids, dim=1)

    sorted_batch_rule_ids = tensor_to_list(sorted_batch_rule_ids)
    sorted_batch_rule_probs = tensor_to_list(sorted_batch_rule_probs)

    return list(zip(sorted_batch_rule_ids, sorted_batch_rule_probs))


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]
