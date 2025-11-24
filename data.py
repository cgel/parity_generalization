import torch
from dataclasses import dataclass


def parity(x: torch.Tensor) -> torch.Tensor:
    """Compute parity bit for a tensor along the last dimension."""
    x = torch.where(x == 1, 1, 0) # in case of negative features convert -1 -> 0
    if x.dim() <= 1:
        return x % 2
    return x.sum(dim=-1) % 2

def random_target(x: torch.Tensor) -> torch.Tensor:
    return torch.randint(0, 2, x.shape[:-1], device=x.device)

def __parity(x: torch.Tensor) -> torch.Tensor:
    """Compute parity bit for a tensor along the last dimension."""
    x = x.clone()
    x[:, :4] = 0
    return x.sum(dim=-1) % 2

# target_fn = random_target
target_fn = parity
# target_fn = __parity

@dataclass
class Dataset:
    cpu_inputs: torch.Tensor

    @property
    def inputs(self) -> torch.Tensor:
        if not hasattr(self, "_inputs"):
            self._inputs = self.cpu_inputs.cuda()
        return self._inputs

    @property
    def targets(self) -> torch.Tensor:
        if not hasattr(self, "_targets"):
            self._targets = target_fn(self.inputs)
        return self._targets

    @property
    def n(self) -> int:
        return self.inputs.shape[1]

    @property
    def b(self) -> int:
        return self.inputs.shape[0]


def full_dataset(n_bits: int, negative_values: bool = True) -> Dataset:
    total = 2 ** n_bits
    ranges = torch.arange(total)
    x = torch.empty(total, n_bits, dtype=torch.int32)
    for i in range(n_bits):
        x[:, -(i + 1)] = (ranges & 2 ** i) != 0
    if negative_values:
        x = torch.where(x == 1, 1, -1)
    return Dataset(x)


def train_test_split(dataset: Dataset, test_size: int) -> tuple[Dataset, Dataset]:
    train_size = dataset.b - test_size
    indices = torch.randperm(dataset.b)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_set = Dataset(dataset.inputs[train_indices])
    test_set = Dataset(dataset.inputs[test_indices])
    return train_set, test_set

