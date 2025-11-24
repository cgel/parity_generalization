import torch


class MLP(torch.nn.Module):
    """Simple feed-forward network with configurable hidden layer widths."""

    def __init__(self, n_in: int, hidden_dims: list[int], n_out: int) -> None:
        super().__init__()
        all_dims = [n_in] + hidden_dims + [n_out]
        self.dims = all_dims
        layers: list[torch.nn.Module] = []
        for i in range(len(all_dims) - 2):
            layers.append(torch.nn.Linear(all_dims[i], all_dims[i + 1], bias=False))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(all_dims[-2], all_dims[-1], bias=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.to(torch.float32))

