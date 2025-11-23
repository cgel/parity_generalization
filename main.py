import torch
import functools
from dataclasses import dataclass

def parity(x):
    if x.dim() <= 1:
        return x % 2
    else:
        return x.sum(axis=-1) % 2

@dataclass
class Dataset:
    inputs: torch.Tensor
    @property
    def targets(self):
        if not hasattr(self, '_targets'):
            self._targets = parity(self.inputs)
        return self._targets
    @property
    def n(self): return self.inputs.shape[1]
    @property
    def b(self): return self.inputs.shape[0]

def random_dataset(b, n):
    inputs = torch.randint(0, 2, [b, n])
    return Dataset(inputs)

def full_dataset(n):
    b = 2**n
    r = torch.arange(b)
    x = torch.empty(b, n, dtype=torch.int32)
    for i in range(n):
        x[:, -(i+1)] = (r & 2**i) != 0 
    return Dataset(x)

def downsample_dataset(dataset, new_size):
    """Sample without replacement from the dataset"""
    indices = torch.randperm(dataset.b)[:new_size]
    return Dataset(dataset.inputs[indices])

class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden_dims, n_out):
        super().__init__()
        all_dims = [n_in] + hidden_dims + [n_out] 
        self.layers = []
        for i in range(len(all_dims) - 2):
            self.layers.append(torch.nn.Linear(all_dims[i], all_dims[i + 1]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(all_dims[-2], all_dims[-1]))
        self.layers = torch.nn.Sequential(*self.layers)
    def forward(self, x):
        return self.layers(x.to(torch.float32))

def predictions(model, dataset):
    return model(dataset.inputs).argmax(axis=-1)

def accuracy(model, dataset):
    return (predictions(model, dataset) == dataset.targets).to(torch.float32).mean().item()

n = 8
full_D = full_dataset(n)
D = full_D
# D = downsample_dataset(full_D, full_D.b // 2)
print("dataset size:", D.b)
print("dataset.inputs:\n", D.inputs)
print("dataset.targets:\n", D.targets)
# model = MLP(n, [1024*32], 2)
model = MLP(n, [4096, 4096], 2)

def train(model, dataset, steps, lr=1e-3):
    x, y = dataset.inputs, dataset.targets
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.001)
    for _ in range(steps):
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        loss.backward()
        if _ % (steps // 10) == 0:
            acc = accuracy(model, dataset)
            full_acc = accuracy(model, full_D)
            grad_norm = sum(p.grad.norm() for p in model.parameters())
            print(f'Step {_}: loss: {loss.item():.2e} grad_norm: {grad_norm:.2e} accuracy: {acc*100:.2f}% full_accuracy: {full_acc*100:.2f}%')
        for p in model.parameters():
            p.data -= lr * p.grad
            p.grad.zero_()
train(model, D, 1000, lr=1e-3)
