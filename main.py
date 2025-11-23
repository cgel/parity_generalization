import torch
import functools
from dataclasses import dataclass
import matplotlib.pyplot as plt

def parity(x):
    if x.dim() <= 1:
        return x % 2
    else:
        return x.sum(axis=-1) % 2

@dataclass
class Dataset:
    cpu_inputs: torch.Tensor
    @property
    def inputs(self):
        return self.cpu_inputs.cuda()
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

def train_test_split(dataset, test_size):
    """Split the dataset into training and testing sets"""
    train_size = dataset.b - test_size
    indices = torch.randperm(dataset.b)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_set = Dataset(dataset.inputs[train_indices])
    test_set = Dataset(dataset.inputs[test_indices])
    return train_set, test_set
    
class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden_dims, n_out):
        super().__init__()
        all_dims = [n_in] + hidden_dims + [n_out] 
        self.dims = all_dims
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

def measure_loss(model, dataset):
    return torch.nn.CrossEntropyLoss()(model(dataset.inputs), dataset.targets)

def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def measure_grad_norm(model):
    return sum(p.grad.norm() for p in model.parameters())


def train(model, train_set, test_set, full_set, steps, lr, report_every):
    steps_recorded = []
    history = {
        "train": {"loss": [], "grad": [], "acc": []},
        "test": {"loss": [], "grad": [], "acc": []},
        "full": {"loss": [], "grad": [], "acc": []},
    }

    def record(step, results):
        steps_recorded.append(step)
        for split, (loss_val, grad_val, acc_val) in results.items():
            history[split]["loss"].append(loss_val)
            history[split]["grad"].append(grad_val)
            history[split]["acc"].append(acc_val)

    def metrics(model, dataset):
        zero_grad(model)
        loss = measure_loss(model, dataset)
        loss.backward()
        grad_norm = measure_grad_norm(model)
        acc = accuracy(model, dataset)
        zero_grad(model)
        return loss.item(), grad_norm.item(), acc

    def report(step):
        results = {
            "train": metrics(model, train_set),
            "test": metrics(model, test_set),
            "full": metrics(model, full_set),
        }
        record(step, results)

        train_loss, train_grad_norm, train_acc = results["train"]
        test_loss, test_grad_norm, test_acc = results["test"]
        full_loss, full_grad_norm, full_acc = results["full"]
        progress = 100.0 * step / max(steps, 1)
        # Use ANSI colors: train=cyan, test=yellow, full=magenta
        CYAN = "\033[36m"
        YELLOW = "\033[33m"
        MAGENTA = "\033[35m"
        RESET = "\033[0m"
        print(
            f"Step {step} ({progress:5.1f}%): "
            f"{CYAN}train{{ loss {train_loss:.2e} grad {train_grad_norm:.2e} acc {train_acc*100:.2f}% }}{RESET} "
            f"{YELLOW}test{{ loss {test_loss:.2e} grad {test_grad_norm:.2e} acc {test_acc*100:.2f}% }}{RESET} "
            f"{MAGENTA}full{{ loss {full_loss:.2e} grad {full_grad_norm:.2e} acc {full_acc*100:.2f}% }}{RESET}"
        )

    for step in range(steps):
        output = model(train_set.inputs)
        loss = torch.nn.CrossEntropyLoss()(output, train_set.targets)
        loss.backward()
        if step % report_every == 0:
            report(step)
        for p in model.parameters():
            p.data -= lr * p.grad
            p.grad.zero_()
    report(steps)

    plot_metrics(steps_recorded, history, "training_metrics.png")


def plot_metrics(steps_recorded, history, output_path):
    """Render loss/gradient norm/accuracy curves and persist to disk."""
    colors = {"train": "c", "test": "y", "full": "m"}
    metrics_to_plot = [
        ("loss", "Loss"),
        ("grad", "Gradient Norm"),
        ("acc", "Accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric_key, title) in zip(axes, metrics_to_plot):
        for split, color in colors.items():
            ax.plot(steps_recorded, history[split][metric_key], label=split.title(), color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")
        if metric_key == "loss":
            ax.set_ylabel("Loss")
        elif metric_key == "grad":
            ax.set_ylabel("Gradient Norm")
        else:
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
    axes[0].legend()
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an MLP on the full or downsampled dataset.")
    parser.add_argument("--n", type=int, default=10, help="Number of input bits/features.")
    parser.add_argument("--h", type=int, default=32768, help="Hidden layer size.")
    parser.add_argument("--l", type=int, default=1, help="Number of hidden layers.")
    parser.add_argument("--test_set_size", type=int, default=32, help="Size of test set.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--report_num", type=int, default=100, help="Number of reports.")

    args = parser.parse_args()
    n, h, l, test_set_size, steps, lr = args.n, args.h, args.l, args.test_set_size, args.steps, args.lr

    dataset = full_dataset(n)
    print("dataset size:", dataset.b)
    train_set, test_set = train_test_split(dataset, test_set_size)
    print("train set size:", train_set.b)
    print("test set size:", test_set.b)

    model = MLP(n, [h]*l, 2).cuda()
    print("model.dims: ", model.dims)

    report_every = steps // args.report_num
    train(model, train_set, test_set, dataset, steps, lr, report_every)

