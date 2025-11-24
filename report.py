from collections.abc import Mapping
from typing import Sequence

import matplotlib.pyplot as plt


def print_step_metrics(step: int, progress: float, results: Mapping[str, tuple[float, float, float]]) -> None:
    """Pretty-print loss/gradient/accuracy metrics with ANSI colors."""
    cyan = "\033[36m"
    yellow = "\033[33m"
    magenta = "\033[35m"
    reset = "\033[0m"

    train_loss, train_grad_norm, train_acc = results["train"]
    test_loss, test_grad_norm, test_acc = results["test"]
    full_loss, full_grad_norm, full_acc = results["full"]

    print(
        f"Step:{step:6.0f} /{progress:3.0f}% "
        f"{cyan}train{{ loss {train_loss:.2e} grad {train_grad_norm:.2e} acc {train_acc*100:.2f}% }}{reset} "
        f"{yellow}test{{ loss {test_loss:.2e} grad {test_grad_norm:.2e} acc {test_acc*100:.2f}% }}{reset} "
        f"{magenta}full{{ loss {full_loss:.2e} grad {full_grad_norm:.2e} acc {full_acc*100:.2f}% }}{reset}"
    )


def plot_metrics(steps_recorded: Sequence[int], history: Mapping[str, Mapping[str, list[float]]], output_path: str) -> None:
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

