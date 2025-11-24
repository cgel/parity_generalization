from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from data import Dataset, full_dataset, train_test_split
from model import MLP
from report import plot_metrics, print_step_metrics


def predictions(model: torch.nn.Module, dataset: Dataset) -> torch.Tensor:
    return model(dataset.inputs).argmax(dim=-1)

def accuracy(model: torch.nn.Module, dataset: Dataset) -> float:
    return (predictions(model, dataset) == dataset.targets).to(torch.float32).mean().item()

def measure_loss(model: torch.nn.Module, dataset: Dataset) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss()(model(dataset.inputs), dataset.targets)

def zero_grad(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

def measure_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    params = list(model.parameters())
    norms = [param.grad.norm() for param in params if param.grad is not None]
    if not norms:
        device = params[0].device if params else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(norms).sum()

def metrics(current_model: torch.nn.Module, dataset: Dataset) -> tuple[float, float, float]:
    zero_grad(current_model)
    loss = measure_loss(current_model, dataset)
    loss.backward()
    grad_norm = measure_grad_norm(current_model)
    acc = accuracy(current_model, dataset)
    zero_grad(current_model)
    return loss.item(), grad_norm.item(), acc


def train(
    model: torch.nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    full_set: Dataset,
    steps: int,
    lr: float,
    report_every: int,
    run_dir: Path | None,
) -> dict[str, Any]:
    steps_recorded: list[int] = []
    history: dict[str, dict[str, list[float]]] = {
        "train": {"loss": [], "grad": [], "acc": []},
        "test": {"loss": [], "grad": [], "acc": []},
        "full": {"loss": [], "grad": [], "acc": []},
    }
    reports: list[dict[str, Any]] = []

    def record(step: int, progress: float, results: dict[str, tuple[float, float, float]]) -> None:
        steps_recorded.append(step)
        reports.append(
            {
                "step": step,
                "progress": progress,
                "metrics": {
                    split: {"loss": loss_val, "grad": grad_val, "acc": acc_val}
                    for split, (loss_val, grad_val, acc_val) in results.items()
                },
            }
        )
        for split, (loss_val, grad_val, acc_val) in results.items():
            history[split]["loss"].append(loss_val)
            history[split]["grad"].append(grad_val)
            history[split]["acc"].append(acc_val)

    def report_step(step: int) -> None:
        results = {
            "train": metrics(model, train_set),
            "test": metrics(model, test_set),
            "full": metrics(model, full_set),
        }
        progress = 100.0 * step / max(steps, 1)
        record(step, progress, results)
        print_step_metrics(step, progress, results)

    for step in range(steps):
        if step % report_every == 0: report_step(step)
        output = model(train_set.inputs)
        loss = torch.nn.CrossEntropyLoss()(output, train_set.targets)
        loss.backward()
        for param in model.parameters():
            param.data -= lr * param.grad
            param.grad.zero_()
    report_step(steps)

    if run_dir is not None:
        metrics_path = run_dir / "metrics.png"
        plot_metrics(steps_recorded, history, metrics_path.as_posix())

    return {
        "steps_recorded": steps_recorded,
        "history": history,
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on the full or downsampled dataset.")
    parser.add_argument("--n", type=int, default=10, help="Number of input bits/features.")
    parser.add_argument("--h", type=int, default=32768, help="Hidden layer size.")
    parser.add_argument("--l", type=int, default=1, help="Number of hidden layers.")
    parser.add_argument("--test_set_size", type=int, default=32, help="Size of test set.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--report_num", type=int, default=50, help="Number of reports.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for this training run.")
    parser.add_argument(
        "--write_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist metrics and logs to disk; disable with --no-write-results.",
    )
    args = parser.parse_args()
    if args.report_num > args.steps:
        args.report_num = args.steps
    return args

def main() -> None:
    args = parse_args()
    n, h, l = args.n, args.h, args.l
    test_set_size, steps, lr = args.test_set_size, args.steps, args.lr

    dataset = full_dataset(n)
    print("dataset size:", dataset.b)
    train_set, test_set = train_test_split(dataset, test_set_size)
    print("train set size:", train_set.b)
    print("test set size:", test_set.b)

    model = MLP(n, [h] * l, 2).cuda()
    print("model.dims:", model.dims)

    report_every = max(1, steps // max(args.report_num, 1))

    save_results = args.write_results
    run_dir: Path | None = None
    run_name = args.run_name
    if save_results:
        run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    training_artifacts = train(
        model,
        train_set,
        test_set,
        dataset,
        steps,
        lr,
        report_every,
        run_dir=run_dir,
    )

    if save_results and run_dir is not None:
        log_payload = {
            "run_name": run_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "dataset": {
                "full_size": dataset.b,
                "train_size": train_set.b,
                "test_size": test_set.b,
            },
            **training_artifacts,
        }

        log_path = run_dir / "log.json"
        log_path.write_text(json.dumps(log_payload, indent=2))


if __name__ == "__main__":
    main()

