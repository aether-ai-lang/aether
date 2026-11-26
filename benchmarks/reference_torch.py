"""Standard PyTorch reference implementation used by `aether bench`.

Same workload and hyperparameters as the NumPy reference: synthetic dataset,
MLP classifier, SGD + momentum, gradient clipping, evaluation pass. Runs on
CPU with the ambient thread budget.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch reference training benchmark")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    threads = int(os.environ.get("AETHER_THREADS") or (os.cpu_count() or 1))
    torch.set_num_threads(threads)

    # Warm oneDNN/BLAS kernels so the timer measures steady state, matching the
    # Aether rows (whose process startup and library load sit outside timers).
    warm_model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    warm_opt = torch.optim.SGD(warm_model.parameters(), lr=0.01)
    warm_x = torch.randn(4, 8)
    for _ in range(3):
        warm_opt.zero_grad()
        warm_loss = warm_model(warm_x).sum()
        warm_loss.backward()
        warm_opt.step()
    del warm_model, warm_opt, warm_x, warm_loss

    started = time.perf_counter()
    generator = torch.Generator().manual_seed(42)

    # dataset -------------------------------------------------------------
    X = torch.randn(args.samples, args.features, generator=generator)
    true_w = torch.randn(args.features, args.classes, generator=generator) / (
        args.features**0.5
    )
    logits = X @ true_w + 0.25 * torch.randn(args.samples, args.classes, generator=generator)
    y = logits.argmax(dim=1)

    # model ----------------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(args.features, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.classes),
    )
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    train_started = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    for _epoch in range(args.epochs):
        perm = torch.randperm(args.samples, generator=generator)
        for start in range(0, args.samples, args.batch):
            index = perm[start : start + args.batch]
            xb = X[index]
            yb = y[index]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # self_heal semantics: evaluate after every epoch, keep the best weights
        with torch.no_grad():
            epoch_correct = 0
            epoch_loss = 0.0
            for start in range(0, args.samples, 512):
                xb = X[start : start + 512]
                yb = y[start : start + 512]
                out = model(xb)
                epoch_loss += loss_fn(out, yb).item() * len(yb)
                epoch_correct += int((out.argmax(dim=1) == yb).sum().item())
        validation_loss = epoch_loss / args.samples
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_ms = (time.perf_counter() - train_started) * 1000.0

    infer_started = time.perf_counter()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, args.samples, 512):
            xb = X[start : start + 512]
            yb = y[start : start + 512]
            out = model(xb)
            loss_sum += loss_fn(out, yb).item() * len(yb)
            correct += int((out.argmax(dim=1) == yb).sum().item())
    infer_ms = (time.perf_counter() - infer_started) * 1000.0

    print(
        json.dumps(
            {
                "impl": "pytorch reference",
                "ms": (time.perf_counter() - started) * 1000.0,
                "train_ms": train_ms,
                "infer_ms": infer_ms,
                "accuracy": correct / args.samples,
                "loss": loss_sum / args.samples,
                "steps": args.epochs * (args.samples // args.batch),
                "params": params,
                "threads": threads,
            }
        )
    )


if __name__ == "__main__":
    main()
