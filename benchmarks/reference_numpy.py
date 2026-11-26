"""Standard Python/NumPy reference implementation used by `aether bench`.

This is the hand-written equivalent of the .ae program under test: synthetic
dataset, an MLP classifier, SGD with momentum, gradient clipping, and a final
evaluation pass. It is deliberately idiomatic NumPy -- no Aether code involved.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

F32 = np.float32


def _evaluate(
    X: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    classes: int,
) -> tuple[float, float]:
    correct = 0
    loss_sum = 0.0
    for start in range(0, len(X), 512):
        xb = X[start : start + 512]
        yb = y[start : start + 512]
        hidden = xb @ w1
        hidden += b1
        np.maximum(hidden, 0.0, out=hidden)
        out = hidden @ w2
        out += b2
        out -= out.max(axis=1, keepdims=True)
        np.exp(out, out=out)
        out /= out.sum(axis=1, keepdims=True)
        loss_sum += float(-np.log(out[np.arange(len(yb)), yb] + 1e-9).sum())
        correct += int((out.argmax(axis=1) == yb).sum())
    return correct / len(X), loss_sum / len(X)


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy reference training benchmark")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=5.0)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(42)

    # dataset -------------------------------------------------------------
    X = rng.standard_normal((args.samples, args.features), dtype=F32)
    true_w = rng.standard_normal((args.features, args.classes), dtype=F32) * F32(
        1.0 / np.sqrt(args.features)
    )
    logits = X @ true_w + F32(0.25) * rng.standard_normal(
        (args.samples, args.classes), dtype=F32
    )
    y = logits.argmax(axis=1).astype(np.int64)

    # model (He init) ------------------------------------------------------
    w1 = (
        rng.standard_normal((args.features, args.hidden), dtype=F32)
        * F32(np.sqrt(2.0 / args.features))
    ).astype(F32)
    b1 = np.zeros(args.hidden, dtype=F32)
    w2 = (
        rng.standard_normal((args.hidden, args.classes), dtype=F32)
        * F32(np.sqrt(2.0 / args.hidden))
    ).astype(F32)
    b2 = np.zeros(args.classes, dtype=F32)
    params = int(w1.size + b1.size + w2.size + b2.size)

    vw1 = np.zeros_like(w1)
    vb1 = np.zeros_like(b1)
    vw2 = np.zeros_like(w2)
    vb2 = np.zeros_like(b2)
    lr = args.lr
    heals = 0
    steps = 0

    train_started = time.perf_counter()
    best_loss = float("inf")
    best_accuracy = 0.0
    best_weights = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
    for _epoch in range(args.epochs):
        order = rng.permutation(args.samples)
        for start in range(0, args.samples, args.batch):
            index = order[start : start + args.batch]
            xb = X[index]
            yb = y[index]
            count = len(index)

            hidden = xb @ w1
            hidden += b1
            np.maximum(hidden, 0.0, out=hidden)
            out = hidden @ w2
            out += b2
            out -= out.max(axis=1, keepdims=True)
            np.exp(out, out=out)
            out /= out.sum(axis=1, keepdims=True)
            loss = -np.log(out[np.arange(count), yb] + 1e-9).mean()

            dout = out
            dout[np.arange(count), yb] -= 1.0
            dout /= count
            gw2 = hidden.T @ dout
            gb2 = dout.sum(axis=0)
            dh = dout @ w2.T
            dh *= hidden > 0
            gw1 = xb.T @ dh
            gb1 = dh.sum(axis=0)

            norm_sq = (
                float(np.square(gw1, dtype=np.float64).sum())
                + float(np.square(gb1, dtype=np.float64).sum())
                + float(np.square(gw2, dtype=np.float64).sum())
                + float(np.square(gb2, dtype=np.float64).sum())
            )
            if not math.isfinite(loss) or not math.isfinite(norm_sq):
                lr = max(lr * 0.5, 1e-4)
                heals += 1
                continue
            scale = args.clip / math.sqrt(norm_sq) if norm_sq > args.clip**2 else 1.0
            for parameter, gradient, velocity in (
                (w1, gw1, vw1),
                (b1, gb1, vb1),
                (w2, gw2, vw2),
                (b2, gb2, vb2),
            ):
                velocity *= args.momentum
                velocity -= lr * scale * gradient
                parameter += velocity
            steps += 1

        # self_heal semantics: evaluate after every epoch, keep the best weights
        accuracy, validation_loss = _evaluate(X, y, w1, b1, w2, b2, args.classes)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = accuracy
            best_weights = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
    w1, b1, w2, b2 = best_weights
    train_ms = (time.perf_counter() - train_started) * 1000.0

    # evaluation -----------------------------------------------------------
    infer_started = time.perf_counter()
    correct = 0
    loss_sum = 0.0
    for start in range(0, args.samples, 512):
        xb = X[start : start + 512]
        yb = y[start : start + 512]
        hidden = xb @ w1
        hidden += b1
        np.maximum(hidden, 0.0, out=hidden)
        out = hidden @ w2
        out += b2
        out -= out.max(axis=1, keepdims=True)
        np.exp(out, out=out)
        out /= out.sum(axis=1, keepdims=True)
        loss_sum += float(-np.log(out[np.arange(len(yb)), yb] + 1e-9).sum())
        correct += int((out.argmax(axis=1) == yb).sum())
    infer_ms = (time.perf_counter() - infer_started) * 1000.0

    print(
        json.dumps(
            {
                "impl": "python/numpy reference",
                "ms": (time.perf_counter() - started) * 1000.0,
                "train_ms": train_ms,
                "infer_ms": infer_ms,
                "accuracy": correct / args.samples,
                "loss": loss_sum / args.samples,
                "heals": heals,
                "steps": steps,
                "params": params,
            }
        )
    )


if __name__ == "__main__":
    main()
