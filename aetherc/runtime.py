"""Aether runtime: hardware detection, tensor materialization, chunked inference."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SEED = 0xAE7E
F32 = np.float32
MODEL_MAGIC = b"AEB1"
MODEL_VERSION = 1
MODEL_HEADER = struct.Struct("<4sIIIIIII")


@dataclass
class Tensor:
    array: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]

    @property
    def mib(self) -> float:
        return self.array.nbytes / (1024 * 1024)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    classes: int
    samples: int
    features: int

    @property
    def mib(self) -> float:
        return (self.X.nbytes + self.y.nbytes) / (1024 * 1024)


@dataclass
class Model:
    in_dim: int
    hidden: int
    out_dim: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    policy: dict = field(default_factory=dict)
    model_kind: str = "mlp"
    classes: int = 0
    trained: bool = False

    @property
    def params(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    @property
    def mib(self) -> float:
        return (self.w1.nbytes + self.b1.nbytes + self.w2.nbytes + self.b2.nbytes) / (1024 * 1024)


@dataclass
class StreamResult:
    rows: int
    out_dim: int
    chunks: int
    gflops: float
    elapsed_ms: float
    gflops_per_s: float
    checksum: float
    digest: str
    accuracy: float | None = None
    loss: float | None = None


@dataclass
class TrainingResult:
    epochs: int
    samples: int
    accuracy: float
    loss: float
    elapsed_ms: float
    heals: int
    steps: int
    lr: float

    @property
    def samples_per_s(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return self.samples * self.epochs / (self.elapsed_ms / 1000.0)


def detect_device() -> dict:
    threads = int(os.environ.get("AETHER_THREADS") or (os.cpu_count() or 1))
    features: list[str] = []
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as cpu_features

        for name in ("AVX512F", "AVX512VNNI", "AVX2", "FMA3", "SSE42", "NEON"):
            if cpu_features.get(name):
                features.append(name)
    except Exception:
        features = []

    blas = "unknown"
    try:
        config = np.show_config(mode="dicts")
        blas_info = config.get("Build Dependencies", {}).get("blas", {}) or {}
        blas = str(blas_info.get("name") or blas_info.get("version") or "unknown")
    except Exception:
        pass

    return {
        "device": "cpu",
        "machine": platform.machine().lower() or "unknown",
        "features": features,
        "threads": threads,
        "blas": blas,
        "dtype": "f32",
    }


class Runtime:
    def __init__(
        self,
        chunks: int = 8,
        on_event: Callable[[dict], None] | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.chunks = max(1, min(int(chunks), 256))
        self.on_event = on_event or (lambda event: None)
        self.rng = np.random.default_rng(SEED)
        self.device = detect_device()
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._buffer_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._emit_event("device", **self.device)

    def _emit_event(self, event: str, **payload: object) -> None:
        self.on_event({"event": event, **payload})

    # ------------------------------------------------------------- allocation
    def tensor(self, rows: int, cols: int) -> Tensor:
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"tensor dimensions must be positive, got ({rows}, {cols})")
        array = self.rng.standard_normal((rows, cols), dtype=F32)
        tensor = Tensor(array=array)
        self._emit_event("tensor", rows=rows, cols=cols, mib=tensor.mib, seed=SEED)
        return tensor

    def load_data(self, samples: int, features: int, classes: int = 10) -> Dataset:
        samples, features, classes = int(samples), int(features), int(classes)
        if samples <= 0 or features <= 0 or classes < 2:
            raise ValueError("load_data requires positive samples/features and at least 2 classes")

        rng = np.random.default_rng(SEED ^ (samples << 12) ^ (features << 4) ^ classes)
        X = rng.standard_normal((samples, features), dtype=F32)
        true_w = (
            rng.standard_normal((features, classes), dtype=F32)
            * F32(1.0 / np.sqrt(features))
        )
        logits = X @ true_w + F32(0.25) * rng.standard_normal(
            (samples, classes), dtype=F32
        )
        y = np.argmax(logits, axis=1).astype(np.int64)

        dataset = Dataset(X=X, y=y, classes=classes, samples=samples, features=features)
        self._emit_event(
            "dataset",
            samples=samples,
            features=features,
            classes=classes,
            mib=dataset.mib,
            balance=float(np.bincount(y, minlength=classes).max() / samples),
        )
        return dataset

    def neural_net(
        self,
        input: Tensor,
        out_dim: int,
        hidden: int = 512,
        policy: dict | None = None,
    ) -> Model:
        if not isinstance(input, Tensor):
            raise TypeError("neural_net 'in' must be a tensor")
        return self._build_model(
            int(input.shape[1]),
            int(out_dim),
            int(hidden),
            policy,
            model_kind="mlp",
            trained=True,
        )

    def classifier(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 128,
        policy: dict | None = None,
    ) -> Model:
        return self._build_model(
            int(in_dim),
            int(classes),
            int(hidden),
            policy,
            model_kind="classifier",
            trained=False,
        )

    def _build_model(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        policy: dict | None,
        model_kind: str,
        trained: bool,
    ) -> Model:
        policy = dict(policy or {"speed": "balanced"})
        seed = SEED ^ (in_dim << 20) ^ (hidden << 8) ^ out_dim ^ (0x1000 if model_kind == "classifier" else 0)
        rng = np.random.default_rng(seed)
        w1 = (
            rng.standard_normal((in_dim, hidden), dtype=F32)
            * F32(np.sqrt(2.0 / in_dim))
        ).astype(F32)
        b1 = np.zeros(hidden, dtype=F32)
        w2 = (
            rng.standard_normal((hidden, out_dim), dtype=F32)
            * F32(np.sqrt(2.0 / hidden))
        ).astype(F32)
        b2 = np.zeros(out_dim, dtype=F32)

        model = Model(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            policy=policy,
            model_kind=model_kind,
            classes=out_dim if model_kind == "classifier" else 0,
            trained=trained,
        )
        self._warm_blas(model)
        self._emit_event(
            "model",
            kind=model_kind,
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            params=model.params,
            mib=model.mib,
            policy=policy.get("speed", "balanced"),
        )
        return model

    # ---------------------------------------------------------------- compute
    def _buffers(
        self, rows: int, hidden: int, out_dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (rows, hidden, out_dim)
        if key not in self._buffer_cache:
            self._buffer_cache[key] = (
                np.empty((rows, hidden), dtype=F32),
                np.empty((rows, out_dim), dtype=F32),
            )
        return self._buffer_cache[key]

    def _warm_blas(self, model: Model) -> None:
        """Pay the one-time BLAS thread-pool cost before the timed stream."""
        if getattr(self, "_blas_warm", False):
            return
        batch = min(256, model.in_dim)
        x = np.zeros((batch, model.in_dim), dtype=F32)
        h = np.empty((batch, model.hidden), dtype=F32)
        y = np.empty((batch, model.out_dim), dtype=F32)
        np.matmul(x, model.w1, out=h)
        np.maximum(h, 0.0, out=h)
        np.matmul(h, model.w2, out=y)
        self._blas_warm = True

    def stream(self, model: Model, data: Tensor | Dataset) -> StreamResult:
        if not isinstance(model, Model):
            raise TypeError("run target must be a model built by neural_net")
        if isinstance(data, Dataset):
            if model.model_kind != "classifier":
                raise TypeError("only classifiers can run on a dataset")
            return self._stream_classifier(model, data)
        if not isinstance(data, Tensor):
            raise TypeError("run input must be a tensor")
        if data.shape[1] != model.in_dim:
            raise ValueError(
                f"input width mismatch: model expects {model.in_dim}, got {data.shape[1]}"
            )

        rows = int(data.shape[0])
        total = max(1, min(self.chunks, rows))
        bounds = [i * rows // total for i in range(total + 1)]
        max_chunk = max(bounds[i + 1] - bounds[i] for i in range(total))

        speed = str(model.policy.get("speed", "balanced"))
        # 'low' caps the working set at 64 rows; 'max' reuses buffers across runs.
        working_rows = min(64, max_chunk) if speed == "low" else max_chunk
        if speed == "max":
            h_buffer, y_buffer = self._buffers(working_rows, model.hidden, model.out_dim)
        else:
            h_buffer = np.empty((working_rows, model.hidden), dtype=F32)
            y_buffer = np.empty((working_rows, model.out_dim), dtype=F32)

        self._emit_event(
            "stream_start",
            chunks=total,
            rows=rows,
            out_dim=model.out_dim,
            chunk_rows=bounds[1] - bounds[0],
            working_rows=working_rows,
            policy=speed,
        )

        digest = hashlib.blake2b(digest_size=4)
        checksum = 0.0
        gflops_per_row = 2.0 * (model.in_dim * model.hidden + model.hidden * model.out_dim) / 1e9
        started = time.perf_counter()

        for index in range(total):
            start, end = bounds[index], bounds[index + 1]
            count = 0
            value_sum = 0.0
            square_sum = 0.0
            peak = 0.0
            sample: list[float] | None = None
            chunk_started = time.perf_counter()

            sub_start = start
            while sub_start < end:
                sub_end = min(end, sub_start + working_rows)
                width = sub_end - sub_start
                x = data.array[sub_start:sub_end]
                h = h_buffer[:width]
                y = y_buffer[:width]

                np.matmul(x, model.w1, out=h)   # BLAS sgemm
                h += model.b1                   # fused bias
                np.maximum(h, 0.0, out=h)       # fused ReLU
                np.matmul(h, model.w2, out=y)  # BLAS sgemm
                y += model.b2

                value_sum += float(y.sum(dtype=np.float64))
                square_sum += float(np.square(y, dtype=np.float64).sum())
                peak = max(peak, float(np.abs(y).max()))
                digest.update(y.tobytes())
                if sample is None:
                    sample = [round(float(v), 4) for v in y[0, :4]]
                count += int(y.size)
                sub_start = sub_end

            chunk_ms = (time.perf_counter() - chunk_started) * 1000.0
            mean = value_sum / count if count else 0.0
            variance = max(0.0, square_sum / count - mean * mean) if count else 0.0
            checksum += value_sum

            self._emit_event(
                "chunk",
                index=index + 1,
                total=total,
                row_start=start,
                row_end=end - 1,
                rows=end - start,
                mean=mean,
                std=variance ** 0.5,
                peak=peak,
                sample=sample or [],
                ms=chunk_ms,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        gflops = rows * gflops_per_row
        result = StreamResult(
            rows=rows,
            out_dim=model.out_dim,
            chunks=total,
            gflops=gflops,
            elapsed_ms=elapsed_ms,
            gflops_per_s=gflops / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            checksum=checksum,
            digest="0x" + digest.hexdigest(),
        )
        self._emit_event(
            "summary",
            chunks=total,
            rows=rows,
            out_dim=model.out_dim,
            gflops=gflops,
            ms=elapsed_ms,
            gflops_per_s=result.gflops_per_s,
            checksum=checksum,
            digest=result.digest,
            params=model.params,
        )
        return result

    # --------------------------------------------------------------- inference
    def _stream_classifier(self, model: Model, dataset: Dataset) -> StreamResult:
        if dataset.features != model.in_dim:
            raise ValueError(
                f"input width mismatch: model expects {model.in_dim}, dataset has {dataset.features}"
            )
        if dataset.classes != model.out_dim:
            raise ValueError(
                f"class mismatch: model predicts {model.out_dim}, dataset has {dataset.classes}"
            )

        total = max(1, min(self.chunks, dataset.samples))
        bounds = [i * dataset.samples // total for i in range(total + 1)]
        self._emit_event(
            "stream_start",
            chunks=total,
            rows=dataset.samples,
            out_dim=model.out_dim,
            chunk_rows=bounds[1] - bounds[0],
            working_rows=bounds[1] - bounds[0],
            policy=str(model.policy.get("speed", "balanced")),
            kind="classifier",
        )

        digest = hashlib.blake2b(digest_size=4)
        checksum = 0.0
        correct = 0
        loss_sum = 0.0
        gflops_per_row = 2.0 * (model.in_dim * model.hidden + model.hidden * model.out_dim) / 1e9
        started = time.perf_counter()

        for index in range(total):
            start, end = bounds[index], bounds[index + 1]
            chunk_started = time.perf_counter()
            x = dataset.X[start:end]
            y = dataset.y[start:end]
            probabilities = self._softmax(self._forward(model, x))
            loss = float(
                -np.log(probabilities[np.arange(len(y)), y] + 1e-9).mean()
            )
            predicted = probabilities.argmax(axis=1)
            chunk_correct = int((predicted == y).sum())
            confidence = float(probabilities.max(axis=1).mean())
            correct += chunk_correct
            loss_sum += loss * len(y)
            checksum += float(probabilities.sum(dtype=np.float64))
            digest.update(predicted.astype(np.int8).tobytes())
            chunk_ms = (time.perf_counter() - chunk_started) * 1000.0

            self._emit_event(
                "chunk",
                index=index + 1,
                total=total,
                row_start=start,
                row_end=end - 1,
                rows=end - start,
                accuracy=chunk_correct / len(y),
                confidence=confidence,
                loss=loss,
                ms=chunk_ms,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        accuracy = correct / dataset.samples
        mean_loss = loss_sum / dataset.samples
        gflops = dataset.samples * gflops_per_row
        result = StreamResult(
            rows=dataset.samples,
            out_dim=model.out_dim,
            chunks=total,
            gflops=gflops,
            elapsed_ms=elapsed_ms,
            gflops_per_s=gflops / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            checksum=checksum,
            digest="0x" + digest.hexdigest(),
            accuracy=accuracy,
            loss=mean_loss,
        )
        self._emit_event(
            "summary",
            chunks=total,
            rows=dataset.samples,
            out_dim=model.out_dim,
            gflops=gflops,
            ms=elapsed_ms,
            gflops_per_s=result.gflops_per_s,
            checksum=checksum,
            digest=result.digest,
            params=model.params,
            accuracy=accuracy,
            loss=mean_loss,
            kind="classifier",
        )
        return result

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted, dtype=logits.dtype)
        return exponent / exponent.sum(axis=1, keepdims=True)

    @staticmethod
    def _forward(model: Model, x: np.ndarray) -> np.ndarray:
        hidden = x @ model.w1
        hidden += model.b1
        np.maximum(hidden, 0.0, out=hidden)
        logits = hidden @ model.w2
        logits += model.b2
        return logits

    def _evaluate(
        self, model: Model, dataset: Dataset, batch_size: int = 512
    ) -> tuple[float, float]:
        correct = 0
        loss_sum = 0.0
        for start in range(0, dataset.samples, batch_size):
            x = dataset.X[start : start + batch_size]
            y = dataset.y[start : start + batch_size]
            probabilities = self._softmax(self._forward(model, x))
            loss_sum += float(
                -np.log(probabilities[np.arange(len(y)), y] + 1e-9).sum()
            )
            correct += int((probabilities.argmax(axis=1) == y).sum())
        return correct / dataset.samples, loss_sum / dataset.samples

    # --------------------------------------------------------------- training
    def train(
        self,
        model: Model,
        dataset: Dataset,
        epochs: int,
        batch_size: int = 256,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        clip_norm: float = 5.0,
    ) -> TrainingResult:
        if not isinstance(model, Model):
            raise TypeError("train target must be a model")
        if not isinstance(dataset, Dataset):
            raise TypeError("train data must be a dataset")
        if model.model_kind != "classifier":
            raise TypeError("only classifier(...) models are trainable in v0.2")
        if dataset.features != model.in_dim:
            raise ValueError(
                f"feature mismatch: model expects {model.in_dim}, dataset has {dataset.features}"
            )
        if dataset.classes != model.out_dim:
            raise ValueError(
                f"class mismatch: model predicts {model.out_dim}, dataset has {dataset.classes}"
            )

        epochs = int(epochs)
        batch_size = max(1, min(int(batch_size), dataset.samples))
        rng = np.random.default_rng(SEED ^ 0x7EA1)
        lr = float(learning_rate)
        velocity = [
            np.zeros_like(model.w1),
            np.zeros_like(model.b1),
            np.zeros_like(model.w2),
            np.zeros_like(model.b2),
        ]
        best = [
            model.w1.copy(),
            model.b1.copy(),
            model.w2.copy(),
            model.b2.copy(),
        ]
        heals = 0
        steps = 0
        history: list[dict] = []
        best_loss = float("inf")
        best_accuracy = 0.0

        self._emit_event(
            "train_start",
            epochs=epochs,
            samples=dataset.samples,
            features=dataset.features,
            classes=dataset.classes,
            batch=batch_size,
            lr=lr,
            optimizer="sgd+momentum",
            heal=clip_norm,
        )

        training_started = time.perf_counter()
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            order = rng.permutation(dataset.samples)
            epoch_loss = 0.0
            seen = 0
            heals_epoch = 0

            for start in range(0, dataset.samples, batch_size):
                index = order[start : start + batch_size]
                x = dataset.X[index]
                y = dataset.y[index]
                count = len(index)

                hidden = x @ model.w1
                hidden += model.b1
                np.maximum(hidden, 0.0, out=hidden)
                logits = hidden @ model.w2
                logits += model.b2
                probabilities = self._softmax(logits)
                loss = float(
                    -np.log(probabilities[np.arange(count), y] + 1e-9).mean()
                )

                dlogits = probabilities
                dlogits[np.arange(count), y] -= 1.0
                dlogits /= count
                gw2 = hidden.T @ dlogits
                gb2 = dlogits.sum(axis=0)
                dhidden = dlogits @ model.w2.T
                dhidden *= hidden > 0
                gw1 = x.T @ dhidden
                gb1 = dhidden.sum(axis=0)

                norm_sq = (
                    float(np.square(gw1, dtype=np.float64).sum())
                    + float(np.square(gb1, dtype=np.float64).sum())
                    + float(np.square(gw2, dtype=np.float64).sum())
                    + float(np.square(gb2, dtype=np.float64).sum())
                )
                if not math.isfinite(loss) or not math.isfinite(norm_sq):
                    lr = max(lr * 0.5, 1e-4)
                    heals += 1
                    heals_epoch += 1
                    self._emit_event(
                        "heal",
                        stage="step",
                        epoch=epoch + 1,
                        reason="non-finite loss or gradient",
                        lr=lr,
                    )
                    continue

                scale = clip_norm / math.sqrt(norm_sq) if norm_sq > clip_norm**2 else 1.0
                for parameter, gradient, velocity_buffer in zip(
                    (model.w1, model.b1, model.w2, model.b2),
                    (gw1, gb1, gw2, gb2),
                    velocity,
                ):
                    velocity_buffer *= momentum
                    velocity_buffer -= lr * scale * gradient
                    parameter += velocity_buffer

                epoch_loss += loss * count
                seen += count
                steps += 1

            accuracy, validation_loss = self._evaluate(model, dataset)
            epoch_ms = (time.perf_counter() - epoch_started) * 1000.0
            mean_loss = epoch_loss / seen if seen else float("nan")

            diverged = not math.isfinite(validation_loss) or (
                epoch > 0 and validation_loss > best_loss * 4.0 + 1e-6
            )
            if diverged:
                for parameter, snapshot in zip(
                    (model.w1, model.b1, model.w2, model.b2), best
                ):
                    parameter[...] = snapshot
                lr = max(lr * 0.5, 1e-4)
                heals += 1
                heals_epoch += 1
                accuracy, validation_loss = self._evaluate(model, dataset)
                mean_loss = validation_loss
                self._emit_event(
                    "heal",
                    stage="epoch",
                    epoch=epoch + 1,
                    reason="loss diverged; restored best weights",
                    lr=lr,
                )

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_accuracy = accuracy
                for parameter, snapshot in zip(
                    (model.w1, model.b1, model.w2, model.b2), best
                ):
                    snapshot[...] = parameter

            epoch_record = {
                "epoch": epoch + 1,
                "loss": mean_loss,
                "val_loss": validation_loss,
                "accuracy": accuracy,
                "lr": lr,
                "ms": epoch_ms,
                "heals": heals_epoch,
            }
            history.append(epoch_record)
            self._emit_event("epoch", total=epochs, **epoch_record)

        model.trained = True
        elapsed_ms = (time.perf_counter() - training_started) * 1000.0
        result = TrainingResult(
            epochs=epochs,
            samples=dataset.samples,
            accuracy=best_accuracy if math.isfinite(best_accuracy) else 0.0,
            loss=best_loss if math.isfinite(best_loss) else float("nan"),
            elapsed_ms=elapsed_ms,
            heals=heals,
            steps=steps,
            lr=lr,
        )
        self._emit_event(
            "train_done",
            epochs=epochs,
            samples=dataset.samples,
            ms=elapsed_ms,
            accuracy=result.accuracy,
            loss=result.loss,
            heals=heals,
            steps=steps,
            lr=lr,
            samples_per_s=result.samples_per_s,
            params=model.params,
            history=history,
        )
        return result

    # ------------------------------------------------------------------ export
    def export(self, model: Model, path: str) -> Path:
        if not isinstance(model, Model):
            raise TypeError("export target must be a model")
        if model.model_kind == "classifier" and not model.trained:
            raise ValueError("model must be trained before export")

        target = Path(path)
        if not target.is_absolute():
            target = self.base_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)

        kind = 1 if model.model_kind == "classifier" else 0
        header = MODEL_HEADER.pack(
            MODEL_MAGIC,
            MODEL_VERSION,
            kind,
            model.in_dim,
            model.hidden,
            model.out_dim,
            model.classes,
            0,
        )
        with target.open("wb") as handle:
            handle.write(header)
            for array in (model.w1, model.b1, model.w2, model.b2):
                handle.write(np.ascontiguousarray(array, dtype="<f4").tobytes())

        size = target.stat().st_size
        self._emit_event(
            "export",
            path=str(target),
            bytes=size,
            params=model.params,
            format="AEB1",
            kind=model.model_kind,
        )
        return target

    def load(self, path: str) -> Model:
        target = Path(path)
        if not target.is_absolute():
            target = self.base_dir / target
        with target.open("rb") as handle:
            header = handle.read(MODEL_HEADER.size)
            if len(header) != MODEL_HEADER.size:
                raise ValueError(f"{target} is not an Aether model file")
            magic, version, kind, in_dim, hidden, out_dim, classes, _ = MODEL_HEADER.unpack(
                header
            )
            if magic != MODEL_MAGIC or version != MODEL_VERSION:
                raise ValueError(f"{target} is not an Aether v{MODEL_VERSION} model file")
            expected = in_dim * hidden + hidden + hidden * out_dim + out_dim
            payload = np.fromfile(handle, dtype="<f4", count=expected)
        if payload.size != expected:
            raise ValueError(f"{target} is truncated")
        offset = 0

        def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal offset
            array = payload[offset : offset + count].reshape(shape)
            offset += count
            return np.array(array, dtype=F32, copy=True)

        model = Model(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            w1=take(in_dim * hidden, (in_dim, hidden)),
            b1=take(hidden, (hidden,)),
            w2=take(hidden * out_dim, (hidden, out_dim)),
            b2=take(out_dim, (out_dim,)),
            policy={"speed": "max"},
            model_kind="classifier" if kind == 1 else "mlp",
            classes=classes,
            trained=True,
        )
        self._emit_event(
            "load",
            path=str(target),
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            params=model.params,
            kind=model.model_kind,
        )
        return model

    # ------------------------------------------------------------- web serving
    def load_model(self, path: str) -> Model:
        """Load an AEB1 model file (alias of Runtime.load)."""
        return self.load(path)

    def serve(self, model: Model, port: int, route: str) -> None:
        """Block serving `model` over HTTP/JSON at http://127.0.0.1:<port><route>."""
        import json as _json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        if not isinstance(model, Model):
            raise TypeError("serve target must be a model")
        port = int(os.environ.get("AETHER_PORT") or port)
        route = os.environ.get("AETHER_ROUTE") or route
        runtime = self
        started = time.perf_counter()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, status: int, payload: dict) -> None:
                body = _json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path.split("?")[0] in ("/health", "/"):
                    self._reply(
                        200,
                        {
                            "status": "ok",
                            "model": "served",
                            "in_dim": model.in_dim,
                            "out_dim": model.out_dim,
                            "route": route,
                        },
                    )
                else:
                    self._reply(404, {"error": "not found", "route": route})

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                request_started = time.perf_counter()
                if self.path.split("?")[0] != route:
                    self._reply(404, {"error": "not found", "route": route})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = _json.loads(self.rfile.read(length) or b"null")
                except ValueError:
                    self._reply(400, {"error": "invalid JSON body"})
                    return
                features = (
                    payload.get("features")
                    if isinstance(payload, dict)
                    else payload
                )
                if not isinstance(features, list) or len(features) != model.in_dim:
                    self._reply(
                        400,
                        {
                            "error": f"features must be an array of {model.in_dim} numbers",
                            "received": len(features) if isinstance(features, list) else None,
                        },
                    )
                    return
                try:
                    x = np.asarray(features, dtype=F32).reshape(1, model.in_dim)
                except (TypeError, ValueError):
                    self._reply(400, {"error": "features must be numbers"})
                    return
                probabilities = runtime._softmax(runtime._forward(model, x))[0]
                prediction = int(probabilities.argmax())
                elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                self._reply(
                    200,
                    {
                        "prediction": prediction,
                        "confidence": float(probabilities[prediction]),
                        "probabilities": [float(v) for v in probabilities],
                        "in_dim": model.in_dim,
                        "out_dim": model.out_dim,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                runtime._emit_event(
                    "request",
                    method="POST",
                    path=self.path.split("?")[0],
                    prediction=prediction,
                    confidence=float(probabilities[prediction]),
                    ms=elapsed_ms,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                )

            def log_message(self, fmt: str, *args: object) -> None:  # silence stderr
                return

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._emit_event(
            "serve_start",
            port=port,
            route=route,
            in_dim=model.in_dim,
            out_dim=model.out_dim,
            backend="python",
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self._emit_event("serve_stop", port=port, backend="python")

    # ------------------------------------------------------------------- logs
    def log(self, value: object) -> None:
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        self._emit_event("log", text=text)

    # ------------------------------------------------------------------ emit
    def emit(self, value: object) -> None:
        if isinstance(value, Tensor):
            self._emit_event(
                "emit_tensor",
                rows=value.shape[0],
                cols=value.shape[1],
                mean=float(value.array.mean(dtype=np.float64)),
                std=float(value.array.std(dtype=np.float64)),
                peak=float(np.abs(value.array).max()),
                mib=value.mib,
            )
        elif isinstance(value, StreamResult):
            payload = {
                "rows": value.rows,
                "out_dim": value.out_dim,
                "checksum": value.checksum,
                "digest": value.digest,
            }
            if value.accuracy is not None:
                payload["accuracy"] = value.accuracy
                payload["loss"] = value.loss
            self._emit_event("emit_stream", **payload)
        elif isinstance(value, Dataset):
            self._emit_event(
                "emit_dataset",
                samples=value.samples,
                features=value.features,
                classes=value.classes,
                mib=value.mib,
            )
        else:
            self._emit_event("print", text=str(value))

    def unsupported(self, name: str) -> None:
        raise RuntimeError(f"intent '{name}' is not implemented by the v0.1 runtime")
