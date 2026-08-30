#!/usr/bin/env python3
"""
cifar_data.py -- canonical CIFAR-10 loader for the whole cifar10_vit flow.

Every other piece of software in this directory (QAT training, the integer
golden model, the testbench vector generator) imports the dataset from here,
so there is exactly one definition of pixel order and normalisation and they
cannot silently drift apart.

Reads the *binary* CIFAR-10 distribution (cifar-10-binary.tar.gz), not the
pickled Python one, because the RTL testbenches want raw bytes anyway:

    <1 byte label> <1024 bytes R> <1024 bytes G> <1024 bytes B>

repeated 10,000 times per file, rows before columns within each channel.

Layout on disk (relative to the repository root):

    data/cifar-10-batches-bin/
        data_batch_1.bin ... data_batch_5.bin   (50,000 train)
        test_batch.bin                          (10,000 test)
        batches.meta.txt

Usage:
    from cifar_data import load_test, load_train, CLASS_NAMES
    images, labels = load_test()        # uint8 (N,3,32,32) in CHW order, int labels
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Dataset constants ───────────────────────────────────────────────────────
IMG_H = 32
IMG_W = 32
IMG_C = 3
IMG_PIXELS = IMG_H * IMG_W          # 1024 per channel
IMG_BYTES = IMG_C * IMG_PIXELS      # 3072
RECORD_BYTES = 1 + IMG_BYTES        # 3073 (label + image)
NUM_CLASSES = 10

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Standard CIFAR-10 channel statistics, used by the float training path.
# The integer hardware path derives its own fixed-point constants from these.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_TRAIN_FILES = tuple(f"data_batch_{i}.bin" for i in range(1, 6))
_TEST_FILES = ("test_batch.bin",)


def data_dir() -> Path:
    """Locate cifar-10-batches-bin/, allowing an override via $CIFAR10_DIR."""
    env = os.environ.get("CIFAR10_DIR")
    if env:
        return Path(env)
    # cifar10_vit/sw/cifar_data.py -> repo root is two levels up
    return Path(__file__).resolve().parents[2] / "data" / "cifar-10-batches-bin"


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if raw.size % RECORD_BYTES != 0:
        raise ValueError(
            f"{path}: size {raw.size} is not a multiple of {RECORD_BYTES}; "
            "the file is truncated or not the binary CIFAR-10 distribution"
        )
    records = raw.reshape(-1, RECORD_BYTES)
    labels = records[:, 0].astype(np.int64)
    images = records[:, 1:].reshape(-1, IMG_C, IMG_H, IMG_W)   # already CHW
    return images, labels


def _load(files) -> tuple[np.ndarray, np.ndarray]:
    d = data_dir()
    missing = [f for f in files if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing CIFAR-10 binary batches {missing} in {d}.\n"
            "Fetch them with:\n"
            "  mkdir -p data && cd data \\\n"
            "    && curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz \\\n"
            "    && tar xzf cifar-10-binary.tar.gz\n"
            "or point $CIFAR10_DIR at an existing cifar-10-batches-bin directory."
        )
    parts = [_read_batch(d / f) for f in files]
    images = np.concatenate([p[0] for p in parts], axis=0)
    labels = np.concatenate([p[1] for p in parts], axis=0)
    return images, labels


def load_train() -> tuple[np.ndarray, np.ndarray]:
    """50,000 training images as uint8 (N,3,32,32) CHW, plus int64 labels."""
    return _load(_TRAIN_FILES)


def load_test() -> tuple[np.ndarray, np.ndarray]:
    """10,000 test images as uint8 (N,3,32,32) CHW, plus int64 labels."""
    return _load(_TEST_FILES)


def to_float_normalised(images: np.ndarray) -> np.ndarray:
    """uint8 (N,3,32,32) -> float32, scaled to [0,1] then mean/std normalised."""
    x = images.astype(np.float32) / 255.0
    mean = np.asarray(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def flat_bytes(image: np.ndarray) -> bytes:
    """One uint8 (3,32,32) CHW image -> the exact 3072-byte stream the RTL sees.

    Channel-major (all of R, then G, then B), row-major within a channel --
    identical to the on-disk CIFAR-10 record layout, so the UART/testbench feed
    and the dataset file agree by construction.
    """
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    if arr.shape != (IMG_C, IMG_H, IMG_W):
        raise ValueError(f"expected (3,32,32) CHW image, got {arr.shape}")
    return arr.tobytes()


if __name__ == "__main__":
    imgs, lbls = load_test()
    print(f"test set: images {imgs.shape} {imgs.dtype}, labels {lbls.shape}")
    counts = np.bincount(lbls, minlength=NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:<11} {counts[i]}")
    assert imgs.shape == (10000, 3, 32, 32) and counts.min() == 1000
    print(f"first image label {lbls[0]} ({CLASS_NAMES[lbls[0]]}), "
          f"{len(flat_bytes(imgs[0]))} bytes to the RTL")
    print("OK")
