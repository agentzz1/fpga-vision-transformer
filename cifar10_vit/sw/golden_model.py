#!/usr/bin/env python3
"""
golden_model.py -- bit-exact integer reference for the CIFAR-10 ViT accelerator.

This is the contract the RTL is verified against. Every operation here is
integer-only and corresponds one-to-one with a hardware stage, including the
exact shift, saturation and lookup-table index arithmetic. If the RTL and this
file ever disagree on any intermediate value, one of them is wrong.

Floating point appears in exactly two places, both outside the inference path:
building the GELU and exp tables (baked into the bitstream at synthesis time)
and normalising input pixels (done on the host before bytes are sent).

Usage
-----
    from golden_model import GoldenModel, Weights
    m = GoldenModel(Weights.random(seed=0))
    pred = m.predict(img_u8)              # img_u8: uint8 (3,32,32)
    pred, trace = m.predict(img_u8, trace=True)   # every intermediate tensor

The trace is what the RTL testbench compares against, stage by stage, so a
mismatch points at the offending module instead of just a wrong final class.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cifar_data import CIFAR10_MEAN, CIFAR10_STD
from model_config import CONFIG, ModelConfig
from quant_ops import (
    Q_BITS, Q_SCALE, add_sat, argmax_first, gelu_q, layernorm_q, matmul_q,
    quantize_pixels, requantize, sat8, softmax_q,
)


# ── Weights ─────────────────────────────────────────────────────────────────
@dataclass
class Weights:
    """int8 parameters, stored (out, in) so each row is one output neuron.

    That layout matches how the weight ROM is addressed in hardware: walking a
    row streams the K values a single output channel needs.
    """
    patch_proj_w: np.ndarray        # (d_model, patch_dim)
    patch_proj_b: np.ndarray        # (d_model,)
    pos_embed: np.ndarray           # (seq_len, d_model)
    wq: np.ndarray                  # (d_model, d_model)
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w1: np.ndarray                  # (d_ff, d_model)
    b1: np.ndarray                  # (d_ff,)
    w2: np.ndarray                  # (d_model, d_ff)
    b2: np.ndarray                  # (d_model,)
    cls_w: np.ndarray               # (n_classes, d_model)
    cls_b: np.ndarray               # (n_classes,)

    _SHAPES = (
        ("patch_proj_w", ("d_model", "patch_dim")),
        ("patch_proj_b", ("d_model",)),
        ("pos_embed",    ("seq_len", "d_model")),
        ("wq",           ("d_model", "d_model")),
        ("wk",           ("d_model", "d_model")),
        ("wv",           ("d_model", "d_model")),
        ("wo",           ("d_model", "d_model")),
        ("w1",           ("d_ff", "d_model")),
        ("b1",           ("d_ff",)),
        ("w2",           ("d_model", "d_ff")),
        ("b2",           ("d_model",)),
        ("cls_w",        ("n_classes", "d_model")),
        ("cls_b",        ("n_classes",)),
    )

    @staticmethod
    def _dims(cfg: ModelConfig, names) -> tuple[int, ...]:
        return tuple(getattr(cfg, n) for n in names)

    @classmethod
    def random(cls, seed: int = 0, cfg: ModelConfig = CONFIG,
               scale: int = 40) -> "Weights":
        """Small random weights, for bringing the RTL up before training.

        Deliberately narrower than the full int8 range: real QAT weights
        cluster near zero, and saturating every accumulator would hide
        arithmetic bugs behind a wall of clipped values.
        """
        rng = np.random.default_rng(seed)
        kw = {}
        for name, dims in cls._SHAPES:
            shape = cls._dims(cfg, dims)
            kw[name] = rng.integers(-scale, scale + 1, size=shape).astype(np.int32)
        return cls(**kw)

    @classmethod
    def load(cls, directory: str | Path, cfg: ModelConfig = CONFIG) -> "Weights":
        """Load flat int8 .bin files exported by the QAT training script."""
        d = Path(directory)
        kw = {}
        for name, dims in cls._SHAPES:
            shape = cls._dims(cfg, dims)
            path = d / f"{name}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"missing weight file {path}")
            raw = np.frombuffer(path.read_bytes(), dtype=np.int8).astype(np.int32)
            expected = int(np.prod(shape))
            if raw.size != expected:
                raise ValueError(
                    f"{path}: expected {expected} bytes for shape {shape}, "
                    f"got {raw.size}"
                )
            kw[name] = raw.reshape(shape)
        return cls(**kw)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, _ in self._SHAPES:
            arr = getattr(self, name)
            if arr.min() < -128 or arr.max() > 127:
                raise ValueError(f"{name} does not fit in int8")
            (d / f"{name}.bin").write_bytes(arr.astype(np.int8).tobytes())

    def validate(self, cfg: ModelConfig = CONFIG) -> None:
        for name, dims in self._SHAPES:
            want = self._dims(cfg, dims)
            got = getattr(self, name).shape
            if got != want:
                raise ValueError(f"{name}: expected shape {want}, got {got}")


# ── Patch extraction ────────────────────────────────────────────────────────
def patchify(img_q: np.ndarray, cfg: ModelConfig = CONFIG) -> np.ndarray:
    """int8 (C,H,W) image -> (seq_len, patch_dim) token matrix.

    Patches are taken in raster order across the grid. Within a patch the
    values run channel-major, then row, then column, so element index
    (c * patch * patch) + (i * patch) + j. The weight ROM is generated against
    this same ordering, so the flattening convention only has to be consistent,
    not any particular one -- but it must be identical here, in the trainer,
    and in the RTL address generator, which is why it lives in one function.
    """
    p, g = cfg.patch, cfg.grid
    out = np.zeros((cfg.seq_len, cfg.patch_dim), dtype=np.int32)
    for t in range(cfg.seq_len):
        row, col = divmod(t, g)
        block = img_q[:, row * p:(row + 1) * p, col * p:(col + 1) * p]
        out[t] = block.reshape(-1)          # C, then H, then W
    return out


# ── The model ───────────────────────────────────────────────────────────────
class GoldenModel:
    def __init__(self, weights: Weights, cfg: ModelConfig = CONFIG):
        weights.validate(cfg)
        self.w = weights
        self.cfg = cfg

    def predict(self, img_u8: np.ndarray, trace: bool = False):
        """Run inference on one uint8 (3,32,32) CHW image.

        Returns the predicted class, or (class, trace_dict) when trace=True.
        """
        cfg, w = self.cfg, self.w
        tr: dict[str, np.ndarray] = {}

        def keep(name: str, value: np.ndarray) -> np.ndarray:
            if trace:
                tr[name] = value.copy()
            return value

        # 1. Quantise pixels and cut them into tokens.
        img_q = keep("pixels_q", quantize_pixels(img_u8, CIFAR10_MEAN, CIFAR10_STD))
        patches = keep("patches", patchify(img_q, cfg))

        # 2. Patch projection + learned positional embedding.
        #    The projection requantises first and the position add saturates
        #    afterwards, matching the RTL's two-step write into the token RAM.
        proj = matmul_q(patches, w.patch_proj_w.T, w.patch_proj_b)
        x = keep("tokens", add_sat(proj, w.pos_embed))

        # 3. Q, K, V. In hardware these are one fused (d_model -> 3*d_model)
        #    projection so the token matrix is read from RAM once instead of
        #    three times; splitting them here is only for readability.
        q = keep("q", matmul_q(x, w.wq.T))
        k = keep("k", matmul_q(x, w.wk.T))
        v = keep("v", matmul_q(x, w.wv.T))

        # 4. Attention scores. ATTN_SHIFT folds the 1/sqrt(head_dim) scaling
        #    into the requantisation shift, so it costs nothing.
        scores = keep("scores", requantize(q.astype(np.int64) @ k.T.astype(np.int64),
                                           cfg.attn_shift))
        probs = keep("probs", softmax_q(scores))

        # 5. Weighted sum of values, then the output projection.
        ctx = keep("ctx", matmul_q(probs, v))
        attn = keep("attn_out", matmul_q(ctx, w.wo.T))

        # 6. Residual + LayerNorm.
        y1 = keep("y1", layernorm_q(add_sat(x, attn), cfg.ln_headroom))

        # 7. Feed-forward network with a ROM-lookup GELU.
        h = keep("fc1", matmul_q(y1, w.w1.T, w.b1))
        h = keep("gelu", gelu_q(h))
        ffn = keep("fc2", matmul_q(h, w.w2.T, w.b2))

        # 8. Second residual + LayerNorm.
        y2 = keep("y2", layernorm_q(add_sat(y1, ffn), cfg.ln_headroom))

        # 9. Global average pool (a shift, since seq_len is a power of two),
        #    then the classifier.
        gap = keep("gap", sat8(y2.astype(np.int64).sum(axis=0) >> cfg.gap_shift))
        logits = keep("logits",
                      matmul_q(gap.reshape(1, -1), w.cls_w.T, w.cls_b)[0])

        pred = argmax_first(logits)
        return (pred, tr) if trace else pred

    def predict_batch(self, imgs_u8: np.ndarray) -> np.ndarray:
        return np.array([self.predict(im) for im in imgs_u8], dtype=np.int64)


# ── Self-test ───────────────────────────────────────────────────────────────
def _test() -> None:
    cfg = CONFIG
    w = Weights.random(seed=1)
    m = GoldenModel(w)

    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(3, 32, 32)).astype(np.uint8)

    pred, tr = m.predict(img, trace=True)
    assert 0 <= pred < cfg.n_classes

    # Every intermediate must be a valid int8 tensor of the expected shape --
    # a shape or range slip here would silently become an RTL mismatch later.
    expect = {
        "pixels_q": (cfg.img_c, cfg.img_h, cfg.img_w),
        "patches": (cfg.seq_len, cfg.patch_dim),
        "tokens": (cfg.seq_len, cfg.d_model),
        "q": (cfg.seq_len, cfg.d_model),
        "k": (cfg.seq_len, cfg.d_model),
        "v": (cfg.seq_len, cfg.d_model),
        "scores": (cfg.seq_len, cfg.seq_len),
        "probs": (cfg.seq_len, cfg.seq_len),
        "ctx": (cfg.seq_len, cfg.d_model),
        "attn_out": (cfg.seq_len, cfg.d_model),
        "y1": (cfg.seq_len, cfg.d_model),
        "fc1": (cfg.seq_len, cfg.d_ff),
        "gelu": (cfg.seq_len, cfg.d_ff),
        "fc2": (cfg.seq_len, cfg.d_model),
        "y2": (cfg.seq_len, cfg.d_model),
        "gap": (cfg.d_model,),
        "logits": (cfg.n_classes,),
    }
    for name, shape in expect.items():
        assert name in tr, f"trace missing {name}"
        assert tr[name].shape == shape, f"{name}: {tr[name].shape} != {shape}"
        lo, hi = int(tr[name].min()), int(tr[name].max())
        assert -128 <= lo and hi <= 127, f"{name} out of int8 range: [{lo},{hi}]"

    # Determinism: the same input must give the same answer every time.
    assert m.predict(img) == pred

    # The model must actually respond to its input rather than emitting a
    # constant -- a stuck datapath would still pass every check above.
    preds = {m.predict(rng.integers(0, 256, size=(3, 32, 32)).astype(np.uint8))
             for _ in range(40)}
    assert len(preds) > 1, f"model always predicts the same class: {preds}"

    # Weight round-trip through the on-disk format used by the RTL ROM.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        w.save(td)
        w2 = Weights.load(td)
        for name, _ in Weights._SHAPES:
            assert np.array_equal(getattr(w, name), getattr(w2, name)), name
        assert GoldenModel(w2).predict(img) == pred

    print(f"golden_model self-tests OK (sample prediction {pred}, "
          f"{len(preds)} distinct classes over 40 random inputs)")


if __name__ == "__main__":
    _test()
