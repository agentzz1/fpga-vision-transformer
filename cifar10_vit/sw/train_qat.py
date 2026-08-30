#!/usr/bin/env python3
"""
train_qat.py -- quantisation-aware training for the CIFAR-10 ViT accelerator.

The network is trained with the hardware's integer arithmetic in the forward
pass, so the weights adapt to the accelerator's approximations instead of being
damaged by them afterwards. Everything is carried in "integer units": tensors
hold integer-valued floats where the value v represents v/128 in Q1.7, exactly
matching golden_model.py. Export is then a cast, not a calibration step.

Straight-through estimators
---------------------------
Three operations in the datapath have no useful derivative, so each uses a
surrogate: the forward pass computes what the hardware computes, and the
backward pass uses the gradient of a smooth stand-in.

  rounding / flooring   forward: floor,          backward: identity
  ROM-lookup GELU       forward: table lookup,   backward: true GELU
  integer softmax       forward: integer softmax, backward: real softmax
  leading-one LayerNorm forward: shift-normalise, backward: real LayerNorm

This is the same approach the MNIST design in this repository used, where the
float-to-integer accuracy gap came out at 0.06 points.

Usage
-----
    python3 train_qat.py --epochs 30 --out ../weights_int8
    python3 train_qat.py --epochs 1 --limit 2000     # quick smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_data import CIFAR10_MEAN, CIFAR10_STD, load_test, load_train
from model_config import CONFIG
from quant_ops import EXP_LUT_Q16, GELU_LUT, Q_SCALE, SM_LUT_DEPTH

CFG = CONFIG
INT8_MIN, INT8_MAX = -128, 127


# ── Straight-through primitives ─────────────────────────────────────────────
class _FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        return g


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


floor_ste = _FloorSTE.apply
round_ste = _RoundSTE.apply


def clamp_ste(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp in the forward pass, pass the gradient through unchanged.

    Letting gradient flow through saturated values keeps a weight that has
    clipped from becoming permanently stuck, which matters here because Q1.7
    saturates often.
    """
    return x + (x.clamp(lo, hi) - x).detach()


def quantize_weight(w: torch.Tensor) -> torch.Tensor:
    """Latent float parameter -> int8-valued tensor, differentiably."""
    return clamp_ste(round_ste(w), INT8_MIN, INT8_MAX)


def requantize(acc: torch.Tensor, shift: int) -> torch.Tensor:
    """Wide accumulator -> int8, matching `acc >> shift` then saturate.

    Division by a power of two followed by floor reproduces the arithmetic
    right shift, including its round-toward-negative-infinity behaviour on
    negative values -- which `trunc` would get wrong.
    """
    return clamp_ste(floor_ste(acc / (2 ** shift)), INT8_MIN, INT8_MAX)


# ── Hardware-matched nonlinearities ─────────────────────────────────────────
class HWGelu(nn.Module):
    """256-entry ROM GELU. Forward is the table; backward is the real GELU."""

    def __init__(self):
        super().__init__()
        self.register_buffer("lut", torch.tensor(GELU_LUT, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = (x.detach().to(torch.int64) & 0xFF).clamp(0, 255)
        hard = self.lut[idx]
        soft = F.gelu(x / Q_SCALE) * Q_SCALE       # smooth surrogate
        return soft + (hard - soft).detach()


class HWSoftmax(nn.Module):
    """Integer softmax over the last dimension, Q1.7 in and out.

    The forward pass reproduces softmax.vhd exactly, including the clamped
    difference, the reversed table index and the floor division. The backward
    pass uses an ordinary softmax so the attention logits still receive a
    sensible gradient.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("exp_lut",
                             torch.tensor(EXP_LUT_Q16, dtype=torch.float64))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            s = scores.detach().to(torch.float64)
            diff = s - s.max(dim=-1, keepdim=True).values          # <= 0
            magnitude = torch.clamp(-diff, max=float(Q_SCALE))
            scaled = torch.floor(magnitude * (SM_LUT_DEPTH - 1)
                                 / (10.0 * Q_SCALE))
            idx = torch.clamp((SM_LUT_DEPTH - 1) - scaled,
                              0, SM_LUT_DEPTH - 1).to(torch.int64)
            e = torch.floor(self.exp_lut[idx] / 512.0).clamp(max=127.0)
            e = torch.where(diff >= 0, torch.full_like(e, 127.0), e)
            denom = e.sum(dim=-1, keepdim=True)
            safe = denom > 0
            hard = torch.where(
                safe,
                torch.floor(e * Q_SCALE / torch.clamp(denom, min=1.0)),
                torch.full_like(e, Q_SCALE / scores.shape[-1]),
            ).clamp(INT8_MIN, INT8_MAX).to(scores.dtype)

        soft = torch.softmax(scores / Q_SCALE, dim=-1) * Q_SCALE
        return soft + (hard - soft).detach()


class HWLayerNorm(nn.Module):
    """Leading-one-detection LayerNorm, matching layernorm_q.

    The hardware replaces 1/sqrt(var) with a shift by (log2(var)+1)/2, so the
    effective divisor is rounded to a power of two. Training with that in the
    loop lets the network absorb the coarseness; the backward pass uses a real
    LayerNorm so gradients stay well-scaled.
    """

    def __init__(self, d_model: int, headroom: int):
        super().__init__()
        self.var_bits = int(np.log2(d_model))
        assert 2 ** self.var_bits == d_model, "d_model must be a power of two"
        self.frac = CFG.q_bits - headroom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xi = x.detach().to(torch.float64)
            mean = torch.floor(xi.sum(-1, keepdim=True) / 2 ** self.var_bits)
            mean_sq = torch.floor((xi * xi).sum(-1, keepdim=True)
                                  / 2 ** self.var_bits)
            var = torch.clamp(mean_sq - mean * mean, min=0.0)
            # leading_one(var): index of the most significant set bit, 0 for 0.
            lod = torch.where(var > 0, torch.floor(torch.log2(
                torch.clamp(var, min=1.0))), torch.zeros_like(var))
            shift = torch.floor((lod + 1) / 2)
            net = shift - self.frac
            diff = xi - mean
            hard = torch.where(
                net >= 0,
                torch.floor(diff / torch.pow(2.0, net)),
                diff * torch.pow(2.0, -net),
            ).clamp(INT8_MIN, INT8_MAX).to(x.dtype)

        soft = F.layer_norm(x, (x.shape[-1],)) * (Q_SCALE / 2 ** CFG.ln_headroom)
        return soft + (hard - soft).detach()


# ── The model ───────────────────────────────────────────────────────────────
class QatViT(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        d, f, s, pd = cfg.d_model, cfg.d_ff, cfg.seq_len, cfg.patch_dim

        def param(*shape, scale=8.0):
            # Latent weights live on the integer scale, so a small spread here
            # corresponds to a sensible spread once rounded to int8.
            return nn.Parameter(torch.randn(*shape) * scale)

        self.patch_w = param(d, pd, scale=4.0)
        self.patch_b = param(d, scale=1.0)
        self.pos = param(s, d, scale=4.0)
        self.wq, self.wk = param(d, d), param(d, d)
        self.wv, self.wo = param(d, d), param(d, d)
        self.w1, self.b1 = param(f, d), param(f, scale=1.0)
        self.w2, self.b2 = param(d, f), param(d, scale=1.0)
        self.cls_w, self.cls_b = param(cfg.n_classes, d), param(cfg.n_classes,
                                                                scale=1.0)

        self.gelu = HWGelu()
        self.softmax = HWSoftmax()
        self.ln1 = HWLayerNorm(d, cfg.ln_headroom)
        self.ln2 = HWLayerNorm(d, cfg.ln_headroom)

    def matmul(self, a, w, bias=None, shift=None):
        """Quantised (B,M,K) x (N,K)^T -> (B,M,N), requantised to int8."""
        acc = a @ quantize_weight(w).t()
        if bias is not None:
            acc = acc + quantize_weight(bias) * Q_SCALE
        return requantize(acc, CFG.q_bits if shift is None else shift)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, seq_len, patch_dim) int8-valued. Returns logits."""
        cfg = self.cfg

        x = self.matmul(patches, self.patch_w, self.patch_b)
        x = clamp_ste(x + quantize_weight(self.pos), INT8_MIN, INT8_MAX)

        q = self.matmul(x, self.wq)
        k = self.matmul(x, self.wk)
        v = self.matmul(x, self.wv)

        scores = requantize(q @ k.transpose(1, 2), cfg.attn_shift)
        probs = self.softmax(scores)

        ctx = requantize(probs @ v, cfg.q_bits)
        attn = self.matmul(ctx, self.wo)

        y1 = self.ln1(clamp_ste(x + attn, INT8_MIN, INT8_MAX))

        h = self.gelu(self.matmul(y1, self.w1, self.b1))
        ffn = self.matmul(h, self.w2, self.b2)

        y2 = self.ln2(clamp_ste(y1 + ffn, INT8_MIN, INT8_MAX))

        gap = clamp_ste(floor_ste(y2.sum(dim=1) / 2 ** cfg.gap_shift),
                        INT8_MIN, INT8_MAX)
        logits = self.matmul(gap.unsqueeze(1), self.cls_w, self.cls_b).squeeze(1)

        # The classifier output is int8, so its dynamic range is tiny compared
        # with what cross-entropy expects. Scaling before the loss keeps the
        # gradient from vanishing without changing the argmax the hardware
        # computes.
        return logits / 8.0

    def export_int8(self) -> dict[str, np.ndarray]:
        """Round every parameter to the int8 arrays golden_model expects."""
        out = {}
        for name, p in [
            ("patch_proj_w", self.patch_w), ("patch_proj_b", self.patch_b),
            ("pos_embed", self.pos),
            ("wq", self.wq), ("wk", self.wk), ("wv", self.wv), ("wo", self.wo),
            ("w1", self.w1), ("b1", self.b1),
            ("w2", self.w2), ("b2", self.b2),
            ("cls_w", self.cls_w), ("cls_b", self.cls_b),
        ]:
            out[name] = (p.detach().round().clamp(INT8_MIN, INT8_MAX)
                         .cpu().numpy().astype(np.int32))
        return out


# ── Data preparation ────────────────────────────────────────────────────────
def quantize_and_patchify(images: np.ndarray) -> np.ndarray:
    """uint8 (N,3,32,32) -> int8-valued (N, seq_len, patch_dim) float32.

    Uses the same normalisation and the same patch flattening order as
    golden_model.patchify, so the trained weights index the pixels the way the
    hardware will present them.
    """
    cfg = CFG
    x = images.astype(np.float32) / 255.0
    mean = np.asarray(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    x = np.rint((x - mean) / std * Q_SCALE).clip(INT8_MIN, INT8_MAX)

    n, p, g = x.shape[0], cfg.patch, cfg.grid
    out = np.zeros((n, cfg.seq_len, cfg.patch_dim), dtype=np.float32)
    for t in range(cfg.seq_len):
        row, col = divmod(t, g)
        block = x[:, :, row * p:(row + 1) * p, col * p:(col + 1) * p]
        out[:, t, :] = block.reshape(n, -1)
    return out


def augment(batch: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip, applied on the image before patchifying."""
    flip = torch.rand(images.shape[0]) < 0.5
    images = images.clone()
    images[flip] = torch.flip(images[flip], dims=[-1])
    return images


# ── Training ────────────────────────────────────────────────────────────────
def evaluate(model: QatViT, patches: torch.Tensor, labels: torch.Tensor,
             batch: int = 500) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, patches.shape[0], batch):
            logits = model(patches[i:i + batch])
            correct += (logits.argmax(-1) == labels[i:i + batch]).sum().item()
    model.train()
    return correct / patches.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0, help="use N training images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "weights_int8")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    print("loading CIFAR-10 ...", flush=True)
    tr_img, tr_lab = load_train()
    te_img, te_lab = load_test()
    if args.limit:
        tr_img, tr_lab = tr_img[:args.limit], tr_lab[:args.limit]

    # Keep the raw uint8 images for flip augmentation, and pre-patchify the
    # test set once since it never changes.
    tr_raw = torch.from_numpy(tr_img.astype(np.float32))
    tr_y = torch.from_numpy(tr_lab)
    te_x = torch.from_numpy(quantize_and_patchify(te_img))
    te_y = torch.from_numpy(te_lab)
    print(f"train {tr_raw.shape[0]}, test {te_x.shape[0]}", flush=True)

    model = QatViT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params:,} (int8 bytes)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        total_steps=args.epochs * max(1, tr_raw.shape[0] // args.batch),
        pct_start=0.25)

    best = 0.0
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(tr_raw.shape[0])
        total_loss, seen = 0.0, 0

        for i in range(0, tr_raw.shape[0] - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            imgs = tr_raw[idx]
            flip = torch.rand(imgs.shape[0]) < 0.5
            imgs[flip] = torch.flip(imgs[flip], dims=[-1])

            xb = torch.from_numpy(
                quantize_and_patchify(imgs.numpy().astype(np.uint8)))
            yb = tr_y[idx]

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            total_loss += loss.item() * yb.shape[0]
            seen += yb.shape[0]

        acc = evaluate(model, te_x, te_y)
        dt = time.time() - t0
        star = ""
        if acc > best:
            best = acc
            np.savez(args.out / "qat_best.npz", **model.export_int8())
            star = " *"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {total_loss / max(seen, 1):.4f}"
              f"  test {acc * 100:.2f}%  best {best * 100:.2f}%  {dt:.0f}s{star}",
              flush=True)

    # Export the best checkpoint as the flat .bin files the golden model and
    # the RTL weight ROM both read.
    blob = np.load(args.out / "qat_best.npz")
    for name in blob.files:
        arr = blob[name].astype(np.int8)
        (args.out / f"{name}.bin").write_bytes(arr.tobytes())
    print(f"\nbest QAT test accuracy {best * 100:.2f}%")
    print(f"exported int8 weights to {args.out}")


if __name__ == "__main__":
    main()
