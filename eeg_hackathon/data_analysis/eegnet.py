"""eegnet.py — EEGNet (Lawhern et al. 2018), the compact-CNN SOTA baseline for
P300 / motor-imagery / MRCP. Few parameters -> trains on hackathon-sized data.
Requires torch. Pure-numpy pipelines (CSP/Riemann/xDAWN) remain the no-torch path.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold

try:
    import torch, torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:
    class EEGNet(nn.Module):
        def __init__(self, n_ch, n_time, n_classes=2, F1=8, D=2, F2=16, kern=64, p=0.5):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),   # depthwise
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(p))
            self.b2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                 # separable
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(p))
            with torch.no_grad():
                d = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_time))).numel()
            self.head = nn.Linear(d, n_classes)

        def forward(self, x):
            x = self.b2(self.b1(x))
            return self.head(x.flatten(1))


def _fit_eval(Xtr, ytr, Xte, yte, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    # determinism for reproducible RUN_LOG numbers across machines/BLAS
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    n_ch, n_time = Xtr.shape[1], Xtr.shape[2]
    net = EEGNet(n_ch, n_time, len(np.unique(ytr)))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr[:, None], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); out = net(Xt); loss = lossf(out, yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte[:, None], dtype=torch.float32)).argmax(1).numpy()
    return (pred == yte).mean()


def evaluate(X, y, folds=5, epochs=60):
    if not _HAS_TORCH:
        return {"method": "EEGNet", "error": "torch not installed"}
    X = np.asarray(X, float); y = np.asarray(y)
    # per-trial z-score (stabilises training)
    X = (X - X.mean((1, 2), keepdims=True)) / (X.std((1, 2), keepdims=True) + 1e-6)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    accs = [_fit_eval(X[tr], y[tr], X[te], y[te], epochs) for tr, te in skf.split(X, y)]
    return {"method": "EEGNet(CNN)", "acc": float(np.mean(accs)),
            "std": float(np.std(accs)), "folds": folds, "n": len(y)}


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from mi_pipeline import _synth_mi
    X, y = _synth_mi(n_per=50)
    print("EEGNet on synthetic MI:", evaluate(X, y, epochs=40))
