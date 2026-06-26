"""run_demo.py — hardware-free end-to-end demo: synthetic EEG -> filter -> epoch
-> features -> cross-validated spacebar classifier. Proves the pipeline works."""
from acquire import SyntheticAcquirer, build_events
from preprocess import epoch_pipeline
from features import make_features
from model import evaluate


def main():
    print("== Unicorn EEG spacebar-detection demo (synthetic) ==")
    data, ts, presses = SyntheticAcquirer(duration_s=180, n_presses=140, seed=7).get_data()
    events = build_events(ts[presses], ts)
    print(f"  raw {data.shape}, {len(presses)} presses, {len(events)} epochs")
    X, y = epoch_pipeline(data, ts, events, use_mne=False)
    print(f"  epochs {X.shape}  (space={int((y==1).sum())}, rest={int((y==0).sum())})")
    for kind in ("bandpower", "all"):
        F, names = make_features(X, kind)
        for mdl in ("lda", "rf"):
            r = evaluate(F, y, mdl)
            print(f"  [{kind:9s} | {mdl:3s}] acc={r['acc']:.3f}+-{r['acc_std']:.3f} "
                  f"auc={r['auc']:.3f}  ({r['n']} epochs, {len(names)} feats)")
    print("Done. (chance = 0.50)")


if __name__ == "__main__":
    main()
