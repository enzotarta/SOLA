"""Per-class deltas and pairwise Spearman correlations from behavioral_mapping.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


IMAGENET_LABEL_URL = (
    "https://raw.githubusercontent.com/raghakot/keras-vis/master/"
    "resources/imagenet_class_index.json"
)


def load_imagenet_labels():
    """Return {class_idx: label_string}. Tries local cache, then URL."""
    cache = Path.home() / ".cache" / "imagenet_class_index.json"
    if cache.exists():
        d = json.loads(cache.read_text())
    else:
        try:
            import urllib.request
            with urllib.request.urlopen(IMAGENET_LABEL_URL, timeout=10) as r:
                d = json.loads(r.read())
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(d))
        except Exception as e:
            print(f"  (couldn't fetch ImageNet labels: {e})")
            return None
    return {int(k): v[1] for k, v in d.items()}


def per_class_acc(per_class_correct, per_class_total):
    correct = np.asarray(per_class_correct, dtype=np.float64)
    total = np.asarray(per_class_total, dtype=np.float64)
    return correct / np.maximum(total, 1.0)


def topk_classes(delta, k, labels=None):
    """Return list of (class_idx, label, delta) sorted by delta sign."""
    idxs = np.argsort(delta)
    most_neg = idxs[:k]
    most_pos = idxs[-k:][::-1]
    def fmt(i):
        lab = labels.get(int(i), "?") if labels else "?"
        return (int(i), lab, float(delta[i]))
    return [fmt(i) for i in most_neg], [fmt(i) for i in most_pos]


def report(input_json, k=15):
    d = json.loads(Path(input_json).read_text())
    print(f"\n{input_json}  ({d['model']})")
    if "baseline" not in d["results"]:
        print("  no baseline; cannot compute deltas")
        return d

    base_correct = d["results"]["baseline"]["per_class_correct"]
    base_total = d["results"]["baseline"]["per_class_total"]
    base_acc = per_class_acc(base_correct, base_total)
    print(f"baseline aggregate top-1 = "
          f"{d['results']['baseline']['aggregate_top1']*100:.2f}%  "
          f"n={d['results']['baseline']['n_eval']}")

    labels = load_imagenet_labels()
    deltas = {}
    for name, r in d["results"].items():
        if name == "baseline": continue
        cur = per_class_acc(r["per_class_correct"], r["per_class_total"])
        delta = cur - base_acc
        deltas[name] = delta
        agg = r["aggregate_top1"]
        agg_base = d["results"]["baseline"]["aggregate_top1"]
        print(f"\n{name}")
        print(f"  aggregate top-1 = {agg*100:.2f}%  "
              f"(delta  = {(agg-agg_base)*100:+.2f} pp)")
        n_down5 = int((delta <= -0.05).sum())
        n_up5 = int((delta >= 0.05).sum())
        n_down10 = int((delta <= -0.10).sum())
        n_up10 = int((delta >= 0.10).sum())
        print(f"  classes with delta  <= -5 pp: {n_down5}  (<= -10 pp: {n_down10})")
        print(f"  classes with delta  >= +5 pp: {n_up5}  (>= +10 pp: {n_up10})")
        print(f"  delta  distribution: mean={delta.mean()*100:+.2f} pp  "
              f"median={np.median(delta)*100:+.2f} pp  "
              f"std={delta.std()*100:.2f}")
        worst, best = topk_classes(delta, k, labels)
        print(f"  top-{k} most degraded:")
        for idx, lab, dv in worst:
            print(f"    [{idx:>4}] {dv*100:+6.2f} pp  {lab}")
        print(f"  top-{k} most improved:")
        for idx, lab, dv in best:
            print(f"    [{idx:>4}] {dv*100:+6.2f} pp  {lab}")

    # Pairwise Spearman of per-class deltas across conditions: whether two
    # operator classes hurt similar classes.
    if len(deltas) >= 2:
        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None
        if spearmanr is not None:
            names = list(deltas.keys())
            print(f"\nPairwise Spearman correlation of per-class delta ")
            print(f"  (low = different capability profile)")
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    rho, p = spearmanr(deltas[names[i]], deltas[names[j]])
                    print(f"  {names[i]:<22} vs {names[j]:<22}  "
                          f"rho={rho:+.3f}  p={p:.2e}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="behavioral_*.json files to analyze")
    ap.add_argument("-k", type=int, default=15)
    args = ap.parse_args()
    for fn in args.input:
        report(fn, k=args.k)


if __name__ == "__main__":
    main()
