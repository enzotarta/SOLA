"""Stage-2 of the taxonomy: reassign the residual `other` class.

Heads with eff_rank_90 <= 40 go to near_global, the others to diffuse_local.
The input taxonomy is kept as taxonomy_*_with_other.json.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

K_THRESHOLD = 40

PAIRS = [
    ("taxonomy_vitb_in21k.json", "svd_spectrum_vitb.json"),
    ("taxonomy_vitb.json",       "svd_spectrum_vitb.json"),
    ("taxonomy_vitb_in1k_pure.json", "svd_spectrum_vitb_in1k.json"),
    ("taxonomy_vits.json",       "svd_spectrum_vits.json"),
    ("taxonomy_deit.json",       "svd_spectrum_deit_tiny.json"),
    ("taxonomy_deit_s.json",     "svd_spectrum_deit_small.json"),
    ("taxonomy_mae.json",        "svd_spectrum_mae.json"),
    ("taxonomy_vitb_dino.json",  "svd_spectrum_dino.json"),
    ("taxonomy_dinov2.json",     "svd_spectrum_dinov2.json"),
    ("taxonomy_clip.json",       "svd_spectrum_clip.json"),
    ("taxonomy_siglip.json",     "svd_spectrum_siglip.json"),
]

EXP = Path(__file__).resolve().parent.parent / "results"


def redistribute_one(tax_name: str, spec_name: str):
    tax_path = EXP / tax_name
    spec_path = EXP / spec_name
    if not tax_path.exists() or not spec_path.exists():
        print(f"  [skip] {tax_name}: missing files")
        return
    backup = tax_path.with_name(tax_path.stem + "_with_other.json")
    if not backup.exists():
        shutil.copy(tax_path, backup)

    tax = json.loads(tax_path.read_text())
    spec = json.loads(spec_path.read_text())
    er90 = {(r["layer"], r["head"]): r["eff_rank_90"]
            for r in spec["per_head"]}

    n_other = n_to_ng = n_to_dl = n_unmapped = 0
    for r in tax["results"]:
        if r["category"] != "other":
            continue
        n_other += 1
        k = er90.get((r["layer"], r["head"]))
        if k is None:
            n_unmapped += 1
            continue
        if k <= K_THRESHOLD:
            r["category"] = "near_global"
            n_to_ng += 1
        else:
            r["category"] = "diffuse_local"
            n_to_dl += 1

    tax["redistribution_note"] = {
        "removed_category": "other",
        "n_other_redistributed": n_other,
        "to_near_global": n_to_ng,
        "to_diffuse_local": n_to_dl,
        "n_unmapped": n_unmapped,
        "threshold_eff_rank_90": K_THRESHOLD,
    }
    tax_path.write_text(json.dumps(tax, indent=2))
    name = tax["model"].split("/")[-1].split(".")[0]
    print(f"  {name:<30} other={n_other:>3}  ->  +NG={n_to_ng:>3}  "
          f"+DL={n_to_dl:>3}  unmapped={n_unmapped}")


def main():
    print(f"Redistributing 'other' (threshold eff_rank_90 = {K_THRESHOLD}):")
    for tax, spec in PAIRS:
        redistribute_one(tax, spec)


if __name__ == "__main__":
    main()
