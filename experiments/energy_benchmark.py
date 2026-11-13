"""Energy per image from NVML power polling during a sustained inference loop."""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import pynvml

sys.path.insert(0, str(Path(__file__).parent))


class PowerMonitor:
    """
    Background thread that samples GPU power.draw via NVML.
    Sample interval ~10 ms (100 Hz). NVML may update every ~30-100 ms
    on Turing, so adjacent samples can repeat; they still enter the
    trapezoid integration for total energy.
    """
    def __init__(self, gpu_index: int, sample_interval_s: float = 0.01):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval = sample_interval_s
        self._stop = threading.Event()
        self._thread = None
        self.samples = []                                 # list[(t, W)]

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except pynvml.NVMLError:
                w = float("nan")
            self.samples.append((t, w))
            time.sleep(self.interval)

    def integrate_energy_J(self):
        """Trapezoid integration of W x dt over collected samples."""
        if len(self.samples) < 2:
            return 0.0, float("nan"), 0.0
        ts = np.array([s[0] for s in self.samples])
        ws = np.array([s[1] for s in self.samples])
        # Drop NaN samples
        valid = ~np.isnan(ws)
        ts = ts[valid]; ws = ws[valid]
        if len(ts) < 2:
            return 0.0, float("nan"), 0.0
        dt = np.diff(ts)
        avg_w = 0.5 * (ws[:-1] + ws[1:])
        energy_J = float((avg_w * dt).sum())
        elapsed = float(ts[-1] - ts[0])
        avg_power_W = energy_J / max(elapsed, 1e-9)
        return energy_J, avg_power_W, elapsed


def calibrate_idle(monitor: PowerMonitor, idle_s: float = 5.0,
                   device: str = "cuda") -> float:
    """Measure average idle GPU power over `idle_s` seconds."""
    torch.cuda.synchronize(device)
    print(f"  Calibrating idle for {idle_s}s ...")
    monitor.start()
    time.sleep(idle_s)
    monitor.stop()
    e, w_avg, dur = monitor.integrate_energy_J()
    print(f"    idle: avg_W={w_avg:.2f}  (over {dur:.2f}s)")
    return w_avg


def time_condition(model, x, monitor: PowerMonitor, idle_W: float,
                   warmup_s: float, run_s: float,
                   device: str = "cuda",
                   tag: str = "?") -> dict:
    """Warmup the model, then run a fixed-duration loop with power
    sampling and forward counting. Returns dict of J/forward, etc."""
    print(f"  [{tag}] warmup for {warmup_s}s ...")
    t_end = time.perf_counter() + warmup_s
    n_warm = 0
    with torch.no_grad():
        while time.perf_counter() < t_end:
            _ = model(x); n_warm += 1
        torch.cuda.synchronize(device)
    print(f"    warmup forwards: {n_warm}")

    print(f"  [{tag}] timed run for {run_s}s ...")
    monitor.start()
    t_start = time.perf_counter()
    n_fwd = 0
    with torch.no_grad():
        while time.perf_counter() - t_start < run_s:
            _ = model(x); n_fwd += 1
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_start
    monitor.stop()

    energy_J, avg_W, sample_dur = monitor.integrate_energy_J()
    compute_W = avg_W - idle_W
    compute_energy_J = compute_W * elapsed
    j_per_forward = compute_energy_J / max(n_fwd, 1)
    j_per_image = j_per_forward / x.shape[0]
    ms_per_forward = 1000.0 * elapsed / max(n_fwd, 1)

    print(f"    {tag}: forwards={n_fwd}  elapsed={elapsed:.2f}s  "
          f"avg_W={avg_W:.2f}  compute_W={compute_W:.2f}  "
          f"J/fwd={j_per_forward:.4f}  J/img={j_per_image:.5f}  "
          f"ms/fwd={ms_per_forward:.2f}")

    return {
        "tag": tag,
        "n_forwards": n_fwd,
        "elapsed_s": elapsed,
        "sample_dur_s": sample_dur,
        "avg_W_total": avg_W,
        "avg_W_compute": compute_W,
        "compute_energy_J": compute_energy_J,
        "j_per_forward": j_per_forward,
        "j_per_image": j_per_image,
        "ms_per_forward": ms_per_forward,
        "batch_size": x.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--broadcast-json", default=None)
    ap.add_argument("--img-size", type=int, default=None,
                    help="image side; default = model's native input")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--run-s", type=float, default=30.0)
    ap.add_argument("--gpu-index", type=int, default=1,
                    help="absolute GPU index for NVML (matches the index "
                         "in nvidia-smi, NOT the CUDA_VISIBLE_DEVICES "
                         "remap; pass the same index here as you did "
                         "in CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--dtype", default="fp16",
                    choices=("fp32", "fp16"))
    ap.add_argument("--include-full", action="store_true",
                    help="also benchmark Triton-FULL (expected very slow)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import timm
    from shift_substitution import detect_model_config
    from kernel_substitution import kernel_from_diagnostics_mean
    from fused_substitutions_triton import (
        install_triton_fused_substitutions,
        restore_forwards as restore_triton,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA required.")

    if args.img_size is None:
        tmp = timm.create_model(args.model, pretrained=False)
        args.img_size = tmp.default_cfg.get("input_size", (3, 224, 224))[-1]
        del tmp
    print(f"Loading {args.model} at img_size={args.img_size} ...")

    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    if args.dtype == "fp16":
        model = model.half()
    cfg = detect_model_config(model, image_size=args.img_size)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size,
                      device=device, dtype=torch_dtype)

    tax = json.loads(Path(args.taxonomy).read_text())
    res_by_lh = {(r["layer"], r["head"]): r for r in tax["results"]}
    shift_cats = set(args.shift_cats.split(","))
    shift_heads = [(r["layer"], r["head"]) for r in tax["results"]
                   if r["category"] in shift_cats]
    kernel_map = {
        (li, hi): kernel_from_diagnostics_mean(
            np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
        for (li, hi) in shift_heads
    }
    broadcast_heads = []
    if args.broadcast_json is not None:
        bc = json.loads(Path(args.broadcast_json).read_text())
        broadcast_heads = [tuple(h) for h in bc.get("low_r_heads", [])]
    print(f"Shifts: n={len(shift_heads)}  Broadcasts: n={len(broadcast_heads)}")

    monitor = PowerMonitor(args.gpu_index)
    idle_W = calibrate_idle(monitor, args.idle_s, device)

    print(f"\nbatch={args.batch_size}, dtype={args.dtype}, "
          f"size={args.img_size}, run={args.run_s}s")

    out = {
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "gpu_index": args.gpu_index,
        "idle_W": idle_W,
        "warmup_s": args.warmup_s,
        "run_s": args.run_s,
        "n_shifts": len(shift_heads),
        "n_broadcasts": len(broadcast_heads),
        "conditions": {},
    }

    # baseline
    res = time_condition(model, x, monitor, idle_W,
                          args.warmup_s, args.run_s, device,
                          tag="baseline")
    out["conditions"]["baseline"] = res

    # Triton S+B
    if shift_heads or broadcast_heads:
        saved = install_triton_fused_substitutions(
            model, kernel_map=kernel_map,
            broadcast_heads=(broadcast_heads or None),
            grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"])
        res = time_condition(model, x, monitor, idle_W,
                              args.warmup_s, args.run_s, device,
                              tag="triton-S+B")
        out["conditions"]["triton_sb"] = res
        restore_triton(model, saved)

    # Triton FULL
    if args.include_full:
        saved = install_triton_fused_substitutions(
            model, kernel_map=kernel_map,
            broadcast_heads=(broadcast_heads or None),
            grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"],
            use_triton_for_sdpa=True)
        res = time_condition(model, x, monitor, idle_W,
                              args.warmup_s, args.run_s, device,
                              tag="triton-FULL")
        out["conditions"]["triton_full"] = res
        restore_triton(model, saved)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Summary
    print("\nENERGY SUMMARY")
    base = out["conditions"]["baseline"]
    print(f"{'condition':<14}  {'avg_W_tot':>9}  {'compute_W':>9}  "
          f"{'ms/fwd':>7}  {'J/fwd':>7}  {'J/img':>9}  "
          f"{'rel-J/img':>9}  {'rel-ms':>7}")
    for cond, r in out["conditions"].items():
        rel_e = r["j_per_image"] / max(base["j_per_image"], 1e-12)
        rel_t = r["ms_per_forward"] / max(base["ms_per_forward"], 1e-12)
        print(f"{cond:<14}  {r['avg_W_total']:>8.2f}W  "
              f"{r['avg_W_compute']:>8.2f}W  "
              f"{r['ms_per_forward']:>6.2f}  {r['j_per_forward']:>6.3f}  "
              f"{r['j_per_image']:>8.5f}  {rel_e:>8.4f}x  {rel_t:>6.4f}x")


if __name__ == "__main__":
    main()
