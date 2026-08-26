"""Matched control channels for the analytic-prior ablation.

The paper claims an *analytic optical* prior shapes the representation at
training time. The claim is only supported if channels that are NOT the
analytic prior fail to reproduce the lift. This module supplies those
controls, each matched to the real prior in tensor shape and (where
stated) in marginal distribution, so the only variable is what the
channels contain.

Controls
--------
zeros
    Two all-zero channels. Isolates the effect of *widening the input*:
    the 5-channel model has two extra first-conv kernels and a different
    initialization draw regardless of channel content. If `zeros`
    reproduces the lift, the prior is irrelevant and the result is an
    architecture artifact.

shuffled
    The real prior channels with their pixels randomly permuted within
    each channel, independently per frame. Marginal distribution is
    preserved *exactly* (a permutation moves no mass); spatial structure
    is destroyed. This is the sharpest control: it separates "the prior's
    values matter" from "the prior's spatial arrangement matters".

random_fixed
    Two fixed smooth random fields, identical for every frame, matched to
    the real prior's per-channel mean and standard deviation. Tests
    whether any fixed structured spatial pattern would do -- i.e. whether
    the network merely benefits from a spatial coordinate-like reference.

phi_dup
    The bare radial fluence map Phi in both channels. Phi is content-free
    (it depends only on pixel position and is identical for every frame).
    This is the control that the mislabelled "Phi-only" arm in the earlier
    manuscript was mistakenly believed to be; running it properly tests
    the content-free-geometry hypothesis on its own terms.

gauss
    Per-frame i.i.d. Gaussian noise matched to the real prior's mean and
    standard deviation. Tests whether stochastic input perturbation alone
    -- a regularizer with no information at all -- accounts for the lift.

Scale matching uses statistics measured over the canonical training split
and stored in `control_prior_stats.json`; regenerate with

    python control_priors.py --compute-stats --data_dir <stage2_data>
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from physics_prior import fluence_map, physics_channels

CONTROL_MODES = ("none", "zeros", "shuffled", "random_fixed", "phi_dup", "gauss")

_STATS_PATH = Path(__file__).resolve().parent / "control_prior_stats.json"
_stats_cache: dict | None = None


def load_stats() -> dict:
    """Per-channel mean/std of the real prior over the training split."""
    global _stats_cache
    if _stats_cache is None:
        if not _STATS_PATH.exists():
            raise FileNotFoundError(
                f"{_STATS_PATH} missing. Run:\n"
                f"  python control_priors.py --compute-stats --data_dir <stage2_data>"
            )
        _stats_cache = json.loads(_STATS_PATH.read_text())
    return _stats_cache


def _smooth_random_field(h: int, w: int, seed: int, cells: int = 8,
                         device=None, dtype=torch.float32) -> torch.Tensor:
    """Low-frequency random field: a cells x cells grid bilinearly upsampled.

    A pure i.i.d. map has no spatial structure and would duplicate `gauss`.
    Upsampling a coarse grid gives a smoothly varying field with structure
    at a scale comparable to the real prior's, which is what we want to
    control for.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    coarse = torch.randn(1, 1, cells, cells, generator=g, dtype=torch.float32)
    field = torch.nn.functional.interpolate(
        coarse, size=(h, w), mode="bilinear", align_corners=False
    )[0, 0]
    field = (field - field.mean()) / (field.std() + 1e-8)
    return field.to(device=device, dtype=dtype)


def control_channels(rgb01: torch.Tensor, mode: str,
                     alpha: float = 4.0,
                     lambda_eff: float | None = None,
                     version: str = "v1",
                     pivot_v2: float = 0.30) -> torch.Tensor:
    """Return [B,2,H,W] control channels matching physics_channels' contract.

    rgb01: [B,3,H,W] in [0,1].
    """
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode {mode!r}; expected {CONTROL_MODES}")
    if mode == "none":
        return physics_channels(rgb01, alpha=alpha, lambda_eff=lambda_eff,
                                version=version, pivot_v2=pivot_v2)

    b, _, h, w = rgb01.shape
    device, dtype = rgb01.device, rgb01.dtype

    if mode == "zeros":
        return torch.zeros(b, 2, h, w, device=device, dtype=dtype)

    if mode == "shuffled":
        real = physics_channels(rgb01, alpha=alpha, lambda_eff=lambda_eff,
                                version=version, pivot_v2=pivot_v2)
        flat = real.reshape(b, 2, h * w)
        # Independent permutation per (sample, channel). argsort of uniform
        # noise gives a uniformly random permutation and stays on-device.
        perm = torch.rand(b, 2, h * w, device=device).argsort(dim=-1)
        return torch.gather(flat, -1, perm).reshape(b, 2, h, w)

    if mode == "phi_dup":
        phi = fluence_map(h, w, lambda_eff=lambda_eff, device=device, dtype=dtype)
        return phi.expand(b, 2, h, w).clone()

    stats = load_stats()
    mean = torch.tensor(stats["mean"], device=device, dtype=dtype).view(1, 2, 1, 1)
    std = torch.tensor(stats["std"], device=device, dtype=dtype).view(1, 2, 1, 1)

    if mode == "random_fixed":
        fields = torch.stack([
            _smooth_random_field(h, w, seed=1234, device=device, dtype=dtype),
            _smooth_random_field(h, w, seed=5678, device=device, dtype=dtype),
        ])                                                   # [2,H,W], unit-normal
        return (fields.unsqueeze(0) * std + mean).expand(b, 2, h, w).clone()

    if mode == "gauss":
        return torch.randn(b, 2, h, w, device=device, dtype=dtype) * std + mean

    raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# stats computation
# --------------------------------------------------------------------------

def _compute_stats(data_dir: str, image_size: int, max_frames: int) -> dict:
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    files = sorted(Path(data_dir, "train").rglob("*.jpg"))
    if not files:
        raise SystemExit(f"no training frames under {data_dir}/train")
    # Deterministic stride so the sample is reproducible and class-spread.
    stride = max(1, len(files) // max_frames)
    sample = files[::stride][:max_frames]

    n = 0
    s = torch.zeros(2, dtype=torch.float64)
    ss = torch.zeros(2, dtype=torch.float64)
    for i, f in enumerate(sample):
        img = tf(Image.open(f).convert("RGB")).unsqueeze(0)
        ch = physics_channels(img)[0].double()               # [2,H,W]
        s += ch.sum(dim=(1, 2))
        ss += (ch ** 2).sum(dim=(1, 2))
        n += ch.shape[1] * ch.shape[2]
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(sample)}]")
    mean = (s / n)
    std = (ss / n - mean ** 2).clamp_min(0).sqrt()
    return {
        "channels": ["P_blood", "H_AFI_weighted"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_frames": len(sample),
        "n_pixels": int(n),
        "image_size": image_size,
        "note": "measured on the canonical training split with physics_prior "
                "v1 defaults (alpha=4.0, lambda_eff=None)",
    }


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--compute-stats", action="store_true")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_frames", type=int, default=2000)
    a = p.parse_args()
    if not a.compute_stats:
        p.error("nothing to do; pass --compute-stats")
    stats = _compute_stats(a.data_dir, a.image_size, a.max_frames)
    _STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"[control_priors] wrote {_STATS_PATH}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
