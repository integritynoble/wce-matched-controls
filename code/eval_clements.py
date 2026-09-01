"""Zero-shot evaluation of Kvasir-trained checkpoints on the Clements cohort.

Third external cohort: 43 de-identified PillCam studies from Clements
University Hospital, read at study level by a gastroenterologist. Analysis plan
fixed in advance in paper/Capsule-Endoscopy/CLEMENTS_PREREGISTRATION_2026-08-31.md;
this script implements that document and nothing beyond it.

No model is retrained and no model has seen a Clements frame. Frames were
extracted from the .gvf exports with given_rapid_reader.py at
--stride 10 --quality-filter --dedup, a setting fixed before any evaluation.

This script writes PER-FRAME probabilities only. Aggregation to study level and
every contrast lives in aggregate_clements.py, so that the expensive GPU pass
never has to be repeated if an analysis choice is questioned -- and so that the
aggregation cannot be quietly tuned while looking at the GPU output.

USAGE
    python eval_clements.py --frames_root /home/S248103/biohpc/clements_frames \
        --scan_root /home/S248103/biohpc/tmi_runs
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "Capsule-Endoscopy"))
sys.path.insert(0, str(_CODE_DIR / "gastroscopy_code_package"))
sys.path.insert(0, str(_CODE_DIR / "galar"))

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]


class ClementsFrames(Dataset):
    """Every extracted frame, flat, tagged with its study GUID.

    Studies are directories named by GUID; no patient identifier appears in any
    path, and none is carried into the saved arrays.
    """

    def __init__(self, root: Path, transform):
        self.items = []
        for study in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(study.glob("*.jpg")):
                self.items.append((str(f), study.name))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, study = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), study, path


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch], [b[2] for b in batch]


def evaluate(ckpt_dir: Path, frames_root: Path, device: str,
             batch_size: int, num_workers: int, rerun: bool) -> dict | None:
    out = ckpt_dir / "clements_predictions.npz"
    if out.exists() and not rerun:
        return {"status": "cached", "dir": ckpt_dir.name}
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        return None

    from eval_zero_shot import _build_model_and_transform  # noqa: WPS433

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    if ckpt["class_names"] != ALL_CLASSES:
        raise RuntimeError(f"unexpected class set in {ckpt_path}")

    # Reuses the Galar builder deliberately. It carries control_prior through
    # from the checkpoint; a control arm evaluated with the TRUE analytic
    # channels sees an input distribution it never trained on and returns a
    # plausible wrong number rather than raising. That bug would invalidate
    # every control arm here exactly as it would have on Galar.
    model, tf_eval, arm = _build_model_and_transform(args, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = ClementsFrames(frames_root, tf_eval)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=_collate,
                    pin_memory=str(device).startswith("cuda"))

    probs, studies, paths = [], [], []
    with torch.no_grad():
        for xs, sids, ps in dl:
            xs = xs.to(device, non_blocking=True)
            logits = model(xs)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            studies.extend(sids)
            paths.extend(ps)

    P = np.concatenate(probs, axis=0).astype(np.float32)
    np.savez_compressed(out, probs=P, studies=np.array(studies),
                        paths=np.array(paths),
                        class_names=np.array(ALL_CLASSES))
    return {"status": "ok", "dir": ckpt_dir.name, "arm": arm,
            "n_frames": int(P.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", type=Path, required=True)
    ap.add_argument("--scan_root", type=Path, action="append", required=True,
                    help="Directory of run dirs to scan; repeatable.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on run dir name")
    a = ap.parse_args()

    dirs = []
    for root in a.scan_root:
        if root.exists():
            dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    dirs = [d for d in dirs if a.only in d.name]
    print(f"[clements] {len(dirs)} candidate run dirs, device={a.device}")

    ok = cached = skipped = failed = 0
    for d in dirs:
        try:
            r = evaluate(d, a.frames_root, a.device, a.batch_size,
                         a.num_workers, a.rerun)
        except Exception:
            traceback.print_exc()
            failed += 1
            continue
        if r is None:
            skipped += 1
        elif r["status"] == "cached":
            cached += 1
        else:
            ok += 1
            print(f"  {r['dir']:52} {r['arm']:22} {r['n_frames']} frames")
    print(f"[clements] done: {ok} evaluated, {cached} cached, "
          f"{skipped} skipped (no checkpoint), {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
