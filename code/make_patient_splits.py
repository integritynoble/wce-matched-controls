"""Generate repeated patient-disjoint splits of Kvasir-Capsule.

Why this exists: the 616-run control matrix uses 44 seeds on ONE patient
partition, so it estimates uncertainty from training stochasticity and not from
which patients happen to land in train/test. Those are different estimands, and
a reviewer is entitled to ask why the treatment effect is inferred from seeds
rather than patients. This script produces additional partitions so the second
source of variance can be measured.

Design choices, all forced by the data:

* 43 videos, one per patient. The canonical split is 30 train / 6 val / 7 test
  videos (31,820 / 8,986 / 6,423 frames) and is patient-disjoint. New splits
  keep those video counts so frame counts stay comparable.
* Several classes are extremely rare, so a random partition can leave a class
  with zero test positives and silently change the macro's denominator -- the
  same denominator problem that distorted the Galar comparison. Each candidate
  partition is therefore required to place at least MIN_TEST_POS positives of
  every reportable class in test, and at least one in train, or it is rejected
  and redrawn.
* Selection is deterministic given --seed, and every accepted partition is
  written with the video lists and per-class counts so it can be audited.

USAGE
    python make_patient_splits.py --n_splits 10 --out_root ~/biohpc/tmi_splits
    python make_patient_splits.py --n_splits 10 --dry_run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path

_DEFAULT_SOURCE = "/home/S248103/biohpc/tmi_data/stage2_data"
SPLIT_VIDEOS = {"train": 30, "val": 6, "test": 7}   # the canonical shape
# The classes any split must be able to score, so macro metrics share a
# denominator across splits. Matches the pinned cross-dataset set.
REPORTABLE = [
    "Angiectasia", "Blood - fresh", "Erosion", "Erythema", "Foreign Body",
    "Ileocecal valve", "Lymphangiectasia", "Normal clean mucosa", "Pylorus",
    "Ulcer",
]
MIN_TEST_POS = 5      # per reportable class, in test
MIN_TRAIN_POS = 5     # per reportable class, in train


def index_frames(SOURCE: Path) -> dict[str, list[tuple[Path, str]]]:
    """video_id -> [(path, class), ...] pooled across the canonical splits."""
    by_video: dict[str, list[tuple[Path, str]]] = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for cls_dir in sorted((SOURCE / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.glob("*.jpg"):
                by_video[f.name.split("_")[0]].append((f, cls_dir.name))
    return by_video


def class_counts(videos, by_video) -> collections.Counter:
    c = collections.Counter()
    for v in videos:
        for _f, cls in by_video[v]:
            c[cls] += 1
    return c


def acceptable(parts, by_video) -> tuple[bool, str]:
    tr = class_counts(parts["train"], by_video)
    te = class_counts(parts["test"], by_video)
    for cls in REPORTABLE:
        if te[cls] < MIN_TEST_POS:
            return False, f"test has {te[cls]} {cls!r} (< {MIN_TEST_POS})"
        if tr[cls] < MIN_TRAIN_POS:
            return False, f"train has {tr[cls]} {cls!r} (< {MIN_TRAIN_POS})"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--out_root", type=Path,
                    default=Path("/home/S248103/biohpc/tmi_splits"))
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--max_tries", type=int, default=20000)
    ap.add_argument("--source", type=Path, default=Path(_DEFAULT_SOURCE),
                    help="canonical staged pool: <source>/{train,val,test}/<class>/*.jpg")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    SOURCE = a.source
    by_video = index_frames(SOURCE)
    all_classes = sorted({cls for v in by_video.values() for _f, cls in v})
    videos = sorted(by_video)
    print(f"indexed {sum(len(v) for v in by_video.values())} frames "
          f"across {len(videos)} videos")
    if len(videos) != sum(SPLIT_VIDEOS.values()):
        print(f"  WARNING: expected {sum(SPLIT_VIDEOS.values())} videos, got {len(videos)}")

    rng = random.Random(a.seed)
    accepted, tries, rejects = [], 0, collections.Counter()
    while len(accepted) < a.n_splits and tries < a.max_tries:
        tries += 1
        shuf = videos[:]
        rng.shuffle(shuf)
        i, j = SPLIT_VIDEOS["train"], SPLIT_VIDEOS["train"] + SPLIT_VIDEOS["val"]
        parts = {"train": shuf[:i], "val": shuf[i:j], "test": shuf[j:]}
        ok, why = acceptable(parts, by_video)
        if not ok:
            rejects[why.split(" (")[0]] += 1
            continue
        if any(set(parts["test"]) == set(p["test"]) for p in accepted):
            continue                     # no duplicate test sets
        accepted.append(parts)

    print(f"accepted {len(accepted)}/{a.n_splits} partitions after {tries} draws")
    if rejects:
        print("  most common rejection reasons:")
        for why, n in rejects.most_common(4):
            print(f"    {n:6d}  {why}")
    if len(accepted) < a.n_splits:
        print("  ERROR: could not find enough partitions; relax MIN_TEST_POS")
        return 1

    for k, parts in enumerate(accepted):
        counts = {s: class_counts(parts[s], by_video) for s in parts}
        total = {s: sum(counts[s].values()) for s in parts}
        print(f"\nsplit {k:02d}  frames train/val/test = "
              f"{total['train']}/{total['val']}/{total['test']}")
        if a.dry_run:
            continue
        root = a.out_root / f"split{k:02d}"
        # Every class directory must exist in every split, empty ones included.
        # ImageFolder derives class_to_idx from the directories present, so a
        # split whose train/ lacks a rare class trains a 12- or 13-class head --
        # not comparable to the 616-run matrix or to the other splits -- and a
        # val/ that disagrees with train/ trips the trainer's class_to_idx
        # assertion outright. The canonical staging carries all 14 directories in
        # all three splits for exactly this reason; omitting them silently
        # produced 85 failed runs across 6 of 10 splits.
        for s in ("train", "val", "test"):
            for cls in all_classes:
                (root / s / cls).mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test"):
            for v in parts[s]:
                for f, cls in by_video[v]:
                    dst = root / s / cls / f.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(f, dst)
                    except FileExistsError:
                        pass          # idempotent: safe to re-run over a partial tree
        (root / "split_manifest.json").write_text(json.dumps({
            "split_index": k,
            "generator_seed": a.seed,
            "source": str(SOURCE),
            "videos": parts,
            "frames": total,
            "per_class_counts": {s: dict(counts[s]) for s in parts},
            "reportable_classes": REPORTABLE,
            "min_test_positives": MIN_TEST_POS,
            "patient_disjoint": True,
        }, indent=2))
    if not a.dry_run:
        print(f"\nstaged {len(accepted)} splits under {a.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
