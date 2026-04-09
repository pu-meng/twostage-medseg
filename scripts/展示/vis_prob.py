"""
Visualize prob_pt for error analysis.
Run standalone: python vis_prob.py
Or import vis_worst_cases() for use in eval_twostage.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─────────────────────────────────────────────
# slice selection helpers
# ─────────────────────────────────────────────

def _best_prob_slice(prob_np):
    """Slice with highest tumor probability."""
    return int(np.argmax(prob_np.max(axis=1).max(axis=1)))


def _worst_miss_slice(label_np, prob_np):
    """Slice where GT tumor voxels have lowest average prob (worst FN)."""
    gt_tumor = label_np == 2
    if gt_tumor.sum() == 0:
        return _best_prob_slice(prob_np)
    scores = np.full(label_np.shape[0], 1.0)
    for z in range(label_np.shape[0]):
        m = gt_tumor[z]
        if m.sum() > 0:
            scores[z] = prob_np[z][m].mean()
    return int(np.argmin(scores))


def _worst_fp_slice(label_np, pred_np):
    """Slice with most FP voxels (pred==2 but GT!=2)."""
    fp = (pred_np == 2) & (label_np != 2)
    counts = fp.sum(axis=(1, 2))
    if counts.max() == 0:
        return None
    return int(np.argmax(counts))


# ─────────────────────────────────────────────
# drawing
# ─────────────────────────────────────────────

def _draw_row(axes, image, label, pred, prob_np, z, row_label):
    gt_tumor   = label[z] == 2
    pred_tumor = pred[z] == 2
    tp = gt_tumor & pred_tumor
    fp = pred_tumor & ~gt_tumor
    fn = gt_tumor & ~pred_tumor

    # col0: CT
    axes[0].imshow(image[z], cmap="gray", vmin=0, vmax=1)
    axes[0].set_ylabel(row_label, fontsize=9, rotation=0, labelpad=65, va="center")

    # col1: GT overlay
    axes[1].imshow(image[z], cmap="gray", vmin=0, vmax=1)
    ov = np.zeros((*label[z].shape, 4))
    ov[label[z] == 1] = [0.2, 0.6, 1.0, 0.35]   # liver = blue
    ov[label[z] == 2] = [1.0, 0.2, 0.2, 0.60]   # tumor = red
    axes[1].imshow(ov)

    # col2: Pred overlay  (TP=green FP=orange liver=blue)
    axes[2].imshow(image[z], cmap="gray", vmin=0, vmax=1)
    ov2 = np.zeros((*pred[z].shape, 4))
    ov2[pred[z] == 1] = [0.2, 0.6, 1.0, 0.35]
    ov2[tp] = [0.0, 0.9, 0.0, 0.60]
    ov2[fp] = [1.0, 0.6, 0.0, 0.70]
    axes[2].imshow(ov2)

    # col3: Miss map  (FN=yellow TP=green)
    axes[3].imshow(image[z], cmap="gray", vmin=0, vmax=1)
    ov3 = np.zeros((*label[z].shape, 4))
    ov3[tp] = [0.0, 0.9, 0.0, 0.50]
    ov3[fn] = [1.0, 1.0, 0.0, 0.80]
    axes[3].imshow(ov3)

    # col4: prob heatmap
    im = axes[4].imshow(prob_np[z], cmap="hot", vmin=0, vmax=1)

    for ax in axes:
        ax.axis("off")
    return im


def _vis_one(image, label, prob_np, meta, save_path: Path):
    """
    Draw a 3-row x 5-col figure for one case and save it.
    image, label: numpy [D,H,W]
    prob_np: numpy [D,H,W]  (stage2 tumor probability)
    meta: dict with tumor_dice, tumor_recall, tumor_FDR, etc.
    """
    # derive hard pred from prob
    pred = (prob_np > 0.5).astype(np.int64)
    # restore liver label from GT (we only care about tumor errors)
    pred[label == 1] = np.where(pred[label == 1] == 0, 1, pred[label == 1])

    z_prob = _best_prob_slice(prob_np)
    z_fn   = _worst_miss_slice(label, prob_np)
    z_fp   = _worst_fp_slice(label, pred)

    rows = [(z_prob, f"Best Prob\nz={z_prob}"),
            (z_fn,   f"Worst Miss\nz={z_fn}")]
    if z_fp is not None:
        rows.append((z_fp, f"Worst FP\nz={z_fp}"))

    n_rows = len(rows)
    fig, axes_all = plt.subplots(n_rows, 5, figsize=(24, 5 * n_rows))
    if n_rows == 1:
        axes_all = axes_all[np.newaxis, :]

    for row_i, (zi, rlabel) in enumerate(rows):
        im = _draw_row(axes_all[row_i], image, label, pred, prob_np, zi, rlabel)

    col_titles = [
        "CT Image",
        "GT  (liver=blue  tumor=red)",
        "Pred  (TP=green  FP=orange)",
        "Miss  (FN=yellow  TP=green)",
        "Tumor Prob",
    ]
    for col_i, t in enumerate(col_titles):
        axes_all[0][col_i].set_title(t, fontsize=10)

    plt.colorbar(im, ax=axes_all[:, 4].tolist(), fraction=0.02, pad=0.01, label="Prob")

    case_name = meta.get("case_name", save_path.stem)
    fig.suptitle(
        f"{case_name}  |  Tumor Dice={meta.get('tumor_dice', float('nan')):.4f}"
        f"  Recall={meta.get('tumor_recall', float('nan')):.4f}"
        f"  FDR={meta.get('tumor_FDR', float('nan')):.4f}"
        f"  gt_tumor={meta.get('gt_tumor_voxels', '?')}vox ({meta.get('tumor_size_cat', '?')})",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis_prob] saved: {save_path}")


# ─────────────────────────────────────────────
# public API (called from eval_twostage.py)
# ─────────────────────────────────────────────

def vis_worst_cases(workdir: str, rows: list, n_worst: int = 3, preprocessed_root: str = ""):
    """
    Called at the end of eval to visualize the n_worst cases by tumor_dice.

    workdir           : eval output dir (contains prob_pt/)
    rows              : list of per-case metric dicts (same as written to per_case.csv)
    n_worst           : how many worst cases to visualize
    preprocessed_root : root dir of source .pt files (to resolve basename-only source_pt)
    """
    import os

    prob_dir = Path(workdir) / "prob_pt"
    vis_dir  = Path(workdir) / "vis_prob"

    valid = [r for r in rows if (prob_dir / f"{r['case_name']}_prob.pt").exists()]
    valid.sort(key=lambda r: float(r.get("tumor_dice", 1.0)))
    targets = valid[:n_worst]

    if not targets:
        print("[vis_prob] no cases to visualize")
        return

    print(f"[vis_prob] visualizing worst {len(targets)} cases ...")
    for meta in targets:
        case_name = meta["case_name"]
        prob_path = prob_dir / f"{case_name}_prob.pt"
        source_pt = meta.get("source_pt", "")
        # resolve basename to full path if needed
        if preprocessed_root and not os.path.isabs(source_pt):
            source_pt = os.path.join(preprocessed_root, source_pt)

        try:
            prob_np = torch.load(prob_path, map_location="cpu", weights_only=False).numpy()
            data    = torch.load(source_pt, map_location="cpu", weights_only=False, mmap=True)
            image   = data["image"][0].float().numpy()
            label   = data["label"][0].numpy()

            dice = float(meta.get("tumor_dice", 0))
            save_path = vis_dir / f"{case_name}_dice{dice:.3f}.png"
            _vis_one(image, label, prob_np, meta, save_path)
        except Exception as e:
            print(f"  [vis_prob] skip {case_name}: {e}")


# ─────────────────────────────────────────────
# standalone usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ===== edit parameters here =====
    EVAL_DIR           = "/home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/eval/04-06-12-56-02"
    PREPROCESSED_ROOT  = "/home/pumengyu/Task03_Liver_pt"   # root dir of source .pt files
    CASE               = None    # single case e.g. "liver_121", or None for all
    DICE_THRESHOLD     = 0.7     # only process cases with tumor_dice < this; None = all
    N_WORST            = None    # if set, only process the N worst cases (overrides DICE_THRESHOLD)
    # ================================

    import csv, os
    eval_dir = Path(EVAL_DIR)
    prob_dir = eval_dir / "prob_pt"
    vis_dir  = eval_dir / "vis_prob"

    per_case_path = eval_dir / "per_case.csv"
    meta_map = {}
    with open(per_case_path) as f:
        for row in csv.DictReader(f):
            meta_map[row["case_name"]] = row

    all_cases = sorted(p.stem.replace("_prob", "") for p in prob_dir.glob("*_prob.pt"))

    if CASE is not None:
        cases = [CASE]
    elif N_WORST is not None:
        cases = sorted(all_cases, key=lambda c: float(meta_map.get(c, {}).get("tumor_dice", 1.0)))[:N_WORST]
    elif DICE_THRESHOLD is not None:
        cases = [c for c in all_cases if float(meta_map.get(c, {}).get("tumor_dice", 1.0)) < DICE_THRESHOLD]
    else:
        cases = all_cases

    print(f"Processing {len(cases)} cases ...")

    for case in cases:
        meta = dict(meta_map.get(case, {"case_name": case}))
        meta["case_name"] = case

        prob_path = prob_dir / f"{case}_prob.pt"
        source_pt = meta.get("source_pt", "")
        if PREPROCESSED_ROOT and not os.path.isabs(source_pt):
            source_pt = os.path.join(PREPROCESSED_ROOT, source_pt)

        try:
            prob_np = torch.load(prob_path, map_location="cpu", weights_only=False).numpy()
            data    = torch.load(source_pt, map_location="cpu", weights_only=False, mmap=True)
            image   = data["image"][0].float().numpy()
            label   = data["label"][0].numpy()

            # csv values are strings — convert numeric fields for _vis_one
            num_fields = ["tumor_dice", "tumor_recall", "tumor_FDR", "tumor_precision",
                          "liver_dice", "gt_tumor_voxels", "gt_liver_voxels"]
            for f in num_fields:
                if f in meta:
                    try:
                        meta[f] = float(meta[f])
                    except (ValueError, TypeError):
                        pass

            dice = float(meta.get("tumor_dice", 0))
            save_path = vis_dir / f"{case}_dice{dice:.3f}.png"
            _vis_one(image, label, prob_np, meta, save_path)
        except Exception as e:
            print(f"  skip {case}: {e}")
