from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from medseg.data.dataset_offline import load_pt_paths  # type: ignore
from medseg.models.build_model import build_model  # type: ignore
from medseg.utils.ckpt import load_ckpt  # type: ignore
from monai.inferers.utils import sliding_window_inference  # type: ignore
from twostage_medseg.twostage.roi_utils import compute_bbox_from_mask, crop_3d, paste_3d, bbox_to_dict  # type: ignore


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medseg_root", type=str, required=True)
    p.add_argument("--preprocessed_root", type=str, required=True)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--stage1_model", type=str, default="dynunet")
    p.add_argument("--stage2_model", type=str, default="dynunet")
    p.add_argument("--stage1_patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--stage2_patch", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--stage1_sw_batch_size", type=int, default=1)
    p.add_argument("--stage2_sw_batch_size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--margin", type=int, default=12)
    p.add_argument("--n", type=int, default=0, help="只跑前N个case, 0=全部")
    p.add_argument("--save_dir", type=str, default="./twostage_outputs")
    return p.parse_args()


def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.bool()
    gt = gt.bool()
    inter = (pred & gt).sum().double()
    denom = pred.sum().double() + gt.sum().double()
    return float((2.0 * inter + eps) / (denom + eps))


def main():
    args = parse_args()
    add_medseg_to_syspath(args.medseg_root)

    os.makedirs(args.save_dir, exist_ok=True)

    pt_paths = load_pt_paths(args.preprocessed_root)
    if args.n > 0:
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage1 = build_model(
        args.stage1_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage1_patch),
    ).to(device)
    stage2 = build_model(
        args.stage2_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage2_patch),
    ).to(device)
    load_ckpt(args.stage1_ckpt, stage1, optimizer=None, map_location=device)
    load_ckpt(args.stage2_ckpt, stage2, optimizer=None, map_location=device)
    stage1.eval()
    stage2.eval()

    rows: List[Dict] = []

    with torch.no_grad():
        for i, pt_path in enumerate(pt_paths, start=1):
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )
            image = data["image"].float()  # [1,D,H,W]
            label = data.get("label", None)  # [1,D,H,W] maybe optional

            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                logits1 = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(args.stage1_patch),
                    sw_batch_size=args.stage1_sw_batch_size,
                    predictor=stage1,
                    overlap=args.overlap,
                )
            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]
            logits1 = torch.as_tensor(logits1)
            pred1 = torch.argmax(logits1.float(), dim=1)[0].cpu()  # [D,H,W], 0/1
            liver_mask = pred1 == 1

            if liver_mask.sum().item() == 0:
                tumor_full = torch.zeros_like(pred1)
                bbox_info = None
            else:
                bbox = compute_bbox_from_mask(liver_mask, margin=args.margin)
                image_roi = crop_3d(image, bbox)  # [1,d,h,w]
                x_roi = image_roi.unsqueeze(0).to(device)  # [1,1,d,h,w]
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
                ):
                    logits2 = sliding_window_inference(
                        inputs=x_roi,
                        roi_size=tuple(args.stage2_patch),
                        sw_batch_size=args.stage2_sw_batch_size,
                        predictor=stage2,
                        overlap=args.overlap,
                    )
                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]
                logits2 = torch.as_tensor(logits2)
                pred2 = torch.argmax(logits2.float(), dim=1)[0].cpu()  # [d,h,w], 0/1
                tumor_full = torch.zeros_like(pred1)
                tumor_full = paste_3d(tumor_full, pred2, bbox)
                bbox_info = bbox_to_dict(bbox)

            row = {
                "case_name": Path(pt_path).stem,
                "source_pt": pt_path,
                "pred_liver_voxels": int(liver_mask.sum().item()),
                "pred_tumor_voxels": int((tumor_full == 1).sum().item()),
                "bbox": bbox_info,
            }

            if label is not None:
                gt_liver = label[0] > 0
                gt_tumor = label[0] == 2
                row["liver_dice_stage1"] = dice_binary(liver_mask, gt_liver)
                row["tumor_dice_stage2"] = dice_binary(tumor_full == 1, gt_tumor)

            torch.save(
                {
                    "image": image,
                    "label": label,
                    "stage1_liver_pred": pred1.unsqueeze(0).long(),
                    "stage2_tumor_pred": tumor_full.unsqueeze(0).long(),
                    "meta": row,
                },
                os.path.join(args.save_dir, f"{Path(pt_path).stem}_twostage.pt"),
            )
            rows.append(row)
            print(
                f"[{i}/{len(pt_paths)}] {row['case_name']} tumor_vox={row['pred_tumor_voxels']}"
            )

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    tumor_dices = [r["tumor_dice_stage2"] for r in rows if "tumor_dice_stage2" in r]
    liver_dices = [r["liver_dice_stage1"] for r in rows if "liver_dice_stage1" in r]
    if tumor_dices:
        print(f"mean tumor dice = {sum(tumor_dices) / len(tumor_dices):.4f}")
    if liver_dices:
        print(f"mean liver dice = {sum(liver_dices) / len(liver_dices):.4f}")


if __name__ == "__main__":
    main()
