"""
统计所有131个案例的肿瘤情况,按 split_two_with_monitor 划分,
分为极小/小/中等/大四个类别,输出到终端和 txt 文件。

体素阈值（基于 LiTS 经验）：
  极小: tumor_voxels < 5000
  小:   5000 <= tumor_voxels < 50000
  中等: 50000 <= tumor_voxels < 300000
  大:   tumor_voxels >= 300000
  无肿瘤: tumor_voxels == 0
"""

import os
import sys
import torch

sys.path.insert(0, "/home/pumengyu/medseg_project")
from medseg.data.dataset_offline import split_two_with_monitor

PT_DIR = "/home/pumengyu/Task03_Liver_pt"
OUT_TXT = "/home/pumengyu/twostage_medseg/scripts/split_two_with_mintor.txt"

# 阈值
TINY_MAX = 5_000
SMALL_MAX = 50_000
MED_MAX = 300_000


def classify(vox):
    if vox == 0:
        return "无肿瘤"
    if vox < TINY_MAX:
        return "极小"
    if vox < SMALL_MAX:
        return "小"
    if vox < MED_MAX:
        return "中等"
    return "大"


def get_stats(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    seg = data["label"][0]  # [D,H,W]
    tumor_vox = int((seg == 2).sum().item())
    liver_vox = int((seg == 1).sum().item())
    return liver_vox, tumor_vox


def process_split(paths, split_name):
    rows = []
    for p in sorted(paths):
        name = os.path.basename(p).replace(".pt", "")
        liver_vox, tumor_vox = get_stats(p)
        cat = classify(tumor_vox)
        rows.append((name, liver_vox, tumor_vox, cat))
        print(f"  {name}: liver={liver_vox:>10,}  tumor={tumor_vox:>10,}  [{cat}]")
    return rows


def summary_block(rows, split_name, monitor_names=None):
    cats = ["无肿瘤", "极小", "小", "中等", "大"]
    lines = []
    lines.append("=" * 70)
    lines.append(f"[{split_name}]  共 {len(rows)} 个案例")
    lines.append("=" * 70)
    for cat in cats:
        subset = [r for r in rows if r[3] == cat]
        if not subset:
            continue
        lines.append(f"\n  --- {cat} (n={len(subset)}) ---")
        if cat == "无肿瘤":
            for name, lv, tv, _ in sorted(subset):
                marker = " *" if monitor_names and name in monitor_names else ""
                lines.append(f"  {name:<12} liver={lv:>10,}{marker}")
        else:
            tumor_voxs = [r[2] for r in subset]
            lines.append(
                f"  tumor体素范围: {min(tumor_voxs):,} ~ {max(tumor_voxs):,}  mean={int(sum(tumor_voxs) / len(tumor_voxs)):,}"
            )
            for name, lv, tv, _ in sorted(subset, key=lambda r: r[2]):
                marker = " *" if monitor_names and name in monitor_names else ""
                lines.append(f"  {name:<12} liver={lv:>10,}  tumor={tv:>10,}{marker}")
    if monitor_names:
        lines.append("\n  (* 标记为监控子集,同时参与训练)")
    lines.append("")
    return "\n".join(lines)


def summary_block_simple(rows, split_name):
    """用于monitor和test,不需要marker"""
    return summary_block(rows, split_name, monitor_names=None)


def count_summary(all_splits, split_names):
    cats = ["无肿瘤", "极小", "小", "中等", "大"]
    lines = []
    lines.append("=" * 70)
    lines.append("汇总统计")
    lines.append("=" * 70)
    header = f"  {'类别':<8}" + "".join(f"  {n:<10}" for n in split_names + ["合计"])
    lines.append(header)
    lines.append("  " + "-" * 58)
    for cat in cats:
        counts = [sum(1 for r in rows if r[3] == cat) for rows in all_splits]
        total = sum(counts)
        lines.append(
            f"  {cat:<8}" + "".join(f"  {c:<10}" for c in counts) + f"  {total:<10}"
        )
    lines.append("  " + "-" * 58)
    grand = [len(rows) for rows in all_splits]
    lines.append(
        f"  {'合计':<8}" + "".join(f"  {c:<10}" for c in grand) + f"  {sum(grand):<10}"
    )
    lines.append("")
    lines.append("阈值说明:")
    lines.append(f"  极小: tumor < {TINY_MAX:,} 体素")
    lines.append(f"  小:   {TINY_MAX:,} <= tumor < {SMALL_MAX:,} 体素")
    lines.append(f"  中等: {SMALL_MAX:,} <= tumor < {MED_MAX:,} 体素")
    lines.append(f"  大:   tumor >= {MED_MAX:,} 体素")
    lines.append("")
    lines.append("划分说明:")
    lines.append("  train:   112个全部参与训练（包含monitor子集）")
    lines.append(
        "  monitor: 12个,train的子集,覆盖各类别,用于训练中监控dice/选best ckpt"
    )
    lines.append("  test:    19个,nnUNet fold0验证集,用于最终和nnUNet对比")
    return "\n".join(lines)


if __name__ == "__main__":
    all_pt = [os.path.join(PT_DIR, f) for f in os.listdir(PT_DIR) if f.endswith(".pt")]
    tr, monitor, te = split_two_with_monitor(all_pt)

    monitor_names = {os.path.basename(p).replace(".pt", "") for p in monitor}

    print(f"\n=== 划分结果: train={len(tr)} monitor={len(monitor)} test={len(te)} ===")

    print("\n=== 处理 train (112个) ===")
    tr_rows = process_split(tr, "train")
    print("\n=== 处理 monitor (12个,train子集) ===")
    mo_rows = process_split(monitor, "monitor")
    print("\n=== 处理 test (19个) ===")
    te_rows = process_split(te, "test")

    output = []
    output.append(
        summary_block(
            tr_rows, "TRAIN (112个,*为监控子集)", monitor_names=monitor_names
        )
    )
    output.append(
        summary_block_simple(mo_rows, "MONITOR (12个,train子集,用于选best ckpt)")
    )
    output.append(summary_block_simple(te_rows, "TEST (19个,nnUNet fold0,最终对比)"))
    output.append(
        count_summary([tr_rows, mo_rows, te_rows], ["train", "monitor", "test"])
    )

    txt = "\n".join(output)
    print("\n\n" + txt)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\n已保存到 {OUT_TXT}")
