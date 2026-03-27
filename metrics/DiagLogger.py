from __future__ import annotations


import os

from datetime import datetime

import torch


class DiagLogger:
    """
    训练诊断日志
    目的：
      1. 启动时记录数据集统计，检查数据是否正常
      2. 训练中记录指标变化
      3. 生成可直接贴给AI的诊断摘要
      4. 规避数据泄露、标签错误等灾难性问题

    生成文件：
      diag.txt      人类可读的完整诊断日志
      diag_summary.txt  精简摘要,专门用于贴给AI
    """

    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)
        self.txt_path = os.path.join(workdir, "diag.txt")
        self.summary_path = os.path.join(workdir, "diag_summary.txt")
        self._summary_lines: list[str] = []

    # ──────────────────────────────────────────
    # 内部写入
    # ──────────────────────────────────────────

    def _write(self, line: str = ""):
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _both(self, line: str = ""):
        """同时写入 diag.txt 和 summary 缓存"""
        self._write(line)
        self._summary_lines.append(line)

    def _flush_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._summary_lines) + "\n")

    def _section(self, title: str):
        self._write()
        self._write("=" * 55)
        self._write(f"  {title}")
        self._write("=" * 55)

    # ──────────────────────────────────────────
    # 1. 数据集基本信息
    # ──────────────────────────────────────────

    def log_dataset(self, args, tr, va, te, tr_pos, va_pos):
        self._section("启动信息")
        self._both(f"[时间]       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._both(f"[workdir]    {self.workdir}")
        self._write()
        self._both("[数据集规模]")
        self._both(
            f"  train 总数={len(tr)}  有肿瘤={len(tr_pos)}  "
            f"无肿瘤比例={(len(tr) - len(tr_pos)) / max(1, len(tr)) * 100:.1f}%"
        )
        self._both(
            f"  val   总数={len(va)}  有肿瘤={len(va_pos)}  "
            f"无肿瘤比例={(len(va) - len(va_pos)) / max(1, len(va)) * 100:.1f}%"
        )
        self._both(f"  test  总数={len(te)}")
        self._write()
        self._both("[训练配置]")
        self._both(f"  model={args.model}  loss={args.loss}  lr={args.lr}")
        self._both(f"  patch={args.patch}  batch_size={args.batch_size}")
        self._both(f"  tumor_ratios={args.tumor_ratios}  margin={args.margin}")
        self._both(f"  learnable_loss={args.learnable_loss}")
        self._both(f"  epochs={args.epochs}  val_every={args.val_every}")
        self._both(f"  amp={args.amp}  val_overlap={args.val_overlap}")
        self._flush_summary()

    # ──────────────────────────────────────────
    # 2. 数据泄露检查
    # ──────────────────────────────────────────

    def check_data_leakage(self, tr, va, te):
        """
        检查 train/val/test 是否有重叠
        用文件名（不含路径）做唯一标识
        """
        self._section("数据泄露检查")

        def names(paths):
            return set(os.path.basename(p) for p in paths)

        tr_names = names(tr)
        va_names = names(va)
        te_names = names(te)

        tr_va = tr_names & va_names
        tr_te = tr_names & te_names
        va_te = va_names & te_names

        if not tr_va and not tr_te and not va_te:
            status = "✅ 无泄露"
        else:
            status = "❌ 发现泄露！"

        self._both(f"[数据泄露] {status}")

        if tr_va:
            self._both(f"  ⚠ train∩val={len(tr_va)}个: {sorted(tr_va)[:5]}")
        if tr_te:
            self._both(f"  ⚠ train∩test={len(tr_te)}个: {sorted(tr_te)[:5]}")
        if va_te:
            self._both(f"  ⚠ val∩test={len(va_te)}个: {sorted(va_te)[:5]}")

        self._flush_summary()
        return len(tr_va) + len(tr_te) + len(va_te) == 0

    # ──────────────────────────────────────────
    # 3. 标签统计
    # ──────────────────────────────────────────

    def log_label_stats(self, pt_paths, tag: str = "train", max_cases: int = 10):
        """
        统计每个case的label分布
        检查项：
          - label值域是否只有 {0,1,2} 或 {0,1}
          - tumor体素占比（是否极度稀少）
          - 是否存在全背景case
        """
        self._section(f"Label统计 [{tag}]（抽样{min(max_cases, len(pt_paths))}个）")

        liver_ratios = []
        tumor_ratios = []
        all_bg_cases = []
        unexpected_label_cases = []

        for p in pt_paths[:max_cases]:
            try:
                data = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                label = data["label"].long()
                total = label.numel()
                unique_vals = torch.unique(label).tolist()

                liver_v = int((label == 1).sum())
                tumor_v = int((label == 2).sum())
                bg_v = int((label == 0).sum())

                liver_pct = liver_v / total * 100
                tumor_pct = tumor_v / total * 100

                liver_ratios.append(liver_pct)
                tumor_ratios.append(tumor_pct)

                # 异常检查
                expected = {0, 1, 2}
                unexpected = set(unique_vals) - expected
                if unexpected:
                    unexpected_label_cases.append((os.path.basename(p), unexpected))

                if liver_v == 0 and tumor_v == 0:
                    all_bg_cases.append(os.path.basename(p))

                flag = ""
                if tumor_v == 0:
                    flag = "  ← ⚠ 无tumor"
                elif tumor_pct < 0.05:
                    flag = "  ← 极稀少tumor"

                self._write(
                    f"  {os.path.basename(p):30s} "
                    f"bg={bg_v:>8}  liver={liver_v:>8}({liver_pct:5.2f}%)  "
                    f"tumor={tumor_v:>7}({tumor_pct:5.3f}%){flag}"
                )
            except Exception as e:
                self._write(f"  {os.path.basename(p)}: ❌ 读取失败: {e}")

        # 汇总统计
        if tumor_ratios:
            import statistics

            self._write()
            self._both(f"[{tag} 汇总]")
            self._both(
                f"  liver占比: 中位={statistics.median(liver_ratios):.2f}%  "
                f"min={min(liver_ratios):.2f}%  max={max(liver_ratios):.2f}%"
            )
            self._both(
                f"  tumor占比: 中位={statistics.median(tumor_ratios):.3f}%  "
                f"min={min(tumor_ratios):.3f}%  max={max(tumor_ratios):.3f}%"
            )

        # 警告汇总
        if all_bg_cases:
            self._both(f"  ❌ 全背景case（无liver无tumor）: {all_bg_cases}")
        if unexpected_label_cases:
            self._both(f"  ❌ 非预期label值: {unexpected_label_cases}")

        self._flush_summary()

    # ──────────────────────────────────────────
    # 4. ROI质量检查（Stage2专用）
    # ──────────────────────────────────────────

    def log_roi_stats(
        self, pt_paths, tag: str = "train", max_cases: int = 10, tumor_label=2
    ):
        """
        Stage2的输入是裁剪后的肝脏ROI
        检查：
          - ROI的实际尺寸分布
          - ROI里tumor占比（直接影响Stage2难度）
          - 是否有ROI里tumor=0（说明Stage1漏掉了）
        """
        self._section(f"ROI质量检查 [{tag}]")

        shapes = []
        tumor_in_roi = []
        empty_roi = []

        for p in pt_paths[:max_cases]:
            try:
                data = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                img = data["image"]
                label = data["label"].long()

                shape = tuple(img.shape[-3:])
                shapes.append(shape)

                total = label.numel()
                tumor_v = int((label == tumor_label).sum())  # Stage2里tumor=1
                pct = tumor_v / total * 100
                tumor_in_roi.append(pct)

                flag = ""
                if tumor_v == 0:
                    empty_roi.append(os.path.basename(p))
                    flag = "  ← ❌ ROI内无tumor"
                elif pct < 0.1:
                    flag = "  ← ⚠ tumor极少"

                self._write(
                    f"  {os.path.basename(p):30s} "
                    f"shape={str(shape):20s}  "
                    f"tumor={tumor_v:>7}({pct:5.2f}%){flag}"
                )
            except Exception as e:
                self._write(f"  {os.path.basename(p)}: ❌ 读取失败: {e}")

        if shapes:
            import statistics

            d_list = [s[0] for s in shapes]
            h_list = [s[1] for s in shapes]
            w_list = [s[2] for s in shapes]
            self._write()
            self._both(f"[{tag} ROI尺寸]")
            self._both(
                f"  D: 中位={statistics.median(d_list):.0f}  "
                f"min={min(d_list)}  max={max(d_list)}"
            )
            self._both(
                f"  H: 中位={statistics.median(h_list):.0f}  "
                f"min={min(h_list)}  max={max(h_list)}"
            )
            self._both(
                f"  W: 中位={statistics.median(w_list):.0f}  "
                f"min={min(w_list)}  max={max(w_list)}"
            )

        if tumor_in_roi:
            import statistics

            self._both(f"[{tag} ROI内tumor占比]")
            self._both(
                f"  中位={statistics.median(tumor_in_roi):.2f}%  "
                f"min={min(tumor_in_roi):.2f}%  max={max(tumor_in_roi):.2f}%"
            )

        if empty_roi:
            self._both(f"  ❌ ROI内无tumor的case: {empty_roi}")

        self._flush_summary()

    # ──────────────────────────────────────────
    # 5. 训练过程记录
    # ──────────────────────────────────────────

    def log_epoch(
        self, epoch, train_loss, val_tumor_dice, best, w_liver=None, w_tumor=None
    ):
        alpha_str = ""
        if w_liver is not None and w_tumor is not None:
            alpha_str = f"  w_liver={w_liver:.3f} w_tumor={w_tumor:.3f}"

        line = (
            f"[epoch {epoch:>4}] "
            f"loss={train_loss:.4f}  "
            f"tumor_dice={val_tumor_dice:.4f}  "
            f"best={best:.4f}"
            f"{alpha_str}"
        )
        self._write(line)
        # 每10个val epoch也写入summary
        if epoch % 60 == 0:
            self._both(line)
            self._flush_summary()

    def log_final(self, best_tumor_dice, best_epoch, total_hours):
        self._section("训练结束")
        self._both("[最终结果]")
        self._both(f"  best_tumor_dice = {best_tumor_dice:.4f}")
        self._both(f"  best_epoch      = {best_epoch}")
        self._both(f"  total_hours     = {total_hours:.2f}h")
        self._both(f"[结束时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._flush_summary()
