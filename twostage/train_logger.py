import os
import csv
import math
from datetime import datetime


class TrainLoggerTwoStage:
    """
    two-stage 专用日志器

    设计目标：
    1. 支持只写 tumor（stage2）
    2. 也支持同时写 liver + tumor
    3. 不再用 val_liver 决定整个验证是否存在
    4. 与原始 TrainLogger 隔离，放在 twostage/ 里更安全
    """

    def __init__(self, workdir: str):
        self.csv_path = os.path.join(workdir, "log.csv")
        self.txt_path = os.path.join(workdir, "log.txt")
        self.workdir = os.path.abspath(workdir)
        self.fieldnames = [
            "time",
            "epoch",
            "train_loss",
            "val_liver_dice",
            "val_tumor_dice",
            "best_score",
            "lr",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"{'time':<14} {'epoch':>5} {'loss':>7} "
                f"{'liver':>7} {'tumor':>7} {'best':>7} {'lr':>9}\n"
            )
            f.write("-" * 68 + "\n")

    @staticmethod
    def _is_valid_number(x) -> bool:
        """
        判断 x 是否是有效数字：
        - None -> False
        - NaN  -> False
        - 其他 float/int -> True
        """
        if x is None:
            return False
        try:
            return not math.isnan(float(x))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _fmt_metric(x: float) -> str:
        return f"{float(x):.4f}"

    def log(self, epoch, train_loss, val_liver, val_tumor, best, lr):
        now = datetime.now().strftime("%m-%d %H:%M")

        has_liver = self._is_valid_number(val_liver)
        has_tumor = self._is_valid_number(val_tumor)

        row = {
            "time": now,
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 4),
            "val_liver_dice": round(float(val_liver), 4) if has_liver else "",
            "val_tumor_dice": round(float(val_tumor), 4) if has_tumor else "",
            "best_score": round(float(best), 4),
            "lr": round(float(lr), 6),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

        liver_s = self._fmt_metric(val_liver) if has_liver else "   -   "
        tumor_s = self._fmt_metric(val_tumor) if has_tumor else "   -   "

        line = (
            f"{now:<14} {int(epoch):>5} {float(train_loss):>7.4f} "
            f"{liver_s:>7} {tumor_s:>7} {float(best):>7.4f} {float(lr):>9.2e}\n"
        )

        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)

    def log_extra(self, epoch: int, **kwargs):
        """记录额外的键值对到 extra_log.csv"""

        path = os.path.join(self.workdir, "extra_log.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + list(kwargs.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({"epoch": epoch, **kwargs})
