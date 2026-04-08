import torch
from pathlib import Path

root = "/home/pumengyu/Task03_Liver_pt"
paths = list(Path(root).glob("*.pt"))
neg = sum(
    1
    for p in paths
    if not (
        torch.load(p, map_location="cpu", weights_only=False, mmap=True)["label"] == 2
    ).any()
)
print(f"总共 {len(paths)} cases，无肿瘤 {neg} cases ({neg / len(paths) * 100:.1f}%)")
