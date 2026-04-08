import torch
from typing import Dict
import matplotlib.pyplot as plt


def get_views(vol: torch.Tensor, idxs: Dict[str, int]):
    # 返回 3 个方向切片, 并转成常见显示方向
    axial = vol[idxs["axial"], :, :]  # [H,W]
    coronal = vol[:, idxs["coronal"], :]  # [D,W]
    sagittal = vol[:, :, idxs["sagittal"]]  # [D,H]

    return [
        axial.numpy(),
        coronal.numpy(),
        sagittal.numpy(),
    ]


def pick_slice_indices(mask_or_label: torch.Tensor) -> Dict[str, int]:
    """
    选三个方向上更有信息量的切片:
    - 如果有前景, 取前景中心
    - 如果没有前景, 取体积中心
    输入: [D,H,W]
    返回: {"axial": z, "coronal": y, "sagittal": x}
    coords.numel()是元素总数,
    torch.nonzero()返回非零元素的索引,as_tuple=False返回一个二维张量
    coords是一个二维tensor(矩阵),shape是[N,3],N是非体素个数,3是坐标(z,y,x)
    as_tuple=True返回一个元组,元组中包含三个一维tensor,分别对应z,y,x坐标
    as_tuple=False返回一个二维tensor,shape是[N,3]
    """
    assert mask_or_label.ndim == 3

    D, H, W = mask_or_label.shape
    coords = torch.nonzero(mask_or_label > 0, as_tuple=False)

    if coords.numel() == 0:
        return {
            "axial": D // 2,
            "coronal": H // 2,
            "sagittal": W // 2,
        }

    center = coords.float().mean(dim=0)
    # 这里的dim=0是对每一列求均值，即对z列,y列,x列分别求均值,这个均值就算所有前景体素的重心,选择重心位置的切片,往往是信息最多的切片
    z = int(round(center[0].item()))
    y = int(round(center[1].item()))
    x = int(round(center[2].item()))

    z = max(0, min(z, D - 1))
    y = max(0, min(y, H - 1))
    x = max(0, min(x, W - 1))

    return {
        "axial": z,
        "coronal": y,
        "sagittal": x,
    }


def save_case_visualization(
    save_path: str,
    image: torch.Tensor,  # [1,D,H,W]
    label: torch.Tensor | None,  # [1,D,H,W] or None
    pred1: torch.Tensor,  # [D,H,W]
    tumor_full: torch.Tensor,  # [D,H,W]
    final_pred: torch.Tensor,  # [D,H,W]
    case_name: str,
    liver_filled: torch.Tensor | None = None,  # [D,H,W] bool, 孔洞填充后的肝脏
) -> None:
    """
    保存 one-case 可视化:
    行 = axial/coronal/sagittal
    列 = image / gt / stage1_liver / [liver_filled] / stage2_tumor / final_pred
    liver_filled 不为 None 时额外显示一列
    """
    image3d = image[0].cpu()
    gt3d = label[0].cpu() if label is not None else None

    # 优先根据GT选切片, 没GT就根据预测选
    ref = gt3d if gt3d is not None else final_pred
    idxs = pick_slice_indices(ref)

    img_views = get_views(image3d, idxs)
    gt_views = get_views(gt3d, idxs) if gt3d is not None else [None, None, None]
    liver_views = get_views(pred1, idxs)
    tumor_views = get_views(tumor_full, idxs)
    final_views = get_views(final_pred, idxs)

    show_filled = liver_filled is not None
    if show_filled:
        filled_views = get_views(liver_filled.cpu().long(), idxs)

    row_names = ["axial", "coronal", "sagittal"]
    col_names = ["image", "gt", "stage1_liver"]
    if show_filled:
        col_names.append("liver_filled")
    col_names += ["stage2_tumor", "final_pred"]

    n_cols = len(col_names)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.6 * n_cols, 10))

    for r in range(3):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            col_name = col_names[c]

            if col_name == "image":
                ax.imshow(img_views[r], cmap="gray")
            elif col_name == "gt":
                if gt_views[r] is not None:
                    ax.imshow(gt_views[r], cmap="gray", vmin=0, vmax=2)
                else:
                    ax.text(0.5, 0.5, "No GT", ha="center", va="center")
            elif col_name == "stage1_liver":
                ax.imshow(liver_views[r], cmap="gray", vmin=0, vmax=1)
            elif col_name == "liver_filled":
                ax.imshow(filled_views[r], cmap="gray", vmin=0, vmax=1)
            elif col_name == "stage2_tumor":
                ax.imshow(tumor_views[r], cmap="gray", vmin=0, vmax=1)
            elif col_name == "final_pred":
                ax.imshow(final_views[r], cmap="gray", vmin=0, vmax=2)

            if r == 0:
                ax.set_title(col_name, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_names[r], fontsize=11)

    fig.suptitle(case_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
