# twostage_medseg_scaffold

独立于 `medseg_project` 的 two-stage 外壳：

- **不重做 120GB 的 `.pt` 数据**
- **不污染原来的 `medseg_project` 主干**
- **直接复用 `medseg_project` 的模型 / ckpt / logger / transforms**
- **stage2 训练时在线从 `.pt` 中裁 GT liver ROI**
- **two-stage 推理时用 stage1 预测 liver ROI，再跑 stage2 tumor**

## 目录

```text
 twostage_medseg_scaffold/
 ├── scripts/
 │   ├── train_tumor_roi.py
 │   └── infer_twostage.py
 └── twostage/
     ├── __init__.py
     ├── roi_utils.py
     ├── dataset_tumor_roi.py
     └── train_eval_tumor.py
```

## 依赖关系

这个工程本身不复制 `medseg_project`，而是运行时通过 `--medseg_root` 指向它。

例如：

```bash

cd /home/pumengyu/twostage_medseg

python -m scripts.train_tumor_roi \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments_twostage \
  --exp_name tumor_roi_dynunet \
  --model dynunet \
  --epochs 200 \
  --batch_size 1 \
  --lr 0.003 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 6 \
  --num_workers 2 \
  --amp \
  --loss dicece \
  --overlap 0.5 \
  --repeats 3 \
  --tumor_ratios 0.2 0.8 \
  --margin 12 \
  --resume /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt



```

## 训练逻辑

1. 读取你已有的 `.pt` case。
2. 在线根据 `label > 0` 得到 GT liver mask。
3. 取 liver bbox，并按 `margin` 外扩。
4. 裁出 liver ROI。
5. 把标签映射为 tumor 二分类：
   - 0 = non-tumor
   - 1 = tumor (`label == 2`)
6. 复用 `medseg_project/medseg/data/transforms_offline.py` 做 patch 采样和增强。
7. 复用 `medseg_project/medseg/engine/train_eval.py` 做训练和滑窗验证。

## 推理逻辑

1. 用 stage1 liver 模型在**全图**推理，输出 liver 前景。
2. 根据 stage1 预测的 liver mask 取 bbox。
3. 在该 ROI 上运行 stage2 tumor 模型。
4. 再把 tumor ROI 贴回整图。

## 注意

- 训练 stage2 用的是 **GT liver ROI**。
- 推理 two-stage 用的是 **stage1 预测 liver ROI**。
- 这两者不是完全一致，这是 cascade 常见做法。
- 如果你后面要追求论文级闭环一致性，再考虑引入 “predicted-liver ROI training”。

