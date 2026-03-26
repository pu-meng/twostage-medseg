# 实验记录 — Two-Stage 肝脏/肿瘤分割

**任务:** MSD Task03 Liver，两阶段分割（Stage1 肝脏 → Stage2 肿瘤）
**模型:** DynUNet (MONAI)
**数据:** 预处理后的 .pt 文件，共约 131 cases，按 seed=0 划分
- train: 92 cases（有肿瘤 82，无肿瘤 10）
- val:   26 cases
- test:  13 cases

---

## Stage 1 — 肝脏模型（所有实验共用，不再重训）

| 项目 | 值 |
|------|----|
| 目录 | `dynunet_liver_only/train/03-14-01-11-56` |
| 模型 | DynUNet |
| Optimizer | SGD, lr=3e-3, momentum=0.99, weight_decay=3e-5 |
| LR Scheduler | Poly 0.9 |
| Loss | DiceCE |
| Patch | 144×144×144 |
| Epochs | 200 |
| batch_size | 1 |
| 数据备注 | label 1（肝脏）和 label 2（肿瘤）合并为 1 训练，train=92, val=26 |
| 训练时长 | 22.93h |
| GPU | NVIDIA RTX 2080 Ti |
| **Val liver Dice (best)** | **0.8905**（epoch 198） |
| **Test liver Dice** | **0.9594**（后续 eval 中测得） |

---

## Stage 2 实验一 — 基础肿瘤模型

**目录:** `twostage/tumor_dynunet/train/03-17-09-22-42`

| 项目 | 值 |
|------|----|
| init_ckpt | Stage 1 best.pt（迁移学习） |
| Optimizer | AdamW, lr=3e-3 |
| LR Scheduler | CosineAnnealingLR, T_max=300 |
| Loss | DiceFocal |
| Patch | 96×96×96 |
| Epochs | 300 |
| batch_size | 1 |
| 数据 | **train=82（仅肿瘤阳性 case），val=24** |
| bbox_jitter | ❌ 无 |
| random_margin | ❌ 无，固定 margin=8 |
| tumor_ratios | [0.05, 0.95]（5% 背景，95% tumor 前景采样） |
| repeats | 6 |
| 训练时长 | 29.02h |
| **Val tumor Dice (best)** | **0.3801**（epoch 264） |

### Eval 结果（test split，13 cases）

> ⚠️ 这三次 eval 没有保存完整命令，具体参数（margin/overlap 等）不可考。

| eval 时间 | liver Dice | tumor Dice | 备注 |
|-----------|-----------|-----------|------|
| 03-18-15-13 | 0.8927 | 0.5096 | liver Dice 偏低，推测 stage1 ckpt 不同或未做最大连通分量过滤 |
| 03-18-16-29 | 0.9594 | **0.5677** | 本次最优 |
| 03-18-17-34 | 0.9594 | 0.5401 | 结果略差，原因不明 |

---

## Stage 2 实验二 — ROI Jitter 肿瘤模型

**核心改动：**
1. 训练数据从 82 扩展到 **92 cases**（加入无肿瘤 case）
2. 新增 **bbox_jitter**（max_shift=8）：随机扰动 bbox 边界，模拟 Stage1 预测误差
3. 新增 **random_margin**（范围 [8, 20]）：随机化 ROI margin，模拟 bbox 尺度变化

### Run 1（中断，仅作热启动）

| 项目 | 值 |
|------|----|
| 目录 | `tumor_dynunet_roi_jitter/train/03-22-10-47-09` |
| init_ckpt | Stage 1 best.pt |
| batch_size | 1 |
| 实际完成 | **仅 3 epoch 即中断** |
| 备注 | last.pt 被 Run 2 的 --resume 继承 |

### Run 2（主体训练，接续 Run 1）

**目录:** `twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00`

| 项目 | 值 |
|------|----|
| resume | Run 1 的 last.pt（从第 3 epoch 继续） |
| init_ckpt | Stage 1 best.pt（同时保留，用于 shape 匹配） |
| Optimizer | AdamW, lr=3e-3 |
| LR Scheduler | CosineAnnealingLR, T_max=300 |
| Loss | DiceFocal |
| Patch | 96×96×96 |
| Epochs | 300 |
| batch_size | **2**（从 Run1 的 1 改为 2） |
| sw_batch_size | 2 |
| num_workers | 4 |
| bbox_jitter | ✅ max_shift=8 |
| random_margin | ✅ margin 范围 [8, 20] |
| 数据 | train=92, val=26（含无肿瘤 case） |
| 训练时长 | **45.92h** |
| **Val tumor Dice (best)** | **0.4697**（epoch 243） |

**训练曲线（每60epoch快照）：**

| epoch | loss | val tumor Dice | best |
|-------|------|---------------|------|
| 60 | 0.5158 | 0.1809 | 0.2478 |
| 120 | 0.4591 | 0.2958 | 0.3768 |
| 180 | 0.3439 | 0.3882 | 0.4277 |
| 240 | 0.3367 | 0.4294 | 0.4430 |
| 300 | 0.3111 | 0.4415 | **0.4697** |

### Eval 结果（test split，13 cases，eval 时间 03-24-10-42）

| 指标 | mean | std | min | max |
|------|------|-----|-----|-----|
| liver Dice | 0.9594 | 0.026 | 0.879 | 0.976 |
| tumor Dice | **0.5816** | 0.288 | 0.023 | 1.000 |
| tumor Recall | 0.491 | 0.253 | 0.000 | 0.853 |
| tumor Precision | 0.631 | 0.377 | 0.000 | 0.989 |
| tumor FDR | 0.292 | 0.341 | 0.000 | 0.988 |
| tumor FNR | 0.432 | 0.243 | 0.000 | 0.923 |
| tumor Jaccard | 0.389 | 0.259 | 0.000 | 0.755 |

**Per-case 明细：**

| case | liver Dice | tumor Dice | Recall | FDR | pred_tumor_voxels |
|------|-----------|-----------|--------|-----|-------------------|
| liver_7 | 0.9709 | 0.860 | 0.853 | 0.132 | 26228 |
| liver_87 | 0.9764 | 1.000 | 0.000 | 0.000 | 0（无肿瘤 case，正确预测为空） |
| liver_27 | 0.9745 | 0.805 | 0.828 | 0.216 | 110624 |
| liver_78 | 0.9600 | 0.786 | 0.673 | 0.057 | 19710 |
| liver_36 | 0.9700 | 0.773 | 0.680 | 0.106 | 44437 |
| liver_37 | 0.9741 | 0.752 | 0.662 | 0.130 | 16444 |
| liver_40 | 0.9758 | 0.702 | 0.553 | 0.040 | 103445 |
| liver_71 | 0.9479 | 0.478 | 0.315 | 0.011 | 103552（大肿瘤，只找到 31%） |
| liver_4 | 0.8793 | 0.511 | 0.354 | 0.080 | 558902（超大肿瘤，只找到 35%） |
| liver_92 | 0.9738 | 0.459 | 0.360 | 0.366 | 797（预测极少） |
| liver_107 | 0.9715 | 0.327 | 0.548 | 0.767 | 10067（严重误报） |
| liver_77 | 0.9344 | 0.086 | 0.077 | 0.902 | 15348（几乎全漏+严重误报） |
| liver_15 | 0.9633 | 0.023 | 0.485 | 0.988 | 24145（预测区域 99% 是错的） |

**问题分析：**
- FNR=0.432（主因）：liver_4、liver_71 为超大肿瘤，模型只能找到约 1/3
- FDR=0.292（次因）：liver_15、liver_77、liver_107 存在严重误报

---

## Stage 2 实验三 — 困难样本 Fine-tune（进行中）

**目录:** `twostage/tumor_dynunet_hardfinetune/train/…`

| 项目 | 值 |
|------|----|
| init_ckpt | 实验二 best.pt（03-22-11-44-00） |
| Optimizer | AdamW, lr=**3e-4**（降低一个数量级） |
| LR Scheduler | CosineAnnealingLR, T_max=60 |
| Loss | DiceFocal |
| Patch | 96×96×96 |
| Epochs | **60** |
| batch_size | 2 |
| hard_cases_only | ✅ 仅训练小肿瘤（<5000 voxel）+ 无肿瘤 case |
| small_tumor_thresh | 5000 voxels |
| normal_case_ratio | 0.2（混入 20% 普通样本防灾难遗忘） |
| bbox_jitter | ✅ max_shift=8 |
| random_margin | ✅ [8, 20] |
| **目标** | **降低 FNR（当前 0.432），提升 Recall** |
| 状态 | **训练中** |

---

---

## 代码改动记录 — transforms 增强（2026-03-25）

**动机：** 实验二/三的 per-case 分析显示两类明显问题：
- 大肿瘤覆盖不全（liver_4 Recall=0.354、liver_71 Recall=0.315），根因是 `num_samples=1` 每个 case 只采 1 个 patch，大肿瘤只有局部被看到
- 小肿瘤 loss 信号弱，模型对小结构不敏感

**改动文件：**

| 文件 | 改动 |
|------|------|
| `medseg_project/medseg/data/transforms_offline.py` | `num_samples: 1→2`；新增 `RandZoomd(min_zoom=0.7, max_zoom=1.4, prob=0.3, keep_size=True)` |
| `twostage/dataset_tumor_roi.py` | 移除 `len(out)>1` 的 RuntimeError，允许 transform 返回 list（交由 `list_data_collate` 处理） |
| `scripts/train_tumor_roi.py` | train DataLoader 加 `collate_fn=list_data_collate`，将 num_samples=2 的 list 展平为 `batch_size*2` 的 tensor batch |

**机制说明：**
- `num_samples=2`：同一 case 同一步采 2 个独立前景 patch → 大肿瘤不同区域同时参与训练，覆盖更完整
- `RandZoomd zoom_in(>1.0)`：小区域放大填满 patch → 小肿瘤体素占比上升，loss 贡献更大
- `RandZoomd zoom_out(<1.0)`：内容压缩 + 补零 → 提升模型对不同尺度的鲁棒性

**注意：** `num_samples=2` 使有效 batch size 翻倍（`batch_size * 2`），训练时如遇显存不足需将 `--batch_size` 减半。

---

## nnUNet 官方基线（论文对比用）

> **来源：** `/home/pumengyu/nnUNet_result/`，官方 nnUNet v1 流程，**仅训练 1 个 fold（fold 0）**，非 5-fold ensemble。
> **数据：** MSD Task03 Liver，验证集 **19 cases**（nnUNet 自动划分，与本项目 seed=0 划分不同，不可直接对比 per-case）。
> **训练时间：** 2026-03-18 22:41 — 2026-03-19（见 training_log_*.txt）。
> **模型文件：** `model_best.model` / `model_best.model.pkl`，fold_0 子目录有 `model_final_checkpoint`。

### 验证集 Mean 指标（fold 0，19 cases）

| 指标 | Liver (Class 1) | Tumor (Class 2) |
|------|----------------|----------------|
| **Dice** | **0.9652** | **0.7569** |
| Jaccard | 0.9330 | 0.6444 |
| Precision | 0.9706 | 0.8797 |
| Recall | 0.9605 | 0.7133 |
| Accuracy | 0.9981 | 0.9995 |
| Dice std | 0.0111 | 0.2146 |
| Dice min | 0.9417 | 0.0887 |
| Dice max | 0.9825 | 0.9583 |

**关键数字（写论文用）：**
- Liver Dice：**0.9652 ± 0.011**
- Tumor Dice：**0.7569 ± 0.215**（1-fold，5-fold ensemble 论文报告约 0.81~0.82）
- 所有 19 个验证 case 的 Tumor Dice 均 > 0（无全漏的 case）

**注意事项（论文写作时）：**
1. 这是 **1 fold** 结果，会比 nnUNet 论文（5-fold ensemble）低约 3~5 个点
2. 本项目 test 集（13 cases）与 nnUNet 验证集（19 cases）划分不同，两组数字**不在同一测试集上**，对比需注明
3. 若要公平对比，需在同一 test split（seed=0, test_ratio=0.1）上跑 nnUNet 推理

---

## 横向对比

| 实验 | 核心变化 | Val tumor Dice | Test tumor Dice |
|------|---------|---------------|----------------|
| 实验一 | 基础版，仅 82 阳性 case | 0.3801 | 0.5677（最优 eval） |
| 实验二 | +全 92 case +bbox_jitter +random_margin | **0.4697** | **0.5816** |
| 实验三 | fine-tune 困难样本 | 待测 | 待测 |

> Val Dice 和 Test Dice 不直接可比（val 是训练时滑窗验证，test 是完整两阶段推理）。

---

## 经验总结 — 超参数选择

### lr 选择（fine-tune from init_ckpt）

**场景：** 从已有 ckpt 出发，同时换了损失函数（如 tversky）+ 加大 patch（如 128³）

**教训（2026-03-25）：**
- lr=1e-4 收敛太慢 → epoch 5 时 tumor dice 仅 0.06，实验被提前终止
- 不要因为是 fine-tune 就把 lr 压得太低；换了 loss 和 patch 相当于重新适应，模型需要足够的学习率
- **建议范围：lr=5e-4 ~ 8e-4**（参考 dicefocal from-scratch 用 8e-4 跑出 dice=0.49@ep44）
- 若是在收敛后的 ckpt 上做小幅调整（同 loss、同 patch），才适合用 1e-4

---

## Eval 脚本说明（已修复的行为）

修复前：eval 输出目录用 eval 执行时间命名（难以对应训练 run）
修复后：eval 输出目录名取自 `stage2_ckpt` 的父目录名（与训练 run 一一对应）

例：`--stage2_ckpt .../train/03-22-11-44-00/best.pt` → 输出到 `.../eval/03-22-11-44-00/`
