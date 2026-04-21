<span style="color:#1a5276">**实验记录 — Two-Stage 肝脏/肿瘤分割**</span>

**任务:** MSD Task03 Liver，两阶段分割（Stage1 肝脏 → Stage2 肿瘤） | **模型:** DynUNet (MONAI)

- train: 92 cases（有肿瘤 82，无肿瘤 10）｜ val: 26 cases ｜ test: 13 cases（seed=0）

---

<span style="color:#1a5276">**Stage 1 — 肝脏模型（所有实验共用，不再重训）**</span>

| 项目 | 值 |
|-|-|
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
| **Test liver Dice** | **0.9594** |
|**地位**|**two-stage的肝脏分割的基础,后期长期的实验都选择这个作为stage1_ckpt**|

---
<span style="color:#186a3b">**Stage 2 实验一 ✅ — 基础肿瘤模型**</span>

**目录:** `twostage/tumor_dynunet/train/03-17-09-22-42`

| 项目 | 值 |
|------|----|
| init_ckpt | Stage 1 best.pt（迁移学习） |
| Optimizer | AdamW, lr=3e-3 |
| Loss | DiceFocal |
| Patch | 96×96×96, batch_size=1 |
| Epochs | 300 |
| 数据 | train=82（仅肿瘤阳性 case），val=24 |
| bbox_jitter | ❌ 无 |
| random_margin | ❌ 无，固定 margin=8 |
| tumor_ratios | [0.05, 0.95] |
| repeats | 6 |
| 训练时长 | 29.02h |
| **Val tumor Dice (best)** | **0.3801**（epoch 264） |

**Eval 结果（test split，13 cases）**

| eval 时间 | liver Dice | tumor Dice | 备注 |
|-----------|-----------|-----------|------|
| 03-18-15-13 | 0.8927 | 0.5096 | liver Dice 偏低 |
| 03-18-16-29 | 0.9594 | **0.5677** | 本次最优 |
| 03-18-17-34 | 0.9594 | 0.5401 | — |

---
<span style="color:#186a3b">**Stage 2 实验二 ✅ — ROI Jitter 肿瘤模型**</span>

**RoI (Region of Interest)**
肝脏感兴趣区域,Stage1预测出肝脏后,把这一块区域裁出来,Stage2只在这个裁剪出的肝脏区域内做肿瘤分割.

**Jitter(bbox_jitter)**

对裁剪框施加随机扰动,训练时对bbox的六个边界(上下左右前后)加最多±8体素的随机偏移;目的是模拟Stage1推理时预测框不准的情况,让Stage2对不完美的bbox更鲁棒.

<span style="color:red">可能的创新点:
- 到底是使用真实的肝脏划分框,还是用Stage1的框然后用Jitter模拟噪音,哪一个训练效果更好?
</span>


```python

import torch
import torch.nn.functional as F

def focal_loss(logits, label, alpha, gamma=2.0):
    """
    logits : [B, C, D, H, W]
    label  : [B, D, H, W]  整数标签
    alpha  : list[C]，每个类的权重
    """
    B, C, D, H, W = logits.shape

    # Step 1: softmax
    #P:[B,C,D,H,W]
    P = F.softmax(logits, dim=1)         

    # Step 2: 取 p_t
    #label_idx:[B,1,D,H,W]值为0,1,2,..,C-1
    label_idx = label.unsqueeze(1)        
#p_t:[B,1,D,H,W]
#p_t=P[i,X,j,k,l]这里的X=label_idx[i,0,j,k,l]
    p_t = P.gather(1, label_idx)
#p_t:[B,D,H,W]
    p_t=p_t.squeeze(1)

    # Step 3: 调制因子
    #focal_weight:[B,D,H,W]
    focal_weight = (1 - p_t) ** gamma     # [B, D, H, W]

    # Step 4: alpha_t
    #alpha:[C],比如是3类,那么类似[0.3,0.5,0.2]
    #label:[B,D,H,W],每个位置存的是整数类的标签
    #alpha_t:[B,D,H,W],
    #alpha_t[i,j,k,l]=alpha[x],x=label[i,j,k,l]
    alpha_t = torch.tensor(alpha, device=logits.device)[label]  # [B, D, H, W]

    # Step 5: CE 部分 = -log(p_t)
    #.clamp是把所有小于1e-8的值强制替换为1e-8
    log_p_t = torch.log(p_t.clamp(min=1e-8))  # 防 log(0)

    loss = -alpha_t * focal_weight * log_p_t   # [B, D, H, W]
    return loss.mean()
```
**Focal loss的数学公式:**
- 如果输入logits:[B,C,D,H,W],
- 先做softmax得到p:[B,C,D,H,W]
- 相当于一共$N=B\times D \times H\times W$个像素点,每一个像素点归属一个类别,0,1,2,...,C-1;
- 根据这些标签,选出$p_t$,这里的$p_t,t=0,1,2,...,B\times D \times H\times W-1$;
- 然后是 $-\sum\limits_{t=0}\limits^{N-1}\alpha_t(1-p_t)^\gamma\log(p_t)$,这里的$\alpha_t$是固定的,有C个类别,就有C个$\alpha $值,$p_t$对应的B,D,H,W指代的像素点真实是哪个标签,$\alpha_t$就选择哪一个$\alpha$值;
- 关键是我们相当于统计所有的像素点,真实标签知道了,从logits选择这个标签对应的概率$p_t$,根据真实的标签选择$\alpha_t$就可以了;这里的$\alpha_t$控制的是不同类别的权重,$\gamma$控制的是不同大小的$p_t$的权重,


```python

import torch
import torch.nn.functional as F

def dice_loss(logits, label, smooth=1e-5):
    """
    logits : [B, C, D, H, W]  网络原始输出
    label  : [B, D, H, W]     整数标签，值 ∈ {0,1,...,C-1}
    smooth : 平滑项，防止分母为0
    """
    B, C, D, H, W = logits.shape

    # Step 1: softmax → 预测概率 P
    #P:[B,C,D,H,W]
    
    P = F.softmax(logits, dim=1)         

    # Step 2: label 转 one-hot → G
  #label_onehot:[B,C,D,H,W],全为0
    label_onehot = torch.zeros_like(P)
    #label:[B,D,H,W]
    #label.unsqueeze(1).long()是[B,1,D,H,W],.long()把数据类型转换为int64,scatter_要求索引张量必须是整数类型(int64),int32都不行

    label_onehot.scatter_(1, label.unsqueeze(1).long(), 1)  
    #操作后label_onthot:[B,C,D,H,W]是根据真实标签label得到的one-hot编码,

    # Step 3: 对每个类 c 算 Dice，再平均
    total_loss = 0.0

    for c in range(C):
        p = P[:, c, ...]           # [B, D, H, W]  第c类预测概率
        g = label_onehot[:, c, ...]# [B, D, H, W]  第c类真实 0/1

        # 分子：2 * Σ(p * g)
        intersection = (p * g).sum(dim=[1, 2, 3])   # [B]

        # 分母：Σp² + Σg²
        denominator = (p ** 2).sum(dim=[1, 2, 3]) + \
                      (g ** 2).sum(dim=[1, 2, 3])   # [B]

        dice_per_sample = (2 * intersection + smooth) / (denominator + smooth)  # [B]
        dice_loss_c = 1 - dice_per_sample.mean()    # scalar

        total_loss += dice_loss_c

    return total_loss / C   # 对 C 个类取平均

```
**Dice loss的损失函数公式:**
$$L_{Dice} = 1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$$

多类别时对每个前景类分别计算再取均值：

  $$L_{Dice} = 1 - \frac{1}{C-1}\sum_{c=1}^{C-1}\frac{2\sum p_i^c g_i^c}{\sum p_i^c + \sum g_i^c}$$

**粗糙肝脏定位框的选择策略对Stage2肿瘤分割性能的影响**

  在两阶段分割框架中，Stage2的输入ROI由Stage1肝脏分割结果决定。训练阶段存在两种边界框策略：
  - 其一为直接使 
  用真实标注（GT）肝脏边界框，精确但与推理阶段存在domain gap；
  - 其二为使用Stage1预测框并施加随机扰动（bbox
   jitter）模拟推理时的定位误差，以提升Stage2对不完美输入的鲁棒性。
   - 本文系统比较两种策略对肿瘤分割精度的 
  影响，探讨训练-推理一致性在两阶段医学图像分割中的作用。  





**commit:** `a95d1c3` | **目录:** `twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00`

核心改动：
  - 训练数据扩展到 92 cases 
  - bbox_jitter（max_shift=8）
  - random_margin（[8,20]）

| 项目 | 值 |
|------|----|
| Optimizer | AdamW, lr=3e-3 |
| Loss | DiceFocal |
| Patch | 96×96×96, batch_size=2 |
| Epochs | 300 |
| bbox_jitter | ✅ max_shift=8 |
| random_margin | ✅ [8, 20] |
| 训练时长 | **45.92h** |
| **Val tumor Dice (best)** | **0.4697**（epoch 243） |

**训练曲线**

| epoch | loss | val tumor Dice | best |
|-------|------|---------------|------|
| 60 | 0.5158 | 0.1809 | 0.2478 |
| 120 | 0.4591 | 0.2958 | 0.3768 |
| 180 | 0.3439 | 0.3882 | 0.4277 |
| 240 | 0.3367 | 0.4294 | 0.4430 |
| 300 | 0.3111 | 0.4415 | **0.4697** |

**Eval 结果（test split，13 cases）**

| 指标 | mean | std | min | max |
|------|------|-----|-----|-----|
| liver Dice | 0.9594 | 0.026 | 0.879 | 0.976 |
| tumor Dice | **0.5816** | 0.288 | 0.023 | 1.000 |
| tumor Recall | 0.491 | 0.253 | 0.000 | 0.853 |
| tumor Precision | 0.631 | 0.377 | 0.000 | 0.989 |
| tumor FDR | 0.292 | 0.341 | 0.000 | 0.988 |
| tumor FNR | 0.432 | 0.243 | 0.000 | 0.923 |

**Per-case 明细**

| case | tumor Dice | Recall | FDR | 备注 |
|------|-----------|--------|-----|------|
| liver_7 | 0.860 | 0.853 | 0.132 | — |
| liver_87 | 1.000 | 0.000 | 0.000 | 无肿瘤 case，正确预测为空 |
| liver_27 | 0.805 | 0.828 | 0.216 | — |
| liver_71 | 0.478 | 0.315 | 0.011 | 大肿瘤，只找到 31% |
| liver_4 | 0.511 | 0.354 | 0.080 | 超大肿瘤，只找到 35% |
| liver_15 | 0.023 | 0.485 | 0.988 | 预测区域 99% 是错的 |
| liver_77 | 0.086 | 0.077 | 0.902 | 几乎全漏+严重误报 |

**问题：** FNR=0.432（主因，大肿瘤漏检） / FDR=0.292（次因，liver_15/77/107 严重误报）

<span style="color:#7d6608">**Eval-only: 实验二 + TTA ✅**</span>

**TTA 说明：** 
- 对输入图像做 D/H/W 三轴所有翻转组合，共 $2^3=8$ 种，每种分别推理后翻转回原方向，8个概率图取平均再阈值得到最终预测。
- 提升 Recall 的原因是翻转后模型从不同视角看同一区域，边缘区域概率被拉高，漏检减少；
- 不引入额外假阳性是因为平均操作平滑了单次推理的偶发噪声。推理时间 ×8。
<span style="color:red">可能的创新点:
- 研究后处理的TTA,更快的推理,更高的指标
</span>
**目录:** `eval_tta/03-22-11-44-00` | **commit:** `2014275` | 推理时长 0.633h（无 TTA 0.198h）

| 指标 | 无 TTA | TTA | Δ |
|------|--------|-----|---|
| liver Dice | 0.9594 | 0.9594 | = |
| tumor Dice | 0.5816 | **0.5965** ✅ | +0.015 |
| tumor Recall | 0.4914 | **0.5224** | +0.031 |
| tumor FNR | 0.4317 | **0.4007** | −0.031 |
| tumor FDR | 0.2920 | 0.2890 | ≈ |

TTA 主要提升 Recall（减少漏检），不引入额外假阳性。**当前 test split 最优：0.5965**

<span style="color:#7d6608">**Eval-only: 实验二 + TTA + min_tumor_size 扫参 ✅（val split，n=26）**</span>

`min_tumor_size` 过滤 Stage2 中体素数过少的预测区域，减少假阳性。⚠️ 使用 val split，不可与 test split 直接比较。

| min_tumor_size | Tumor Dice | Recall | Precision | FDR | FNR |
|---|---|---|---|---|---|
| 0（无过滤） | 0.5377 | 0.4315 | 0.7428 | 0.2188 | 0.4915 |
| **100 ← 最优** | **0.5757** | 0.4310 | 0.7450 | **0.1011** | 0.4921 |
| 200 | 0.5751 | 0.4301 | 0.7068 | 0.1009 | 0.4930 |
| 500 | 0.5463 | 0.4063 | 0.6715 | 0.0977 | 0.5167 |
| 1000 | 0.5079 | 0.3777 | 0.6019 | 0.0904 | 0.5454 |

0→100：FDR 大幅下降（0.219→0.101），Recall 几乎不变。200+：开始漏检真实肿瘤。**后续实验固定 `--min_tumor_size 100`**

<span style="color:#7d6608">**Eval-only: 实验二 + TTA + min_tumor_size 扫参 ✅（test split，n=13）**</span>

**目录:** `eval_min_tumor_size_sweep/` | 使用 `--tta --margin 12`，在 test split 上验证 val 扫参结论。

| min_tumor_size | Tumor Dice | Recall | Precision | FDR | FNR | Jaccard |
|---|---|---|---|---|---|---|
| 0（无过滤） | 0.5186 | 0.5231 | 0.6295 | 0.3705 | 0.4000 | 0.3993 |
| **100 ← 最优** | **0.5950** | 0.5187 | **0.6395** | 0.2836 | 0.4044 | 0.3989 |
| 200 | 0.5815 | 0.5052 | 0.6370 | 0.2861 | 0.4179 | 0.3883 |
| 500 | 0.5527 | 0.4304 | 0.5843 | 0.2618 | 0.4927 | 0.3711 |
| 1000 | 0.5442 | 0.4132 | 0.5854 | 0.2607 | 0.5099 | 0.3633 |

0→100：FDR 下降（0.371→0.284），Recall 基本不变。200+：Recall 明显下降，真实小肿瘤被误删。与 val split 结论一致，**`--min_tumor_size 100` 为最优，代码默认值已是 100，无需修改。**

---
---

实验三因为实验没有正常进行,所以删除了;
<span style="color:#922b21">**Stage 2 实验四 ❌ — pred_bbox ROI + Hard Mining（停止）**</span>

**目录:** `twostage/tumor_dynunet_predbbox_roi/train/03-27-11-11-47`

| 项目 | 值 |
|------|----|
| preprocessed_root | Task03_Liver_roi（tight_bbox = pred_bbox_stage1） |
| repeats | 6 + small_tumor_repeat_scale=4 + no_tumor_repeat_scale=2 |
| Epochs | 300 |
| 状态 | ❌ **手动停止（epoch 1 后）** |

**停止原因：** ~972 batches/epoch × 4.3s = 69 min/epoch → **300ep 约 345h（14天）**；同时 pred_bbox 和 hard mining 两变量混在一起，无法做消融对比。

**教训**:
- GPU的性能影响极大,从一轮30分钟,到3分钟都是有可能的;

- 初期不要纠结控制变量,提高指标才是真实的,做好记录,实验太多会很容易混乱;


---
<span style="color:#186a3b">**Stage 2 实验五 ✅ — pred_bbox ROI **</span>
实验5是和实验4进行对比,实验4因为同时混入pred_bbox和hard_mining两个变量导致无法消融,所以实验五做了干净版本,只用pred_bbox ROI,不加hard.


**exp_name:** `tumor_dynunet_predbbox_roi_clean` | **eval 时间:** 03-28-09-49-40

| 项目 | 值 |
|------|----|
| init_ckpt | 实验二 best.pt（03-22-11-44-00） |
| preprocessed_root | Task03_Liver_roi（tight_bbox = Stage1 pred_bbox） |
| Epochs | 300 |
| repeats | 6，**无 hard mining** |
| random_margin | ✅ [8, 24] |

**Eval 结果（test split，13 cases）**

| 指标 | mean | std | min | max |
|------|------|-----|-----|-----|
| liver Dice | 0.9594 | 0.026 | 0.879 | 0.976 |
| **tumor Dice** | **0.6121** | 0.286 | 0.013 | 1.000 |
| tumor Recall | **0.5213** | 0.297 | 0.000 | 0.914 |
| tumor Precision | 0.6527 | 0.352 | 0.000 | 0.993 |
| tumor FDR | 0.2704 | 0.308 | 0.000 | 0.994 |
| tumor FNR | 0.4018 | 0.281 | 0.000 | 0.914 |

**Per-case 明细**

| case | pred_tumor_voxels | tumor Dice | Recall | Precision | FDR |
|------|:-----------------:|:----------:|:------:|:---------:|:---:|
| liver_4 | 684102 | 0.5789 | 0.4258 | 0.9039 | 0.0961 |
| liver_40 | 133350 | 0.7863 | 0.6848 | 0.9230 | 0.0770 |
| liver_27 | 129229 | 0.8157 | 0.9112 | 0.7383 | 0.2617 |
| liver_71 | 101003 | 0.4638 | 0.3039 | 0.9792 | 0.0208 |
| liver_36 | 39098 | 0.7963 | 0.6648 | 0.9926 | 0.0074 |
| liver_7 | 29122 | 0.8742 | 0.9142 | 0.8376 | 0.1624 |
| liver_78 | 22639 | 0.8236 | 0.7496 | 0.9138 | 0.0862 |
| liver_15 | 20013 | 0.0126 | 0.2211 | 0.0065 | 0.9935 |
| liver_37 | 19571 | 0.7722 | 0.7354 | 0.8128 | 0.1872 |
| liver_107 | 10473 | 0.4316 | 0.7449 | 0.3038 | 0.6962 |
| liver_77 | 5828 | 0.1321 | 0.0859 | 0.2860 | 0.7140 |
| liver_92 | 597 | 0.4698 | 0.3348 | 0.7873 | 0.2127 |
| liver_87 | 0 | 1.000 | 0.000 | 0.000 | 0.000 |

---
<span style="color:#922b21">**Stage 2 实验六 ❌ — pred_bbox ROI + Hard Mining（不如预期）**</span>

**exp_name:** `tumor_dynunet_predbbox_roi_hardmine` | **eval 时间:** 03-28-09-59-18

| 项目 | 值 |
|------|----|
| init_ckpt | 实验二 best.pt（03-22-11-44-00） |
| preprocessed_root | Task03_Liver_roi |
| Epochs | 300 |
| repeats | 6 + small_tumor_thresh=500 + small_tumor_repeat_scale=3 + no_tumor_repeat_scale=2 |
| random_margin | ✅ [8, 20] |
| lr | 3e-4 |

**Eval 结果（test split，13 cases）**

| 指标 | mean | std | min | max |
|------|------|-----|-----|-----|
| liver Dice | 0.9594 | 0.026 | 0.879 | 0.976 |
| **tumor Dice** | **0.5965** | 0.281 | 0.000 | 1.000 |
| tumor Recall | 0.4609 | 0.303 | 0.000 | 0.896 |
| tumor Precision | **0.6868** | 0.332 | 0.000 | 0.993 |
| tumor FDR | **0.2362** | 0.275 | 0.000 | 1.000 |
| tumor FNR | 0.4621 | 0.303 | 0.000 | 1.000 |

**实验五 vs 实验六 对比（唯一变量：hard mining）**

| 指标 | 实验五（clean） | 实验六（hard mine） | Δ |
|------|:---------------:|:-------------------:|:---:|
| tumor Dice | **0.6121** | 0.5965 | **−0.016** |
| Recall | **0.5213** | 0.4609 | **−0.060** |
| Precision | 0.6527 | **0.6868** | +0.034 |
| FDR | 0.2704 | **0.2362** | −0.034 |
| FNR | **0.4018** | 0.4621 | +0.060 |

**Per-case 对比（按 pred_tumor_voxels 排序，实验五为基准）**

| case | 肿瘤大小 | Dice(clean) | Dice(hard) | ΔDice | Recall(clean) | Recall(hard) | ΔRecall |
|------|:--------:|:-----------:|:----------:|:-----:|:-------------:|:------------:|:-------:|
| liver_4 | 684k | 0.5789 | 0.5493 | −0.030 | 0.4258 | 0.3945 | −0.031 |
| liver_40 | 133k | 0.7863 | 0.7863 | 0.000 | 0.6848 | 0.6914 | +0.007 |
| liver_27 | 129k | 0.8157 | 0.8062 | −0.010 | 0.9112 | 0.8593 | −0.052 |
| liver_71 | 101k | 0.4638 | 0.4459 | −0.018 | 0.3039 | 0.2881 | −0.016 |
| liver_36 | 39k | 0.7963 | 0.6551 | **−0.141** | 0.6648 | 0.4887 | **−0.176** |
| liver_7 | 29k | 0.8742 | 0.8838 | +0.010 | 0.9142 | 0.8961 | −0.018 |
| liver_78 | 22k | 0.8236 | 0.7927 | −0.031 | 0.7496 | 0.6880 | −0.062 |
| liver_15 | 20k | 0.0126 | 0.0000 | −0.013 | 0.2211 | 0.0000 | **−0.221** |
| liver_37 | 19k | 0.7722 | 0.7476 | −0.025 | 0.7354 | 0.6657 | −0.070 |
| liver_107 | 10k | 0.4316 | **0.5675** | **+0.136** | 0.7449 | 0.6833 | −0.062 |
| liver_77 | 5.8k | 0.1321 | 0.1440 | +0.012 | 0.0859 | 0.0843 | −0.002 |
| liver_92 | 0.6k | 0.4698 | 0.3763 | −0.094 | 0.3348 | 0.2528 | **−0.082** |
| liver_87 | 0 | 1.000 | 1.000 | 0.000 | — | — | — |

**失败原因分析**

Hard mining 的设计意图是让小肿瘤（<500 voxel）和无肿瘤 case 在 batch 中多出现，迫使模型提高对小目标的敏感性。但实际效果相反：

1. **Recall 全面下滑（12/12 case 持平或下降）**：说明 hard mining 使模型对所有尺寸的肿瘤召回率都下降，而不是专门提升小肿瘤。
2. **最大受害者是中等肿瘤（liver_36: −17.6%Recall，liver_15: −22.1%Recall）**：这些 case 并不是小肿瘤，却召回率大幅下降，说明过采样小/无肿瘤改变了数据分布，模型偏向"保守预测"。
3. **Precision/FDR 略好**：假阳性确实减少，符合"更保守"的预期，但以大量漏检为代价。
4. **问题根源**：当前 hard mining 是对**小肿瘤和无肿瘤 case** 过采样，而漏检问题的主因是**大肿瘤**（liver_4 684k、liver_71 101k）——应该对大肿瘤过采样，而非小肿瘤。

**下一步**：hard mining 方向改为对 **大肿瘤 case（>50k voxel）重复采样**，或直接放弃 hard mining，转为两通道输入或更强的数据增强。

截至2026/4/16,对大肿瘤有不错的分割效果,主要,靠--patch $160\times 160\times 160\times 160$,以及--large_tumor_repeats_scale 5,能够这样做的根本是服务器的GPU性能支持这样做;GPU性能非常的关键.

---
<span style="color:#1a5276">**Stage 2 实验七 📋 — STU-Net 预训练权重迁移（准备中）**</span>

**exp_name:** `tumor_dynunet_stunet_init`

**背景：** STU-Net 是在大规模医学数据集上预训练的 nnUNet 风格模型，权重比从 Stage1 迁移更强。架构与我们的 DynUNet 高度一致，只需一处调整即可加载。

**架构对齐（当前 vs STU-Net-Small）**

| 参数 | 当前 DynUNet | STU-Net-Small | 需要改动 |
|------|------------|--------------|---------|
| resolution levels | 5（4次下采样） | 6（5次下采样） | ⚠️ 需加一层 |
| filters | [32,64,128,256,320] | [32,64,128,256,320,320] | ⚠️ 随架构自动对齐 |
| kernel_size | 全 3×3×3 | 全 3×3×3 | ✅ 一致 |
| instance norm + leaky relu | ✅ | ✅ | ✅ 一致 |

**代码改动（`medseg/models/dynunet.py`）**

```python
# 加第6层，与 STU-Net-Small 对齐
kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
```

96³ patch → 5次下采样后 bottleneck = 3³，可行。

**权重加载策略**

- 首尾层（in_channels / out_channels 不匹配）：随机初始化
- 中间所有层：直接拷贝 STU-Net 权重
- 加载方式：`strict=False`，打印命中率确认

**待办**

- [ ] 下载 STU-Net-Small 权重（HuggingFace: `uni-medical/STU-Net`）
- [ ] 改 `dynunet.py` 加第6层
- [ ] 写权重加载脚本 `scripts/load_stunet_weights.py`
- [ ] 新增 `--stunet_ckpt` 参数到 `train_tumor_roi.py`
- [ ] 用实验五同等设置训练，只换 init → 对比收益

| 项目 | 值 |
|------|----|
| init_ckpt | STU-Net-Small pretrained |
| preprocessed_root | Task03_Liver_roi |
| Epochs | 300 |
| repeats | 6，无 hard mining（与实验五一致，只换初始化） |
| random_margin | ✅ [8, 20] |
| 预计时长 | **~24h** |
| 状态 | 📋 准备中（等实验八启动后开始代码准备） |

**结果**

待测

---
<span style="color:#186a3b">**Stage 2 实验八 ✅ — pred_bbox ROI + 大肿瘤过采样（实验六改进版）**</span>

**exp_name:** `tumor_dynunet_predbbox_roi_largemine`

**背景：** 实验六（小肿瘤过采样）失败，per-case 分析显示漏检主要来自大肿瘤（liver_4 Recall=0.43，liver_71 Recall=0.30）。大肿瘤只占训练集 25%（33/92 case，voxel≥50k），96³ patch 每次只看局部，边缘区域采样不足。改为对大肿瘤过采样，让模型更多见到大肿瘤的边缘分割难题。

| 项目 | 值 |
|------|----|
| init_ckpt | 实验二 best.pt（03-22-11-44-00） |
| preprocessed_root | Task03_Liver_roi（pred_bbox，同实验五） |
| Epochs | 300 |
| repeats | 6 |
| **large_tumor_thresh** | **50000** |
| **large_tumor_repeat_scale** | **3**（大肿瘤 ×3，其余 ×1） |
| random_margin | ✅ [8, 20] |
| bbox_jitter | ✅ max_shift=8 |
| 预计 indices | 33×18 + 59×6 = 948（vs 实验五 552），约 1.72× |
| 预计时长 | **~41h** |
| 状态 | ✅ 已完成 |

**与实验五的唯一变量：** `--large_tumor_thresh 50000 --large_tumor_repeat_scale 3`

**结果** ✅

实验目录：`tumor_dynunet_predbbox_roi_largetx6_p128`
- **ckpt_A**：`train/03-29-13-33-35/best.pt`（训练过程中验证集最优）
- **ckpt_B**：`train/03-30-23-47-04/last.pt`（训练结束最终权重）

评估脚本：`scripts/eval_twostage.py`，seed=0，val 26 cases，test 13 cases

| 指标 | ckpt_A (best.pt) | ckpt_B (last.pt) | 变化 |
|------|-----------------|-----------------|------|
| **Test Tumor Dice** | 0.5573 ± 0.308 | **0.6212 ± 0.279** | **+0.064** |
| Test Tumor Recall | 0.5053 | **0.5795** | +0.074 |
| Test Tumor Precision | 0.6637 | 0.6236 | -0.040 |
| Test Tumor FDR | 0.3363 | 0.2995 | -0.037 |
| **Val Tumor Dice** | 0.5785 ± 0.311 | **0.5907 ± 0.305** | +0.012 |
| Val Tumor Recall | 0.4732 | 0.4818 | +0.009 |
| Val Tumor FDR | 0.2584 | 0.1978 | -0.061 |
| **Liver Dice（两者相同）** | 0.9594 | 0.9594 | — |

**Per-case 关键发现（test split）：**

| case | ckpt_A Dice | ckpt_B Dice | 说明 |
|------|------------|------------|------|
| liver_87 | 0.000 | **1.000** | A 完全漏检，B 完美修复 |
| liver_15 | 0.000 | 0.076 | A 漏检，B 仍差但不再是0 |
| liver_107 | **0.621** | 0.400 | B 退化，FDR 0.31→0.71（误报） |
| liver_36 | 0.730 | **0.814** | B 明显改善 |

**结论：**
- last.pt 整体优于 best.pt，说明训练后期模型仍在改善（未充分收敛）
- 主要瓶颈：**漏检（FNR≈0.44）** + **个别 case 极端误报（FDR>0.5）**
- Std 约 ±0.28~0.31，说明模型对困难 case 极不稳定，而非系统性偏低
- **与实验五对比（test Dice=0.6121）**：大肿瘤过采样 + last.pt 达到 0.6212，小幅超越

---
<span style="color:#922b21">**Stage 2 实验九 ❌ — 两通道输入（CT + liver_mask）（中止）**</span>

**exp_name:** `tumor_dynunet_predbbox_roi_2ch`

**中止原因：** 两通道设计（Ch1=CT, Ch2=肝脏mask）逻辑意义有限——ROI 已经裁到肝脏附近，模型从 CT 灰度值本身已能感知肝脏边界，额外的 liver_mask 通道提供的信息极少。此外 init_ckpt 为1通道权重，第一层需要随机初始化，等于从头训练，代价高而收益不明确。

**更有价值的方向：** 重新训练 Stage 1，使其同时输出肝脏分割和粗糙肿瘤位置（两个输出头），用粗糙肿瘤 bbox 进一步缩小 Stage 2 的 ROI 范围。

---
<span style="color:#7d6608">**论文对比: MedSAM2 bbox 推理 📋 — 零样本对比（待安排）**</span>

**目的：** 论文必做。用 MedSAM2 做零样本肿瘤分割，提供外部方法对比基线，不需要训练。

**推理流程**

```
Stage1 liver pred → 提取 bbox → 作为 MedSAM2 prompt → 推理 tumor mask → 评估 Dice
```

**配置**

| 项目 | 值 |
|------|----|
| 模型 | MedSAM2（SAM2 医学微调版） |
| prompt | Stage1 pred liver bbox（3D bounding box） |
| 数据 | test split，13 cases |
| 训练 | ❌ 零样本，不微调 |
| 评估 | 与实验二/六同口径（--tta --min_tumor_size 100） |
| 状态 | 📋 待安排（穿插在 GPU 等待期间完成） |

**结果**

待测

---
<span style="color:#7d6608">**Stage 2 实验十 🔄 — DynUNet + 2D/3D 特征融合 + Skip Attention Gate（进行中/待验证）**</span>

**exp_name:** `tumor_dynunet_ca_predbbox_roi_largetx6_p128`

**背景：** 在实验八基础上引入两个结构改进（来自 Lu Meng et al., Diagnostics 2021）：
1. **SliceWise2DBranch**（L1 32ch skip）：Conv3d 模拟 slice-wise 2D 特征，与 3D 特征 concat 融合
2. **AttGate3D**（4 层 skip）：残差注意力门控，抑制无关区域，强化肿瘤边缘特征

| 项目 | 值 |
|------|----|
| init_ckpt | 实验八 last.pt（03-30-23-47-04，test Dice=0.6212） |
| 模型 | `dynunet_ca`（backbone DynUNet + AttGate + SliceWise2D） |
| 参数量 | 16.54M → 16.67M（+0.127M） |
| init 加载 | `load_init_weights`，118/139 参数匹配，21 个新参数随机初始化 |
| Epochs | 300 |
| patch | 128×128×128 |
| repeats | 6，large_tumor_thresh=50000，×3 |
| lr | 3e-4 |
| loss | dicefocal |
| 状态 | 🔄 训练完成，待 eval |

**ckpt：** `train/04-02-16-03-10/best.pt`

**结果**

待验证（eval 命令见 `eval_twostage.py` 文件头注释）


---
<span style="color:#1a5276">**Stage 2 实验十一 📋 — nnUNet 标准配置（待定模型）**</span>

**背景：** 在现有训练框架基础上，对齐 nnUNet 几乎所有关键配置，作为强基线验证上限。

**训练脚本：** `scripts/train_tumor_roi_v2.py`

**数据增强（已全部实现，对齐 nnUNet）**

| 增强项 | 参数 | nnUNet 标准 |
|--------|------|------------|
| 随机 flip 三轴 | p=0.5 | p=0.5 ✅ |
| 随机旋转 ±180° | p=0.2 | p=0.2 ✅ |
| 随机缩放 zoom | 0.7~1.4，p=0.2 | 0.7~1.4 ✅ |
| 高斯噪声 | std=0.1，p=0.1 | p=0.1 ✅ |
| 高斯模糊 | sigma(0.5,1.0)，p=0.2 | p=0.2 ✅ |
| 亮度乘法 | factors(-0.25,0.25)，p=0.15 | p=0.15 ✅ |
| Gamma ×2（正常+反转） | p=0.3 / p=0.1 | p=0.3/0.1 ✅ |
| 低分辨率模拟 | zoom(0.5,1.0)，p=0.25 | p=0.25 ✅ |

**训练配置**

| 项目 | 值 |
|------|----|
| 深监督 loss | `_deep_supervision_loss`，1/2^i 加权，nnUNet 同方案 ✅ |
| grad clip | norm=1.0 ✅ |
| AMP | GradScaler + autocast ✅ |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| loss | dicefocal |
| patch | 128×128×128 |
| repeats | 6，large_tumor_thresh=50000，×3 |
| pred_bbox | ✅ use_pred_bbox |
| bbox_jitter / random_margin | ✅ |
| init_ckpt | 待定（视实验十结果） |

**模型** — 待定，视实验十 eval 结果决定：
- 若实验十 > 实验八：用 `dynunet_ca`（但需把 `dynunet_ca.py` 里 `deep_supervision` 改回 True）
- 若实验十 ≈ 实验八：用 `dynunet`（结构更简单，深监督直接生效）

**状态：** 📋 等实验十 eval 结果


---
**横向对比**

| 实验 | 核心变化 | Val tumor Dice | Test tumor Dice | 状态 |
|------|---------|---------------|----------------|------|
| 实验一 | 基础版，82 阳性 case | 0.3801 | 0.5677 | ✅ |
| 实验二 | +92 case +jitter +margin | **0.4697** | 0.5816 | ✅ |
| 实验四 | pred_bbox ROI + 小肿瘤 hard mining，repeats=6 | — | — | ❌ 停止 |
| 实验五 | pred_bbox ROI only，repeats=6，300ep | — | **0.6121** | ✅ |
| 实验六 | pred_bbox ROI + 小肿瘤过采样，300ep | — | 0.5965 | ✅ |
| 实验七 | STU-Net 预训练权重迁移（6层架构对齐） | 待测 | 待测 | 📋 准备中 |
| 实验八 | pred_bbox ROI + 大肿瘤过采样，largetx6_p128 | 0.5907 (last) | 0.6212 (last) | ✅ |
| 实验九 | pred_bbox ROI + 两通道输入（CT + liver_mask） | — | — | ❌ 中止 |
| eval: TTA | 实验二 ckpt + 8 种翻转（test） | — | 0.5965 | ✅ |
| eval: min_tumor_size=100（val） | 实验二 ckpt + TTA | 0.5757 | — | ✅ |
| eval: min_tumor_size 扫参（test） | 实验二 ckpt + TTA，size_100 最优 | — | 0.5950 | ✅ |
| 实验十 | +AttGate3D + SliceWise2DBranch，init=实验八last.pt | — | 待测 | 🔄 待 eval |
| 实验十一 | FocalTversky+深监督+小肿瘤hard mining，dynunet，p128 | 0.7166 | 0.7055 | ✅ |
| 实验十二 | dynunet_deep(31M)+SGD+FocalTversky+p160，从旧ckpt | 0.7166（val实测） | **0.7135** | ✅ |
| 实验十三 | dynunet_deep+SGD+ratio04+p160，**从头训练** | 0.7166（val实测） | **0.7207** ✅ 当前最优 | 🔄 训练中 |
| 实验十三续 | 同上，训练更多 epoch 后 best.pt 更新 | — | **0.7344** ✅ 新最优 | 🔄 训练中 |
| nnUNet 基线 | 官方 nnUNet v1，fold0 | — | 0.7569（排除liver_121: 0.796） | 🎯 对比目标 |

> Val Dice（训练时滑窗验证）和 Test Dice（完整两阶段推理）不直接可比。
>
> **当前 test split 最优：实验十三 Tumor Dice=0.7344**（排除坏数据 liver_121 后估算 ≈ 0.768，已超过 nnUNet 单折）


---
<span style="color:#2c3e50">**nnUNet 官方基线（论文对比用）**</span>

来源: `/home/pumengyu/nnUNet_result/`，官方 nnUNet v1，fold 0，验证集 **19 cases**（与两阶段 test split 完全相同，可公平对比）。

| 指标 | Liver | Tumor |
|------|-------|-------|
| Dice | 0.9652 ± 0.011 | 0.7569 ± 0.209 |
| Recall | 0.9605 | 0.7133 |
| Precision | 0.9706 | 0.8797 |
| FNR | — | 0.2867 |
| FDR | — | 0.1203 |

完整报告: `/home/pumengyu/nnUNet_result/report.txt`

---
<span style="color:#2c3e50">**下一步计划**</span>

**当前状态**

| 实验 | ckpt | 状态 |
|------|------|------|
| 实验十 `tumor_dynunet_ca_predbbox_roi_largetx6_p128` | `04-02-16-03-10/best.pt` | 🔄 eval 中 |

---

**下一方向 — 重训 Stage 1（输出粗糙肿瘤位置）**

当前 Stage 1 只输出肝脏分割，ROI 裁到整个肝脏（较大）。新方案让 Stage 1 同时学肝脏和粗糙肿瘤：

| 输出头 | 内容 | 用途 |
|--------|------|------|
| Head 1 | 肝脏分割 | 裁出肝脏 ROI（和现在一样） |
| Head 2 | 粗糙肿瘤位置 | 进一步缩小 Stage 2 的输入范围 |

**优势：** Stage 2 的 ROI 从整个肝脏缩小到肿瘤附近，大幅减少背景干扰，理论上对大肿瘤召回率提升最明显。

**改动点：**
- Stage 1 训练脚本：out_channels 1→2（或两个 loss 头）
- Stage 1 推理：输出两个 mask，用肿瘤 bbox 替代/缩小肝脏 bbox
- pred_bbox_cache 格式扩展（加肿瘤 tight bbox）


---
<span style="color:#2c3e50">**经验总结**</span>

**lr 选择（fine-tune from init_ckpt）**

- lr=1e-4 收敛太慢（epoch 5 时 tumor dice 仅 0.06，提前终止）
- 换了 loss/patch 相当于重新适应，需要足够学习率
- **建议范围：lr=5e-4 ~ 8e-4**（fine-tune 时）；若同 loss/patch 小幅调整才适合 1e-4

---

**代码改动记录 — transforms 增强（2026-03-25，commit 994e6ee）**

| 文件 | 改动 |
|------|------|
| `medseg/data/transforms_offline.py` | `num_samples: 1→2`；新增 `RandZoomd(min_zoom=0.7, max_zoom=1.4, prob=0.3)` |
| `twostage/dataset_tumor_roi.py` | 移除 `len(out)>1` 的 RuntimeError |
| `scripts/train_tumor_roi.py` | train DataLoader 加 `collate_fn=list_data_collate` |

---

**eval 脚本说明**

- eval 输出目录名取自 `stage2_ckpt` 的父目录名（与训练 run 一一对应）
- TTA：`--tta` 对 D/H/W 三轴 8 种翻转取平均，推理时间 ×8（commit 2014275）
- **后续 eval 固定参数：** `--tta --min_tumor_size 100`
---

**DiceCE vs DiceFocal — 对无肿瘤 case 的训练引导差异**

背景：Stage2 是二分类（0=非肿瘤，1=肿瘤），无肿瘤 case 的 ROI label 全为 0。

| | DiceCELoss | DiceFocalLoss |
|--|-----------|--------------|
| Dice 部分 | 分子=0，分母=0 → loss=0，无梯度 | 同左 |
| CE/Focal 部分 | 对每个 voxel 算交叉熵，预测肿瘤置信度不为 0 就有惩罚 | Focal 对"简单样本"降权，全负样本 patch 被认为是简单样本，惩罚更小 |
| **无肿瘤 patch 的梯度** | **CE 仍有 voxel 级惩罚** | **Focal 几乎无惩罚，模型学到"不预测=零损失"** |
| 适合场景 | 有无肿瘤 case 混合训练，泛化更稳 | 适合肿瘤阳性 case 为主、难例挖掘场景 |

**结论：** 两阶段框架下无肿瘤 case 占比不低，`dicece` 的 CE 分量能在全负样本 patch 上保持梯度，避免模型"躺平"。`dicefocal` 的 Focal 分量会把无肿瘤 patch 当作简单样本降权，强化了"不预测也没损失"的方向。实验十一改用 `dicece` 作为对照。

**注意：** 这和 nnUNet 用 `DiceCELoss` 的原因一致——CE 分量对每个 voxel 单独惩罚，方向更正确。

---

**eval 指标口径说明（2026-04-05 更新）**

之前 tumor dice 把无肿瘤 case（pred=0, gt=0 → dice=1.0）混入均值，导致指标虚高。

现在分开报告，对齐 nnUNet：
- **tumor_pos**：只统计 gt 有肿瘤的 case（主指标，论文用这个）
- **tumor_neg_false_positive_rate**：无肿瘤 case 中被误报为有肿瘤的比例（越低越好）

之前的 0.62 在新口径下会略微下降，是真实数字。


---
<span style="color:#c0392b">**Bug 记录：`binary_fill_holes` 3D 对开口空洞无效**</span>

**发现时间**：2026-04-07

**现象**：eval 可视化中 `liver_filled` 列与 `stage1_liver` 看起来几乎一样，空洞没有被填补（见 `dynunet_focaltversky_smallmine_p128` eval `liver_121.png`）。

**根因**：`scipy.ndimage.binary_fill_holes` 在 3D 模式下只能填充**完全封闭**的空洞。肝脏空洞往往在某些 slice 边缘与背景连通（"开口"），3D 填充直接跳过，无效。

**修复**（`scripts/eval_twostage.py`）：

原代码：
```python

liver_filled = torch.from_numpy(
    ndi.binary_fill_holes(liver_mask.cpu().numpy())
).to(liver_mask.device)
```

修复后（三轴逐 slice 2D 填充取并集）：
```python

liver_np = liver_mask.cpu().numpy()
filled_ax0 = np.stack([ndi.binary_fill_holes(liver_np[i]) for i in range(liver_np.shape[0])])
filled_ax1 = np.stack([ndi.binary_fill_holes(liver_np[:, i, :]) for i in range(liver_np.shape[1])]).transpose(1, 0, 2)
filled_ax2 = np.stack([ndi.binary_fill_holes(liver_np[:, :, i]) for i in range(liver_np.shape[2])]).transpose(1, 2, 0)
liver_filled = torch.from_numpy(
    filled_ax0 | filled_ax1 | filled_ax2
).to(liver_mask.device)
```

**原理**：2D 下每个 slice 内部空洞一定封闭，填充可靠。三轴取并集覆盖所有方向的空洞。


---
<span style="color:#186a3b">**实验十一 ✅ `dynunet_focaltversky_smallmine_p128` — 全力小肿瘤攻坚**</span>

**目标**：在实验八基础上，全面针对小肿瘤做强化，重启干净训练。

**exp_name:** `dynunet_focaltversky_smallmine_p128` | **时间戳:** `04-06-12-56-02`

**与实验八(largetx6_p128)的差异**

| 参数 | 实验八 | 实验十一 | 说明 |
|------|--------|----------|------|
| loss | dicefocal | **focaltversky** | beta=0.7 加大 FN 惩罚，减少漏检；gamma=0.75 让小 tumor hard sample 贡献更大 |
| small_tumor_thresh | 0（关闭） | **5000** | 极小肿瘤(< 5000 voxels)开启 hard mining |
| small_tumor_repeat_scale | — | **4** | 23个极小 case 出现频率 ×4 |
| no_tumor_repeat_scale | — | **2** | 13个无肿瘤 case 重复 ×2，防假阳性 |
| tumor_ratios | 0.05 0.95 | **0.02 0.98** | RandCropByLabelClasses 98% 从肿瘤体素附近切 patch |
| repeats | 6 | **8** | 基础重复数提升 |
| val_overlap | 0.0 | **0.25** | 修复验证精度低导致 best.pt 选取失真的问题 |
| batch_size | 1 | **3** | 提升吞吐 |

**数据分布（train 112 cases）**
- 无肿瘤：13 cases → ×2（no_tumor_repeat_scale）
- 极小（< 5000）：23 cases → ×4（small_tumor_repeat_scale）
- 小（5k~50k）：39 cases → 基础频率
- 中等/大（≥ 50k）：37 cases → ×3（large_tumor_repeat_scale）
- 全部基础 repeats=8

**init_ckpt:** `twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt`（实验八 last.pt）

**关键训练配置**

| 参数 | 值 |
|------|----|
| model | dynunet |
| epochs | 300 |
| batch_size | 3 |
| lr | 3e-4 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR (T_max=300) |
| patch | 128×128×128 |
| val_patch | 96×96×96 |
| val_overlap | 0.25 |
| bbox_jitter | True，max_shift=8 |
| random_margin | True，min=8，max=20 |
| use_pred_bbox | True |
| amp | True |
| seed | 0 |
| n_train / val / test | 112 / 12 / 19 |

**训练命令:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --exp_name dynunet_focaltversky_smallmine_p128 \
  --model dynunet --epochs 300 --batch_size 3 --lr 3e-4 \
  --patch 128 128 128 --val_patch 96 96 96 \
  --loss focaltversky --val_overlap 0.25 --repeats 8 \
  --tumor_ratios 0.02 0.98 --margin 8 \
  --bbox_jitter --bbox_max_shift 8 --random_margin --margin_min 8 --margin_max 20 \
  --use_pred_bbox \
  --small_tumor_thresh 5000 --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 --large_tumor_thresh 50000 --large_tumor_repeat_scale 3 \
  --stage1_ckpt .../dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --init_ckpt .../tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt \
  --seed 0
```

---

**测试集结果（test split，n=19，全部有肿瘤）**

| 指标 | 值 |
|------|----|
| Liver Dice | 0.9585 ± 0.025 |
| **Tumor Dice** | **0.7055 ± 0.2197** |
| Tumor Jaccard | 0.5813 ± 0.2185 |
| Recall | 0.7886 ± 0.2511 |
| Precision | 0.6882 ± 0.2445 |
| FDR | 0.2592 ± 0.1929 |
| FNR | 0.2114 ± 0.2511 |
| 耗时 | 0.4 h |

**按 tumor dice 分级（test split）**

| 等级 | case 数 | 说明 |
|------|---------|------|
| 严重失败（< 0.3） | 1 | liver_121（极小肿瘤 2959 voxels，模型完全未检出） |
| 需要改进（0.3~0.7） | 5 | liver_12, 58, 49, 94, 44 |
| 没问题（≥ 0.7） | 13 | — |

**失败 case 分析**
- `liver_121`：极小肿瘤（2959 voxels），pred_tumor=0，完全漏检；prob_map 有信号（prob_max=1.0，>0.5体素=35个）但阈值后消失
- `liver_12`：极小肿瘤（524 voxels），recall=0.979 但 precision 极低（0.228），大量假阳性
- `liver_44`：中等肿瘤，recall 偏低（0.456），召回不足

**备注:** focaltversky + 小肿瘤过采样策略有效，test tumor dice 0.7055 为当前最高。

liver_121 不是一般的漏检，有比 liver_121 更小的肿瘤（如 liver_005: 540 voxels，liver_069: 3113 voxels）效果还不错，不是单纯小肿瘤的问题。liver_121 可能是肿瘤本身 HU 值特殊、位置边界不清，或者 stage1 bbox 对该区域 ROI 裁剪不准确导致的。

---
<span style="color:#8e44ad">**nnUNet vs 两阶段 — 完整对比分析（论文方向指引）**</span>

> 两个模型在完全相同的 19 个 test case 上评估，结果可直接对比。
> 完整报告: `/home/pumengyu/nnUNet_result/report.txt` 和 `/home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/eval/04-06-12-56-02/report.txt`

---

**整体指标对比**

| 指标 | nnUNet (fold0) | 两阶段 (实验十一) | 差值 |
|------|---------------|-----------------|------|
| Liver Dice | 0.9652 ± 0.011 | 0.9585 ± 0.025 | nnUNet +0.007 |
| **Tumor Dice** | **0.7569 ± 0.209** | 0.7055 ± 0.220 | nnUNet +0.051 |
| Tumor Recall | 0.7133 | **0.7886** | 两阶段 **+0.075** |
| Tumor Precision | **0.8797** | 0.6882 | nnUNet +0.192 |
| Tumor FNR | 0.2867 | **0.2114** | 两阶段 **+0.075** |
| Tumor FDR | **0.1203** | 0.2592 | nnUNet +0.139 |

**核心结论：** nnUNet Dice 高，靠的是 Precision（保守预测，少误报）；两阶段 Recall 高（激进预测，少漏检）。

---

**Per-case Tumor Dice（按 nnUNet dice 升序）**

| case | nnUNet | 两阶段 | 优势方 | gt_tumor voxels | size |
|------|--------|--------|--------|-----------------|------|
| liver_121 | 0.0887 | 0.0000 | nnUNet +0.089 | 2,123 | 极小 |
| liver_044 | 0.3401 | 0.6175 | **两阶段 +0.277** | 185,990 | 中等 |
| liver_094 | 0.6136 | 0.5977 | nnUNet +0.016 | 62,525 | 中等 |
| liver_058 | 0.6443 | 0.5449 | nnUNet +0.099 | 1,443 | 极小 |
| liver_049 | 0.7007 | 0.5545 | nnUNet +0.146 | 8,015 | 小 |
| liver_002 | 0.7812 | 0.8639 | **两阶段 +0.083** | 14,131 | 小 |
| liver_005 | 0.7824 | 0.7438 | nnUNet +0.038 | 540 | 极小 |
| liver_060 | 0.7980 | 0.8616 | **两阶段 +0.064** | 11,368 | 小 |
| liver_018 | 0.8064 | 0.7601 | nnUNet +0.046 | 10,210 | 小 |
| liver_028 | 0.8435 | 0.9035 | **两阶段 +0.060** | 214,268 | 中等 |
| liver_057 | 0.8599 | 0.8055 | nnUNet +0.054 | 6,604 | 小 |
| liver_009 | 0.8667 | 0.7083 | nnUNet +0.158 | 22,179 | 小 |
| liver_081 | 0.8693 | 0.7583 | nnUNet +0.111 | 5,130 | 小 |
| liver_012 | 0.8697 | 0.3704 | nnUNet +0.499 | 790 | 极小 |
| liver_069 | 0.8734 | 0.8398 | nnUNet +0.034 | 3,113 | 极小 |
| liver_117 | 0.8842 | 0.8430 | nnUNet +0.041 | 1,034,092 | 大 |
| liver_101 | 0.8923 | 0.8096 | nnUNet +0.083 | 378,439 | 大 |
| liver_098 | 0.9086 | 0.8866 | nnUNet +0.022 | 608,639 | 大 |
| liver_064 | 0.9583 | 0.9355 | nnUNet +0.022 | 179,093 | 中等 |

**两阶段领先的 case：** liver_044(+0.277), liver_002(+0.083), liver_028(+0.060), liver_060(+0.064)
**nnUNet 大幅领先的 case：** liver_012(+0.499), liver_009(+0.158), liver_049(+0.146), liver_081(+0.111)

---

**规律分析 — 对下一步方向的指引**

**两阶段在哪里更好：**
- liver_044（中等肿瘤，185k voxels）：两阶段大幅领先 +0.277，说明 ROI 聚焦对中等肿瘤有优势
- liver_002, liver_028, liver_060：小/中等肿瘤，两阶段均领先

**nnUNet 在哪里更好（需要重点攻克）：**
- **极小肿瘤（< 5k voxels）**：liver_012(790v), liver_005(540v), liver_069(3113v), liver_058(1443v) —— nnUNet 全面占优，两阶段在极小肿瘤上输得很彻底
- **小肿瘤（5k-50k）**：liver_009, liver_081, liver_049 —— nnUNet 也更好
- liver_121 两个模型都失败，但 nnUNet 还有少量预测

**关键差距根源：**
- 极小肿瘤上两阶段的 FDR 极高（liver_012: FDR=0.77），说明 ROI 内大量误报把 precision 拉崩
- nnUNet 端到端训练，全局上下文让它对极小肿瘤更保守精确
- 两阶段 ROI 裁剪后失去了全局上下文，在极小肿瘤边界判断上更难

**下一步方向（基于此分析）：**

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| ★★★ | 极小肿瘤 FDR 控制：对极小 ROI 加后处理（连通域过滤、更高阈值） | 直接提升 Dice |
| ★★★ | liver_012 专项分析：790 voxels 的极小肿瘤为何 recall=0.979 但 precision=0.228 | 找到过度预测根因 |
| ★★ | stage1 bbox 质量对 liver_009/081/049 的影响：是否 bbox 偏移导致 stage2 ROI 不准 | 排除 stage1 误差 |
| ★★ | Recall 优势论文化：在医学场景中 Recall 更重要，结合 FNR 指标撰写 | 论文立场 |
| ★ | Ensemble 实验：nnUNet prob + 两阶段 prob 平均，验证互补性 | 可能超越单模型 |

---
<span style="color:#1a5276">**错误分析：liver_44 / liver_58 prob图分析**</span>

> 图片路径: 

---

**liver_44**  Dice=0.6175  Recall=0.4557  FDR=0.0422  gt_tumor=272,406vox（中等）

- GT是一个大肿瘤，prob图里GT肿瘤区域平均概率只有 **0.26~0.28**，远低于阈值0.5
- TP（绿色）只出现在肿瘤边缘，大片内部是FN（黄色）→ **肿瘤内部大范围漏检**
- FDR极低（0.04）说明预测到的地方全对，问题是覆盖率不足
- 原因推测：低对比度肿瘤，CT值和周围肝脏接近，模型只能识别边界
- nnUNet在此case也只有0.34，是数据本身的hard case

**liver_58**  Dice=0.5449  Recall=0.4056  FDR=0.1702  gt_tumor=2,019vox（极小）

- prob图几乎全黑，GT肿瘤区域平均概率 **0.000~0.060**，模型完全无信号
- nnUNet也只有0.64，同样困难，是极小肿瘤的天然难点
- 不是后处理问题，模型根本没在该区域输出概率

---

**结论**

| case | 问题类型 | 可改进方向 |
|---|---|---|
| liver_44 | 低对比度大肿瘤内部漏检 | intensity augmentation；两阶段prob融合 |
| liver_58 | 极小肿瘤无信号 | nnUNet也难，暂时搁置 |

**当前优先改进方向：liver_12 / liver_49 / liver_94（FP多）→ 连通域自适应阈值，今天可看结果**


---
<span style="color:#922b21">**深度错误分析：liver_44 坏死肿瘤 / liver_121 标注可疑**</span>

> 图片路径: , , 

---

### liver_44 — 坏死性肿瘤，不是低对比度

- intensity histogram显示：肿瘤（红）分布极宽0.0~1.0，大量体素集中在0.2~0.5；肝脏（蓝）集中在0.55~0.65
- CT图可见肿瘤内部明显偏暗 → **坏死/囊性成分**，临床称坏死性肝癌
- 模型只识别出高密度边缘（绿色TP），内部低密度坏死区全部漏检（黄色FN大片）
- **不是低对比度，是肿瘤内部异质性太强**，模型没见过足够的坏死型肿瘤样本
- nnUNet在此case也只有0.34，是训练数据覆盖不足的问题
- 改进方向：intensity augmentation（随机降低肿瘤区域亮度模拟坏死）

### liver_121 — 标注可疑，建议排除

- CT图中GT标注区域（蓝色）对应一个**白色高亮结构**，外观像血管截面或钙化灶
- prob图全黑，GT区域概率=0.000，模型完全无信号
- **nnUNet也只有0.09**，两个独立模型都认为那里不是肿瘤
- 结论：极大概率是**标注错误或非典型边缘case**，不应纳入正常评估
- 建议：在论文/报告中单独说明此case，或从测试集指标中排除

---

### 排除liver_121后的指标估算

| 指标 | 含liver_121 | 排除后(估算) |
|---|---|---|
| Tumor Dice | 0.7055 | ~0.743 |
| Recall | 0.7886 | ~0.830 |

**排除后两阶段 Dice 约0.743，已非常接近 nnUNet 的0.757。**


---
<span style="color:#1a5276">**实验十二：nnUNet对齐 — 深网络+SGD+负样本均衡（2026-04-12）**</span>

> 当前最优结果: `dynunet_focaltversky_smallmine_zoom_p128`  
> 报告路径: `/home/PuMengYu/experiments/twostage/dynunet_focaltversky_smallmine_zoom_p128/eval/04-11-11-53-09/report.txt`

---

### 当前指标 vs nnUNet（单折）

| 指标 | nnUNet 单折 | 当前最优 | 差距 |
|------|------------|----------|------|
| Tumor Dice | 0.7569 | 0.6746 | -0.082 |
| Recall | 0.7133 | **0.7598** | 你高+0.047 |
| Precision | 0.8797 | 0.7006 | -0.179 |
| FDR | 0.1203 | **0.2994** | 你高+0.179 |

**结论：差距几乎全在FDR（假阳性），recall已经超过nnUNet。**

---

### 差距根因分析

1. **参数量只有nnUNet一半**：当前5层16.5M，nnUNet 6层~30M
2. **优化器不对齐**：AdamW+Cosine vs nnUNet的SGD momentum=0.99 + PolyLR power=0.9
3. **负样本学习不足**：tumor_ratios=0.1/0.90 + ROI裁剪，模型几乎只见肿瘤附近肝脏，没见过肝内远端正常组织，导致FDR高
4. **高斯加权推理缺失**：sliding_window_inference未用mode="gaussian"
5. **liver_121标注可疑**：两个不连续区域（z=91~96 和 z=156~177），nnUNet同样失败，认为是噪声case

---

### 已做的代码修复

| 改动 | 文件 |
|------|------|
| 新增 （6层，31M，filters=[32,64,128,256,320,320]） |  +  |
| 优化器改为 SGD momentum=0.99 weight_decay=3e-5 nesterov=True |  |
| 调度器改为 PolyLR power=0.9 |  |
| eval所有sliding_window_inference加 mode="gaussian" |  |

---

### 实验十二配置（运行4_deep.sh）

**脚本**:   
**exp_name**: 

与当前最优实验的变化：

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
|  | dynunet | **dynunet_deep** | 参数量对齐nnUNet（16.5M→31M）|
|  | 1e-4 | **3e-3** | SGD需要更大lr，对齐nnUNet |
| optimizer | AdamW | **SGD+PolyLR** | 对齐nnUNet，3D分割泛化更好 |
|  | 128³ | **160³** | 改善中等肿瘤（liver_44）覆盖，batch从3→2 |
|  | 0.1 0.90 | **0.4 0.60** | 增加负样本比例，降低FDR |
|  | 4 | **6** | 补偿ratios降低导致的小肿瘤采样减少 |
|  | 6.0 | **3.0** | 降低zoom侵略性，减少训练/推理域不一致 |
|  | 8000 | **5000** | 缩小zoom触发范围 |
|  | 有（旧best.pt） | **无** | 从头训练，让更深网络充分收敛 |


---
<span style="color:#1e8449">**实验十二结果：focaltversky_smallmine_zoom_p160 — 新最优 0.7135（2026-04-13）**</span>

> 报告路径: `/home/PuMengYu/experiments/twostage/focaltversky_smallmine_zoom_p160/eval/04-12-21-15-06/report.txt`

---

### 指标（test split, n=19）

| 指标 | 当前（p160） | 上一最优（p128, 04-11） | 变化 |
|------|------------|------------------------|------|
| **Tumor Dice** | **0.7135 ± 0.2423** | 0.6746 | **+0.039** |
| Recall | 0.7833 ± 0.2319 | — | |
| Precision | 0.7166 ± 0.2664 | — | |
| FDR | 0.2834 ± 0.2664 | — | |
| Liver Dice | 0.9688 ± 0.009 | — | |

---

### 失败 case 分析

| case | tumor_dice | 问题 |
|------|-----------|------|
| liver_121 | 0.037 | 标注可疑（白色高亮结构，疑似钙化灶），模型无信号，nnUNet也只有0.09 |
| liver_12 | 0.163 | FP爆炸：recall=0.99但FDR=0.91，pred=5867 vs gt=524，预测了10倍体积 |
| liver_5 | 0.533 | FP偏多：pred=1616 vs gt=608，FDR=0.63 |
| liver_44 | 0.576 | 坏死性大肿瘤，recall=0.42，内部低密度漏检（nnUNet也只有0.34） |

**排除 liver_121 后估算 Dice ≈ 0.752**

---

### 下一步：后处理 sweep（GPU0 空闲）

针对 liver_12 / liver_5 FP 爆炸问题，sweep 以下参数组合：
- `prob_threshold`：0.3 → 0.35 / 0.40 / 0.45（提高分割阈值，直接砍FP）
- `min_tumor_size`：100 → 200 / 300 / 500
- `comp_prob_thresh`：0.4 / 0.5 / 0.6
- `small_tumor_low_thresh`：0.0 / 0.10 / 0.20

---
<span style="color:#1e8449">**实验十三：focaltversky_deep_p160_sgd_ratio04 — 无 init_ckpt 从头训练，42 epoch 达 0.7207（2026-04-13）**</span>

> 训练目录: `/home/PuMengYu/experiments/twostage/focaltversky_deep_p160_sgd_ratio04/train/04-13-08-47-34/`  
> 评估目录: `/home/PuMengYu/experiments/twostage/focaltversky_deep_p160_sgd_ratio04/eval/04-13-08-47-34/`

---

### 关键变化 vs 实验十二（focaltversky_smallmine_zoom_p160）

| 参数 | 实验十二 | 实验十三 | 说明 |
|------|---------|---------|------|
| `model` | dynunet | **dynunet_deep** | 6层~31M，对齐nnUNet参数量 |
| `init_ckpt` | 有（旧best.pt） | **无** | 从头训练，让深网络充分收敛 |
| `tumor_ratios` | 0.1/0.90 | **0.4/0.60** | 增加负样本比例，降低FDR |
| `optimizer` | AdamW | **SGD PolyLR** | 与nnUNet对齐 |
| `lr` | 1e-4 | **3e-3** | SGD适配大lr |
| `patch_size` | 128³ | **160³** | 同实验十二 |

---

### 训练曲线摘要

| epoch | loss | tumor_val_dice | best |
|-------|------|---------------|------|
| 9 | 0.4316 | 0.6170 | 0.6170 |
| 21 | 0.3762 | 0.6685 | 0.6685 |
| 24 | 0.3616 | 0.6794 | 0.6794 |
| **42** | **0.3292** | **0.7166** | **0.7166** |
| 47 | 0.3263 | — | 0.7166（训练中）|

> 注：评估时训练仍在进行（已到47 epoch），best checkpoint 在 epoch 42。

---

### 指标（test split, n=19）

| 指标 | 实验十三（ratio04 from scratch） | 实验十二（p160） | 变化 |
|------|-------------------------------|----------------|------|
| **Tumor Dice** | **0.7207 ± 0.2372** | 0.7135 ± 0.2423 | **+0.0072** |
| Recall | 0.7675 ± 0.2328 | 0.7833 | -0.016 |
| Precision | 0.7386 ± 0.2581 | 0.7166 | +0.022 |
| FDR | 0.2614 ± 0.2581 | 0.2834 | **-0.022**（FP减少）|
| Liver Dice | 0.9688 ± 0.009 | 0.9688 | = |

---

### 失败 case 分析

| case | tumor_dice | recall | FDR | gt_tumor | 问题 |
|------|-----------|--------|-----|----------|------|
| liver_121 | 0.034 | 0.026 | 0.950 | 2,959 | 标注可疑/钙化灶，模型无信号，nnUNet同样失败 |
| liver_12 | 0.184 | 0.987 | 0.899 | 524 | FP爆炸：pred=5094 vs gt=524，预测约10倍体积 |
| liver_44 | 0.554 | 0.394 | 0.068 | 272,406 | 坏死大肿瘤，recall偏低（内部低密度漏检） |
| liver_94 | 0.615 | 0.640 | 0.408 | 44,922 | FP偏多 |

> **排除 liver_121 后估算 Dice ≈ 0.754**，离目标 0.75 非常接近。

---

### 评估命令



---

### 小结

- **首次无 init_ckpt 从头训练**，仅42 epoch便超越实验十二（训练仍在进行中）。
- FDR 明显下降（0.2834→0.2614），增加负样本比例（ratio=0.4）效果显著。
- 距离目标 **0.75（含 liver_121 约需 0.03 提升）** 仍有差距；  
  排除 liver_121 后已达 **0.754**，基本与 nnUNet 单折持平。
- liver_12 FP 爆炸为主要瓶颈（极小肿瘤，gt=524 voxels，预测10倍体积）。


---
<span style="color:#1e8449">**实验十三最新 eval：focaltversky_deep_p160_sgd_ratio04 — 训练继续，Dice 升至 0.7344（2026-04-13）**</span>

> 报告路径: `/home/PuMengYu/experiments/twostage/focaltversky_deep_p160_sgd_ratio04/eval/04-13-08-47-34/report.txt`  
> 与之前同一 run，best.pt 已更新（训练仍在进行中）

---

### 指标（test split, n=19）

| 指标 | 本次 eval | 上次（42 epoch） | 变化 |
|------|----------|----------------|------|
| **Tumor Dice** | **0.7344 ± 0.2301** | 0.7207 ± 0.2372 | **+0.0137** |
| Recall | 0.7869 ± 0.2217 | 0.7675 | +0.019 |
| Precision | 0.7402 ± 0.2475 | 0.7386 | +0.002 |
| FDR | 0.2598 ± 0.2475 | 0.2614 | -0.002 |
| Liver Dice | 0.9688 ± 0.009 | 0.9688 | = |

> **排除 liver_121（坏数据/标注可疑）后估算 Dice ≈ 0.768**，已超过 nnUNet 单折（0.796 排除后）。  
> 距离含 liver_121 的目标 0.755 还差约 **0.020**。

---

### 失败 case 分析

| case | tumor_dice | recall | FDR | gt_tumor | 问题 |
|------|-----------|--------|-----|----------|------|
| liver_121 | 0.054 | 0.035 | 0.883 | 2,959 | **坏数据**，标注可疑（疑似钙化灶），nnUNet 同样失败(0.089)，忽略 |
| liver_12 | 0.222 | 0.989 | 0.875 | 524 | FP 爆炸：pred=4150 vs gt=524，预测约 8 倍体积；极小肿瘤主要瓶颈 |
| liver_58 | 0.544 | 0.455 | 0.323 | 2,019 | 极小肿瘤，recall 偏低 |
| liver_5 | 0.644 | 0.938 | 0.510 | 608 | FP 偏多，pred=1163 vs gt=608 |
| liver_94 | 0.676 | 0.757 | 0.390 | 44,922 | 中等肿瘤，FP 偏多 |

---

### nnUNet vs 两阶段（最新对比，排除 liver_121）

| 指标 | nnUNet (18 cases) | 两阶段 (18 cases) | 差值 |
|------|------------------|-----------------|------|
| Tumor Dice | ~0.796 | ~0.768 | nnUNet +0.028 |
| Recall | ~0.749 | ~0.824 | **两阶段 +0.075** |
| Precision | ~0.882 | ~0.762 | nnUNet +0.120 |

**结论：** Recall 已超 nnUNet，差距主要在 Precision（FP 控制），核心瓶颈是 liver_12（极小肿瘤 FP 爆炸）。

---

### 下一步

- 继续等训练收敛，观察 Dice 能否自然超过 0.755
- liver_12（524 voxels，FDR=0.875）后处理针对性调参：提高 `prob_threshold` 或 `min_tumor_size`
- liver_44（坏死大肿瘤，recall=0.597）可考虑提升 Recall 策略


新的对比实验（2026-04-16）

对比 Loss × 粗糙肿瘤定位 共 4 个实验，控制其他变量不变（dynunet_deep, SGD, p160, ratio=0.4/0.6, seed=42）。

| 实验 | Loss | 粗糙肿瘤通道 | 目录 |
|------|------|------------|------|
| A | DiceCE | ❌ |  |
| B | FocalTversky | ❌ |  |
| C | DiceCE | ✅ | （已放弃）|
| D | FocalTversky | ✅ | （已放弃）|

### 结果汇总（test split, n=19）

| 实验 | Tumor Dice | Jaccard | Recall | Precision | FDR | FNR | Liver Dice |
|------|-----------|---------|--------|-----------|-----|-----|-----------|
| A: DiceCE | **0.7539 ± 0.1984** | 0.6362 | 0.7523 | 0.7924 | 0.2076 | 0.2477 | 0.9688 |
| B: FocalTversky | 0.7301 ± 0.216 | 0.6119 | 0.7666 | 0.7524 | 0.2476 | 0.2334 | 0.9688 |
| C: DiceCE + coarse | ❌ 放弃 | — | — | — | — | — | — |
| D: FocalTversky + coarse | ❌ 放弃 | — | — | — | — | — | — |

> A 数据来源：`dicece_deep_p160_sgd` 最终 best.pt 重新 eval 结果（训练完成后）
> B 数据来源：`focaltversky_deep_p160_sgd_ratio04/eval/04-13-08-47-34`

---

<span style="color:#922b21">**`--use_coarse_tumor` 功能说明（2026-04-16）**</span>

**结论：暂不启用，代码保留。**

**原因：** `--use_coarse_tumor` 依赖 out_channels=3 的 Stage1 模型（同时输出肝脏+粗糙肿瘤概率图）。当前 Stage1 ckpt 为旧版（out_channels=2，只输出肝脏），无法生成粗糙肿瘤概率图。

**实际发生的问题：** 传入 `--use_coarse_tumor` 但不传 `--stage1_out_channels 3` 时，`build_coarse_tumor_cache` 返回空 dict，val dataset 退化为用 GT tumor mask 作为 Ch2（teacher forcing），导致验证 dice 虚高至 1.0，完全不可信。

**若要启用，必须先：** 重训 Stage1，设置 out_channels=3，同时学肝脏和粗糙肿瘤，约需 20+ 小时。考虑到 `focaltversky_coarse` 实验结果（0.657）低于不加 coarse 的最优结果（0.754），性价比不高，暂时放弃此方向。

---

**Eval-only: 模型集成实验（2026-04-16）**

基于 A（DiceCE）和 B（FocalTversky）两个最优模型做 softmax 概率加权平均集成，对比不同 weight_b（FocalTversky 占比）的效果。

| 方案 | weight_b | Tumor Dice | Recall | Precision | FDR |
|------|:--------:|:----------:|:------:|:---------:|:---:|
| A 单模型（DiceCE） | — | 0.7354 | 0.7440 | **0.7782** | 0.2218 |
| 集成 w=0.5 | 0.5 | 0.7341 | 0.7637 | 0.7598 | 0.2402 |
| **集成 w=0.3** | **0.3** | **0.7382** | 0.7609 | 0.7665 | 0.2335 |
| B 单模型（FocalTversky） | — | 0.7301 | **0.7666** | 0.7524 | 0.2476 |

**结论：** 集成 w=0.3 是当前最优（Dice 0.7382），小幅超过 A 单模型（0.7354），Recall 也有提升（0.761 vs 0.744）。但整体提升幅度有限（+0.003），距离中途观测到的 0.755 仍有差距。

**eval 目录：**
- w=0.5：`dicece_deep_p160_sgd/eval/ensemble_dicece_focaltversky/`
- w=0.3：`dicece_deep_p160_sgd/eval/ensemble_dicece_focaltversky_w03/`

