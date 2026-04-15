# twostage_medseg 项目说明

两阶段肝脏+肿瘤分割框架：
- Stage 1（在 medseg_project 里训练）：全CT → 分割肝脏（或肝脏+肿瘤三分类），给出肝脏 bbox
- Stage 2（本仓库）：按 Stage1 的 bbox 裁出肝脏 ROI → 专门做肿瘤二分类分割

数据集：Task03_Liver（MSD肝脏肿瘤分割任务，label: 0=背景 1=肝脏 2=肿瘤）

---

## 顶层文件

- `read.md` — 本文件，项目说明
- `setup.py` — 安装包入口，让 `twostage_medseg` 可以 `import`
- `__init__.py` — 包初始化
- `configs/default.yaml` — 默认训练超参数（模型、patch size、loss、margin、hard mining 阈值、stage1 ckpt路径等）
- `experiments_log.ipynb` — 实验记录 notebook，记录每次实验的配置和结果对比
- `无肿瘤数量.py` — 统计数据集中无肿瘤 case 的数量，用于确定 hard mining 比例

---

## twostage/ — 核心模块

### `twostage/dataset_tumor_roi.py` ★ 最核心
Stage 2 的 PyTorch Dataset。每次 `__getitem__` 在线流程：
1. 加载完整 CT 的 `.pt` 文件
2. 用 Stage1 预测 bbox（或 GT bbox + jitter）裁出肝脏 ROI
3. 将 label 重映射为肿瘤二分类（tumor=2→1，其余→0）
4. 支持多通道输入：`use_coarse_tumor=True` 时输入为 [CT, Stage1粗糙肿瘤softmax概率]（两通道）
5. 差异化 oversampling（hard mining）：小肿瘤/无肿瘤 case 在索引列表中出现更多次，被更频繁采样
6. 小肿瘤 zoom-in：对体素数 < 阈值的小肿瘤 case，以肿瘤为中心裁出更小区域再插值放大，等价于放大视野

关键参数：
- `pred_bboxes` — Stage1 预测的 bbox 字典（新格式含 liver+tumor 两个bbox）
- `bbox_jitter` / `bbox_max_shift` — 对bbox加随机扰动，模拟 Stage1 预测误差
- `random_margin` / `margin_min/max` — 随机 margin，模拟 ROI 尺度波动
- `use_coarse_tumor` + `coarse_tumor_cache` — 训练时用 GT 做 teacher forcing，验证时用 Stage1 软概率

### `twostage/dataset_tumor_roi_0322.py`
`dataset_tumor_roi.py` 的旧版本（0322日期），保留用于对比和复现旧实验。

### `twostage/roi_utils.py`
ROI 裁剪相关工具函数：
- `compute_bbox_from_mask(mask, margin)` — 从 [D,H,W] mask 计算紧致 bbox，支持 margin 扩边
- `crop_3d(tensor, bbox)` — 按 bbox 裁剪 [D,H,W] 或 [C,D,H,W] 张量
- `paste_3d(full, patch, bbox)` — 把 patch 贴回 full volume（推理时用）
- `bbox_to_dict(bbox)` — 将 bbox tuple 转为可序列化的 dict

### `twostage/train_eval_tumor.py`
从验证结果 dict 中提取肿瘤指标：
- `tumor_metrics_from_val_result(val_result)` — 兼容 medseg.engine 的返回格式，取出 tumor_dice

### `twostage/metrics.py`
简版指标函数（twostage 专用）：
- `build_final_pred_from_liver_tumor(liver_mask, tumor_mask)` — 合并 liver+tumor → 三类 label（0/1/2）
- `summarize_metric(values)` — 计算 mean/std/min/max

### `twostage/train_logger.py`
训练日志器 `TrainLoggerTwoStage`：
- 同时支持只记录 tumor（Stage2）或同时记录 liver+tumor
- 输出 `log.csv`（机器可读）和 `log.txt`（人类可读表格）
- `log_extra()` — 记录额外键值对到 `extra_log.csv`

### `twostage/vis_utils.py`
推理可视化：
- `pick_slice_indices(mask)` — 选三个方向上前景中心的切片（无前景时取体积中心）
- `get_views(vol, idxs)` — 取 axial/coronal/sagittal 三个方向切片
- `save_case_visualization(...)` — 保存单个 case 的多列对比图（image/GT/stage1_liver/stage2_tumor/final_pred）

### `twostage/__init__.py`
包初始化。

---

## metrics/ — 评估指标工具

### `metrics/metrics_utils.py` ★
主要评估工具（eval脚本直接调用）：
- `dice_binary(pred, gt)` — 二值 Dice，双空 case 返回 nan（不参与均值统计，对齐 nnUNet 惯例）
- `compute_metrics(pred, gt)` — 计算完整指标集（Dice/Jaccard/Precision/Recall/FPR/FDR/FNR/NPV/ACC/TP/FP/FN/TN）
- `summarize_metric(xs)` — 过滤 nan 后统计 mean/std/min/max/n
- `summarize_metrics_list(metrics_list, keys)` — 批量汇总多个 case 的指标

### `metrics/binary_metrics.py`
底层二值指标函数集（更完整，含边界情况处理）：
- `binary_stats` — 返回 TP/FP/FN/TN
- `dice_binary` — 双空返回 1.0（与 metrics_utils 版本行为不同，注意区分）
- `precision_binary` / `recall_binary` / `iou_binary` / `detected_binary`
- `summarize_metric`

### `metrics/filter.py`
后处理工具：
- `filter_largest_component(mask)` — 连通域分析，只保留最大连通域，去掉噪声小连通域（liver/tumor 后处理都会用到）

### `metrics/DiagLogger.py`
训练诊断日志器 `DiagLogger`：
- `log_dataset()` — 记录数据集规模、无肿瘤比例、训练配置
- `check_data_leakage()` — 检查 train/val/test 有无重叠
- `log_label_stats()` — 抽样检查 label 值域、肿瘤体素占比、全背景 case
- `log_roi_stats()` — 检查 Stage2 ROI 的实际尺寸分布和 ROI 内肿瘤占比
- `log_epoch()` / `log_final()` — 训练过程记录
- 输出 `diag.txt`（完整日志）和 `diag_summary.txt`（精简摘要，方便贴给 AI 诊断）

### `metrics/__init__.py`
包初始化。

---

## scripts/ — 训练/评估/分析入口脚本

### `scripts/train_tumor_roi.py` ★ Stage2 主训练脚本
Stage 2 肿瘤分割训练入口，支持的功能：
- 从 Stage1 预测 bbox 缓存（json）加载 ROI 范围，消除训练/推理 domain gap
- `--use_coarse_tumor` — 启用双通道输入（CT + Stage1粗糙肿瘤softmax）
- 差异化 oversampling（hard mining）：小肿瘤/无肿瘤过采样
- 小肿瘤 zoom-in 增强
- 支持 focaltversky / dicece / dicefocal 等 loss
- 用 `LearnableWeightedLoss` 可自动学习 loss 权重
- 用 `DiagLogger` 在训练开始时做数据健康检查

### `scripts/train_tumor_roi_0322.py`
train_tumor_roi.py 的旧版本（0322），保留用于复现早期实验。

### `scripts/train_tumor_roi_v2.py`
train_tumor_roi.py 的另一版本实验（v2），探索不同训练策略。

### `scripts/eval_twostage.py` ★ 两阶段联合评估脚本
Stage1 + Stage2 串联推理并评估：
1. Stage1 滑窗推理全CT → 得到肝脏 mask（和粗糙肿瘤概率）
2. 用肝脏 mask 算 bbox，裁出 ROI
3. Stage2 在 ROI 上滑窗推理 → 得到肿瘤 mask
4. 后处理：最大连通域过滤、最小肿瘤尺寸过滤
5. 合并 liver+tumor → 三分类最终预测
6. 计算 liver Dice / tumor Dice，输出 CSV 报告
7. 可选：`--save_vis` 保存可视化图，`--tta` 开启测试时增强

### `scripts/eval_stage1_tumor_recall.py`
快速评估 Stage1 三分类模型对肿瘤的召回率：
- 直接在 ROI pt 文件上做滑窗推理（无需 Stage2）
- 统计肿瘤的 Recall/Precision/Dice，定量评估 Stage1 能找到多少肿瘤

### `scripts/infer_twostage.py`
纯推理脚本（不评估，无 GT 也可用）：
- Stage1 → Stage2 串联推理，把最终预测保存为 `.pt` 文件
- `paste_3d` 将 ROI 的肿瘤预测贴回全体积坐标系

### `scripts/plan_patch_size.py`
仿照 nnUNet 的 patch size 自动规划工具：
- 统计数据集 median shape，按比例缩放候选 patch
- 实测真实 forward+backward 峰值显存，二分搜索在显存预算内的最大 patch

### `scripts/debug_case_postprocess.py`
单 case 后处理调试脚本，用于检查后处理参数（最小连通域大小等）对结果的影响。

### `scripts/split_two_with_mintor.txt`
train/val/test 数据集划分记录文本。

### `scripts/tumor_stats_all.txt`
全数据集每个 case 肿瘤体素数的统计结果，用于确定 hard mining 阈值。

### `scripts/展示/vis_prob.py`
概率图可视化工具：
- `vis_worst_cases()` — 可视化 worst case（Dice最低的case）的软概率图，辅助错误分析

### `scripts/学习/shell和path学习.ipynb`
学习 shell 和 Python Path 用法的笔记 notebook。

---

## scripts/run/ — 当前实验运行脚本（最新）

- `运行4_deep.sh` — 实验4：用 dynunet_deep + focaltversky loss，无 coarse_tumor 通道，CUDA:1
- `运行5_coarse_tumor_prob.sh` ★ — 实验5：启用 `--use_coarse_tumor`，Stage2 输入双通道（CT + Stage1粗糙肿瘤softmax），Stage1 ckpt 换成三分类模型，CUDA:0
- `运行6_dicece.sh` — 实验6：用 dicece loss 的变体实验
- `eval5_coarse_tumor_prob.sh` — 对实验5的 checkpoint 做 eval_twostage 评估

---

## scripts/shell/ — 历史运行脚本

- `运行.sh` / `运行1.sh` / `运行2.sh` / `运行3.sh` — 早期实验脚本（历史存档）
- `运行1.sh` — 实验十完整流程说明：Stage1重训（三分类）→ Stage2训练 → eval，含详细注释
- `eval_analysis.sh` — 对多个 checkpoint 批量跑 eval + analyze
- `eva_运行.sh` — eval 快速运行脚本
- `exp10_dynunet_ca.sh` — 实验10：dynunet_ca（带 cross-attention）的训练脚本
- `sweep_postprocess.sh` / `sweep_postprocess2.sh` — 后处理参数（最小肿瘤尺寸）扫描
- `sweep_min_tumor_size.sh` — 最小肿瘤尺寸阈值扫描
- `nnunetv2_setup.sh` — nnUNetV2 环境配置脚本
- `代理.sh` / `代理坑.sh` — 代理配置脚本（踩坑记录）
- `history/train_exp1_4.sh` — 实验1~4 的早期训练脚本
- `history/eval.sh` — 早期 eval 脚本

---

## preprocess/ — 数据预处理

### `preprocess/preprocess_liver_roi.py`
一次性离线预处理脚本：把完整 CT 的 `.pt` 文件（540MB~1.9GB）按肝脏 bbox 裁成小的 ROI `.pt` 文件（~100-200MB），加速 Stage2 训练的数据加载。支持用 Stage1 预测 bbox 或 GT bbox 裁剪。

---

## data/ — 数据工具（辅助）

- `data/prepare.py` — 数据准备脚本（原始 NIfTI 转 .pt 格式）
- `data/prepare_nnunet_data.py` — 为 nnUNetV2 准备数据格式的脚本
- `data/nnunet.sh` — nnUNetV2 数据集准备命令
- `data/__init__.py` — 包初始化

---

## analysis/ — 分析工具

- `analysis/analyze_failures.py` — 读取 eval 保存的预测 pt，深度分析失败 case：肿瘤大小 vs Dice、FP/FN分析、worst case 可视化
- `analysis/analyze_tumor_dist.py` — 分析训练集每个 case 的肿瘤体积分布，辅助确定 hard mining 的阈值
- `analysis/analyze_two_mintor.py` — 分析两阶段流程中 monitor（split_two_with_monitor）的行为
- `analysis/analysis.ipynb` — 综合分析 notebook

---

## notebooks/ — 杂项笔记

- `notebooks/read.ipynb` — 阅读笔记 notebook
- `notebooks/fix_claude.sh` — 修复 Claude Code 相关问题的 shell
- `notebooks/代理故障.sh` — 代理故障排查记录
- `notebooks/记录vpn.sh` — VPN 配置记录

---

## 外部依赖

本仓库依赖同级的 `medseg_project`（Stage1 模型、通用训练引擎、数据加载），需要 `sys.path.insert` 或 `setup.py install` 后才能 import：
- `medseg.models.build_model` — 构建 dynunet / dynunet_deep / dynunet_ca 等模型
- `medseg.engine.train_eval` — 通用训练和滑窗验证循环
- `medseg.data.dataset_offline` — `.pt` 文件加载和 train/val/test split
- `medseg.utils.*` — checkpoint 保存/加载、seed、IO 工具

---

## 典型运行流程

```bash
cd /home/PuMengYu/twostage_medseg

# 实验5：Stage2 双通道训练（CT + Stage1粗糙肿瘤概率）
bash scripts/run/运行5_coarse_tumor_prob.sh

# 评估实验5
bash scripts/run/eval5_coarse_tumor_prob.sh

# 或手动 eval
CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root /home/PuMengYu/medseg_project \
  --preprocessed_root /home/PuMengYu/Task03_Liver_pt \
  --stage1_ckpt <stage1_ckpt_path> \
  --stage2_ckpt <stage2_ckpt_path> \
  --stage1_model dynunet --stage2_model dynunet_deep \
  --use_coarse_tumor \
  --split test --tta --min_tumor_size 100
```
