#!/bin/bash
# nnUNetv2 全流程：数据准备 → 预处理 → 训练 → 预测
# 使用前先激活环境: conda activate nnunet
#
# 步骤说明：
#   步骤1: 把 MSD 格式数据转换成 nnUNetv2 格式（自动完成，几分钟）
#   步骤2: 提取数据指纹 + 自动规划实验（自动选patch大小/batch等，10~30分钟）
#   步骤3: 训练 fold0（约12~24小时）
#   步骤4: 预测验证集（可选，用于看 per-case 结果）

set -e  # 遇错即停

# ============================================================
# 环境变量（nnUNetv2 必须设置这三个）
# ============================================================
export nnUNet_raw="/home/pumengyu/nnUNetv2_raw"
export nnUNet_preprocessed="/home/pumengyu/nnUNetv2_preprocessed"
export nnUNet_results="/home/pumengyu/nnUNetv2_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

NNUNET_BIN="/home/pumengyu/miniconda3/envs/nnunet/bin"

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

# ============================================================
# 步骤1: MSD → nnUNetv2 格式转换
# 把 Task03_Liver（MSD格式）转成 nnUNetv2 要求的 Dataset003_Liver
# -t 3 指定 dataset ID，会创建 Dataset003_Liver 目录
# ============================================================
echo ""
echo "===== 步骤1: 转换数据格式 ====="
$NNUNET_BIN/nnUNetv2_convert_MSD_dataset \
    -i /home/pumengyu/Task03_Liver \
    -overwrite_id 3

echo "转换完成，检查:"
ls "$nnUNet_raw/Dataset003_Liver/"

# ============================================================
# 步骤2: 提取数据指纹 + 规划实验
# nnUNetv2 会自动分析数据，决定:
#   - patch size（通常 192x192x不定 或 更大）
#   - batch size
#   - 归一化方式
#   - 网络结构深度
# -d 3 = Dataset003_Liver
# -c 3d_fullres = 只规划3D全分辨率（最常用最好的配置）
# ============================================================
echo ""
echo "===== 步骤2: 数据指纹提取 + 实验规划 ====="
$NNUNET_BIN/nnUNetv2_plan_and_preprocess \
    -d 3 \
    -c 3d_fullres \
    --verify_dataset_integrity

echo "预处理完成"

# ============================================================
# 步骤3: 训练 fold 0
# fold 0 = 用 80% 数据训练，20% 验证（nnUNet自动5折，我们先跑fold0）
# --npz: 保存验证集的 softmax 预测（后续 ensemble 需要，建议加上）
# ============================================================
echo ""
echo "===== 步骤3: 训练 fold 0 ====="
echo "预计耗时 12~24 小时，建议用 nohup 后台运行"
echo ""
echo "运行命令（复制后手动执行）:"
echo "------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES=0 $NNUNET_BIN/nnUNetv2_train \\"
echo "    3d_fullres \\"
echo "    nnUNetTrainer \\"
echo "    3 \\"
echo "    0 \\"
echo "    --npz"
echo "------------------------------------------------------------"
echo ""
echo "或者后台运行:"
echo "nohup CUDA_VISIBLE_DEVICES=0 $NNUNET_BIN/nnUNetv2_train \\"
echo "    3d_fullres nnUNetTrainer 3 0 --npz \\"
echo "    > ~/nnunetv2_fold0.log 2>&1 &"
echo ""
echo "查看训练日志: tail -f ~/nnunetv2_fold0.log"
