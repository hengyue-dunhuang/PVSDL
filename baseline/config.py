"""
Benchmark配置文件
包含训练超参数和路径设置
"""

import os

# ==================== 路径配置 ====================
# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集路径
DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "archive", "solar_panel_dust_segmentation", "images")

# Benchmark结果路径
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")
SPLIT_FILE = os.path.join(BENCHMARK_DIR, "dataset_split.json")

# ==================== 数据集配置 ====================
# 数据集划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子（确保可复现）
RANDOM_SEED = 42

# 图像大小
IMG_SIZE = 224

# ImageNet归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==================== 训练配置 ====================
# GPU训练可以使用更大的batch size（根据显存调整，建议32-128）
# 如果遇到CUDA out of memory错误，可以减小此值
BATCH_SIZE = 64

# 学习率
LEARNING_RATE = 1e-4

# 训练轮数
EPOCHS = 30  # 统一训练轮数
TUNE_EPOCHS = 5  # 每个超参数组合调优的轮数

# 优化器
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-4

# ==================== 超参数调优配置 ====================
# 6种学习率 x 2种权重衰减 = 12种组合
# LR_CANDIDATES = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
LR_CANDIDATES = [1e-3, 1e-4, 5e-4, 1e-5]
WD_CANDIDATES = [1e-4, 1e-5]

# 多随机种子
RANDOM_SEEDS = [42, 123 , 345]

# 学习率调度器
SCHEDULER = "CosineAnnealingLR"
T_MAX = 30  # 与EPOCHS相同

# 早停
EARLY_STOPPING_PATIENCE = 10

# 类别权重（可选，用于处理类别不平衡）
USE_CLASS_WEIGHTS = False

# ==================== 模型配置 ====================
# CPU友好的模型列表（16个模型）
MODELS = {
    # === EfficientNet系列 ===
    "efficientnet_b0": {
        "name": "efficientnet_b0",
        "pretrained": True,
        "num_classes": 2,
        "description": "EfficientNet-B0 (5.3M params) - 轻量级"
    },
    "efficientnet_b1": {
        "name": "efficientnet_b1",
        "pretrained": True,
        "num_classes": 2,
        "description": "EfficientNet-B1 (7.8M params) - 增强版"
    },
    "efficientnet_b2": {
        "name": "efficientnet_b2",
        "pretrained": True,
        "num_classes": 2,
        "description": "EfficientNet-B2 (9.1M params) - 平衡型"
    },
    "efficientnet_b3": {
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 2,
        "description": "EfficientNet-B3 (12.2M params) - 高性能"
    },
    
    # === DenseNet系列 ===
    "densenet121": {
        "name": "densenet121",
        "pretrained": True,
        "num_classes": 2,
        "description": "DenseNet121 (8M params) - 经典深度"
    },
    "densenet169": {
        "name": "densenet169",
        "pretrained": True,
        "num_classes": 2,
        "description": "DenseNet169 (14.3M params) - 深层次结构"
    },
    "densenet201": {
        "name": "densenet201",
        "pretrained": True,
        "num_classes": 2,
        "description": "DenseNet201 (20M params) - 高性能版本"
    },
    
    # === ResNet系列 ===
    "resnet18": {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "description": "ResNet18 (11.7M params) - 轻量级入门"
    },
    "resnet34": {
        "name": "resnet34",
        "pretrained": True,
        "num_classes": 2,
        "description": "ResNet34 (21.8M params) - 中等规模"
    },
    "resnet50": {
        "name": "resnet50",
        "pretrained": True,
        "num_classes": 2,
        "description": "ResNet50 (25.6M params) - 经典骨干网络"
    },
    "resnet101": {
        "name": "resnet101",
        "pretrained": True,
        "num_classes": 2,
        "description": "ResNet101 (44.5M params) - 更深更强"
    },
    
    # === MobileNet系列 ===
    "mobilenetv2_100": {
        "name": "mobilenetv2_100",
        "pretrained": True,
        "num_classes": 2,
        "description": "MobileNetV2 (3.5M params) - 通用轻量"
    },
    "mobilenetv3_small_100": {
        "name": "mobilenetv3_small_100",
        "pretrained": True,
        "num_classes": 2,
        "description": "MobileNetV3 Small (2.5M params) - 超轻量"
    },
    "mobilenetv3_large_100": {
        "name": "mobilenetv3_large_100",
        "pretrained": True,
        "num_classes": 2,
        "description": "MobileNetV3 Large (5.4M params) - 高效平衡"
    },
    
    # === 其他现代架构 ===
    "vit_small_patch16_224": {
        "name": "vit_small_patch16_224",
        "pretrained": True,
        "num_classes": 2,
        "description": "ViT Small (22M params) - Vision Transformer"
    }
}

# ==================== 数据增强配置 ====================
# 训练集数据增强
TRAIN_AUGMENTATION = {
    "random_horizontal_flip": 0.5,
    "random_rotation": 10,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }
}

# ==================== 其他配置 ====================
# 保存检查点的频率（每N个epoch）
SAVE_FREQ = 5

# 打印频率（每N个batch）
PRINT_FREQ = 10

# 工作进程数（GPU训练建议设置为4-8）
NUM_WORKERS = 4

# 设备配置（自动检测CUDA，如果可用则使用GPU，否则使用CPU）
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*80}")
print(f"设备配置")
print(f"{'='*80}")
print(f"使用的设备: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*80}\n")

# ==================== 评估指标 ====================
# 与VLM保持一致的评估指标
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "balanced_accuracy",
    "confusion_matrix",
    "per_class_accuracy"
]

# 类别名称
CLASS_NAMES = ["clean", "dirty"]

# ==================== 可视化配置 ====================
# 是否保存训练曲线
SAVE_PLOTS = True

# 图表DPI
PLOT_DPI = 150

# 图表格式
PLOT_FORMAT = "png"
