"""
数据集处理模块
包含数据加载、划分和数据增强
"""

import os
import json
import random
from typing import Tuple, List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class SolarPanelDataset(Dataset):
    """光伏板灰尘检测数据集"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: 图像文件路径列表
            labels: 对应的标签列表 (0=clean, 1=dirty)
            transform: 数据转换/增强
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_image_label_from_filename(filename: str) -> int:
    """
    从文件名提取标签
    _0.jpg -> 0 (clean)
    _1.jpg -> 1 (dirty)
    """
    if filename.endswith('_0.jpg'):
        return 0
    elif filename.endswith('_1.jpg'):
        return 1
    else:
        raise ValueError(f"无法从文件名提取标签: {filename}")


def load_from_directory(directory: str) -> Tuple[List[str], List[int]]:
    """
    从指定目录加载图像和标签
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(directory):
        print(f"警告: 目录不存在 {directory}")
        return image_paths, labels

    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            try:
                label = get_image_label_from_filename(filename)
                image_paths.append(img_path)
                labels.append(label)
            except ValueError as e:
                continue
    return image_paths, labels


def load_dataset(data_dir: str = DATASET_DIR) -> Tuple[List[str], List[int]]:
    """
    加载数据集，返回图像路径和标签列表
    
    Args:
        data_dir: 数据集目录路径
        
    Returns:
        image_paths: 图像文件路径列表
        labels: 标签列表
    """
    image_paths = []
    labels = []
    
    # 遍历数据集目录
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录不存在 {data_dir}")
        return image_paths, labels

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            try:
                label = get_image_label_from_filename(filename)
                image_paths.append(img_path)
                labels.append(label)
            except ValueError as e:
                print(f"警告: {e}")
                continue
    
    print(f"加载数据集: 总计 {len(image_paths)} 张图像")
    print(f"  - Clean: {labels.count(0)} 张")
    print(f"  - Dirty: {labels.count(1)} 张")
    
    return image_paths, labels


def split_dataset(image_paths: List[str], labels: List[int], 
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO,
                  random_seed: int = RANDOM_SEED,
                  save_split: bool = True) -> Dict:
    """
    划分数据集为训练集、验证集和测试集
    
    Args:
        image_paths: 图像路径列表
        labels: 标签列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        save_split: 是否保存划分结果
    
    Returns:
        包含train/val/test划分的字典
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建索引列表
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    
    # 计算划分点
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分索引
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # 创建划分结果
    split_data = {
        'train': {
            'indices': train_indices,
            'paths': [image_paths[i] for i in train_indices],
            'labels': [labels[i] for i in train_indices]
        },
        'val': {
            'indices': val_indices,
            'paths': [image_paths[i] for i in val_indices],
            'labels': [labels[i] for i in val_indices]
        },
        'test': {
            'indices': test_indices,
            'paths': [image_paths[i] for i in test_indices],
            'labels': [labels[i] for i in test_indices]
        },
        'metadata': {
            'total_samples': n_total,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed
        }
    }
    
    # 打印统计信息
    print("\n数据集划分:")
    print(f"  训练集: {len(train_indices)} ({train_ratio*100:.0f}%)")
    print(f"    - Clean: {split_data['train']['labels'].count(0)}")
    print(f"    - Dirty: {split_data['train']['labels'].count(1)}")
    print(f"  验证集: {len(val_indices)} ({val_ratio*100:.0f}%)")
    print(f"    - Clean: {split_data['val']['labels'].count(0)}")
    print(f"    - Dirty: {split_data['val']['labels'].count(1)}")
    print(f"  测试集: {len(test_indices)} ({test_ratio*100:.0f}%)")
    print(f"    - Clean: {split_data['test']['labels'].count(0)}")
    print(f"    - Dirty: {split_data['test']['labels'].count(1)}")
    
    # 保存划分结果
    if save_split:
        os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
        with open(SPLIT_FILE, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"\n数据集划分已保存到: {SPLIT_FILE}")
    
    return split_data


def load_split() -> Dict:
    """加载已保存的数据集划分"""
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(f"未找到数据集划分文件: {SPLIT_FILE}")
    
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # 规范化路径：处理从Windows系统迁移到Linux时路径格式不兼容的问题
    def normalize_path(path: str) -> str:
        """
        将路径规范化为当前系统格式
        处理混合路径（如: /home/xxx/D:/Project/xxx/... 或冒号+空格开头的路径）的情况
        """
        import re
        
        # 检测是否包含Windows驱动器号（如 D: 或 D:\ 或 D:/）
        # 匹配模式：D: 后跟 : 或 :/ 或 :\
        win_drive_pattern = r'^([A-Za-z]):[\\/]'
        
        # 如果路径以Windows驱动器号开头，则只提取后面的相对路径部分
        match = re.match(win_drive_pattern, path)
        if match:
            drive = match.group(1)
            # 提取驱动器号之后的部分
            relative_path = path[match.end():]
            # 使用 os.path.normpath 规范化路径分隔符
            normalized = os.path.normpath(relative_path)
            return normalized
        
        # 处理冒号后面跟着空格和驱动器号的格式（如 ": D:\Project\..."）
        colon_space_drive_pattern = r'^:\s+([A-Za-z]):[/\\]'
        colon_space_match = re.match(colon_space_drive_pattern, path)
        if colon_space_match:
            # 提取第二个驱动器号之后的部分
            relative_path = path[colon_space_match.end():]
            # 使用 os.path.normpath 规范化路径分隔符
            normalized = os.path.normpath(relative_path)
            return normalized
        
        # 如果不是Windows驱动器路径，直接规范化
        return os.path.normpath(path)
    
    # 规范化训练集路径
    if 'train' in split_data and 'paths' in split_data['train']:
        split_data['train']['paths'] = [normalize_path(p) for p in split_data['train']['paths']]
    
    # 规范化验证集路径
    if 'val' in split_data and 'paths' in split_data['val']:
        split_data['val']['paths'] = [normalize_path(p) for p in split_data['val']['paths']]
    
    # 规范化测试集路径
    if 'test' in split_data and 'paths' in split_data['test']:
        split_data['test']['paths'] = [normalize_path(p) for p in split_data['test']['paths']]
    
    print(f"加载数据集划分: {SPLIT_FILE}")
    print(f"  训练集: {split_data['metadata']['train_samples']}")
    print(f"  验证集: {split_data['metadata']['val_samples']}")
    print(f"  测试集: {split_data['metadata']['test_samples']}")
    
    return split_data


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    获取数据转换
    
    Args:
        train: 是否为训练集（训练集需要数据增强）
    
    Returns:
        transforms.Compose对象
    """
    if train:
        # 训练集：数据增强
        transform_list = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=TRAIN_AUGMENTATION['random_horizontal_flip']),
            transforms.RandomRotation(degrees=TRAIN_AUGMENTATION['random_rotation']),
            transforms.ColorJitter(
                brightness=TRAIN_AUGMENTATION['color_jitter']['brightness'],
                contrast=TRAIN_AUGMENTATION['color_jitter']['contrast'],
                saturation=TRAIN_AUGMENTATION['color_jitter']['saturation'],
                hue=TRAIN_AUGMENTATION['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    else:
        # 验证集/测试集：仅resize和归一化
        transform_list = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(split_data: Dict = None, 
                       batch_size: int = BATCH_SIZE,
                       num_workers: int = NUM_WORKERS,
                       data_dir: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    优先使用项目根目录下的 dataset/train, val, test 目录
    
    Args:
        split_data: 数据集划分字典（如果为None则尝试从目录加载，失效后从文件加载）
        batch_size: 批次大小
        num_workers: 工作进程数
        data_dir: 自定义数据目录路径
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 如果指定了 data_dir，则尝试从该目录的子目录加载或直接从该目录划分
    if data_dir:
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")
        
        if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
            print(f"检测到预分好的数据集目录: {data_dir}")
            train_paths, train_labels = load_from_directory(train_dir)
            val_paths, val_labels = load_from_directory(val_dir)
            test_paths, test_labels = load_from_directory(test_dir)
            
            train_dataset = SolarPanelDataset(train_paths, train_labels, transform=get_transforms(train=True))
            val_dataset = SolarPanelDataset(val_paths, val_labels, transform=get_transforms(train=False))
            test_dataset = SolarPanelDataset(test_paths, test_labels, transform=get_transforms(train=False))
            
            return _build_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)
        else:
            print(f"从指定目录自动划分数据集: {data_dir}")
            image_paths, labels = load_dataset(data_dir)
            split_data = split_dataset(image_paths, labels)

    # 优先检测是否存在预分好的目录
    dataset_base = os.path.join(BENCHMARK_DIR, "dataset")
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "val")
    test_dir = os.path.join(dataset_base, "test")

    if split_data is None and os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        print(f"检测到预分好的数据集目录: {dataset_base}")
        train_paths, train_labels = load_from_directory(train_dir)
        val_paths, val_labels = load_from_directory(val_dir)
        test_paths, test_labels = load_from_directory(test_dir)
        
        print(f"  训练集: {len(train_paths)} 张")
        print(f"  验证集: {len(val_paths)} 张")
        print(f"  测试集: {len(test_paths)} 张")
        
        train_dataset = SolarPanelDataset(train_paths, train_labels, transform=get_transforms(train=True))
        val_dataset = SolarPanelDataset(val_paths, val_labels, transform=get_transforms(train=False))
        test_dataset = SolarPanelDataset(test_paths, test_labels, transform=get_transforms(train=False))
    else:
        # 如果没有目录，退回到 split_data 或 split_file 逻辑
        if split_data is None:
            if os.path.exists(SPLIT_FILE):
                split_data = load_split()
            else:
                print("未发现预分好目录且无划分文件，开始自动划分...")
                image_paths, labels = load_dataset()
                split_data = split_dataset(image_paths, labels)
        
        # 创建数据集
        train_dataset = SolarPanelDataset(
            split_data['train']['paths'],
            split_data['train']['labels'],
            transform=get_transforms(train=True)
        )
        
        val_dataset = SolarPanelDataset(
            split_data['val']['paths'],
            split_data['val']['labels'],
            transform=get_transforms(train=False)
        )
        
        test_dataset = SolarPanelDataset(
            split_data['test']['paths'],
            split_data['test']['labels'],
            transform=get_transforms(train=False)
        )
    
    return _build_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)


def _build_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    """辅助函数：构建DataLoader"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # GPU训练需要pin_memory来加速数据传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集加载
    print("测试数据集模块...")
    
    # 加载数据集
    image_paths, labels = load_dataset()
    
    # 划分数据集
    split_data = split_dataset(image_paths, labels)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(split_data)
    
    print(f"\n数据加载器创建成功:")
    print(f"  训练集批次数: {len(train_loader)}")
    print(f"  验证集批次数: {len(val_loader)}")
    print(f"  测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    images, labels, paths = next(iter(train_loader))
    print(f"\n测试批次:")
    print(f"  图像形状: {images.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  路径数量: {len(paths)}")
