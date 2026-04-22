"""
评估指标计算模块
与VLM评估保持一致的指标计算
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
    
    Returns:
        包含所有指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 每类准确率
    # Clean (0) 的准确率 = TN / (TN + FP)
    # Dirty (1) 的准确率 = TP / (TP + FN)
    clean_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    dirty_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "per_class_accuracy": {
            "clean": float(clean_acc),
            "dirty": float(dirty_acc)
        },
        "total_samples": len(y_true),
        "class_distribution": {
            "clean": int(np.sum(y_true == 0)),
            "dirty": int(np.sum(y_true == 1))
        }
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str = "评估结果"):
    """
    打印格式化的评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    
    # 样本统计
    print("\n样本统计:")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  Clean样本: {metrics['class_distribution']['clean']}")
    print(f"  Dirty样本: {metrics['class_distribution']['dirty']}")
    
    # 主要指标
    print("\n主要指标:")
    print(f"  准确率 (Accuracy):        {metrics['accuracy']*100:.2f}%")
    print(f"  精确率 (Precision):       {metrics['precision']*100:.2f}%")
    print(f"  召回率 (Recall):          {metrics['recall']*100:.2f}%")
    print(f"  F1分数 (F1-Score):        {metrics['f1_score']*100:.2f}%")
    print(f"  平衡准确率:                {metrics['balanced_accuracy']*100:.2f}%")
    
    # 每类准确率
    print("\n每类准确率:")
    print(f"  Clean: {metrics['per_class_accuracy']['clean']*100:.2f}%")
    print(f"  Dirty: {metrics['per_class_accuracy']['dirty']*100:.2f}%")
    
    # 混淆矩阵
    cm = metrics['confusion_matrix']
    print("\n混淆矩阵:")
    print(f"                预测Clean  预测Dirty")
    print(f"  实际Clean:    {cm['true_negative']:6d}    {cm['false_positive']:6d}")
    print(f"  实际Dirty:    {cm['false_negative']:6d}    {cm['true_positive']:6d}")
    
    print("=" * 80)


def compare_metrics(metrics_list: List[Dict], names: List[str]):
    """
    对比多个模型的指标
    
    Args:
        metrics_list: 指标字典列表
        names: 模型名称列表
    """
    print("=" * 100)
    print("模型性能对比")
    print("=" * 100)
    
    # 表头
    header = f"{'模型':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'平衡准确率':>12}"
    print(header)
    print("-" * 100)
    
    # 每个模型的结果
    for name, metrics in zip(names, metrics_list):
        row = (f"{name:<20} "
               f"{metrics['accuracy']*100:>9.2f}% "
               f"{metrics['precision']*100:>9.2f}% "
               f"{metrics['recall']*100:>9.2f}% "
               f"{metrics['f1_score']*100:>9.2f}% "
               f"{metrics['balanced_accuracy']*100:>11.2f}%")
        print(row)
    
    print("=" * 100)


class MetricsTracker:
    """训练过程中的指标跟踪器"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(self, epoch: int, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """更新训练历史"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        
        # 更新最佳结果
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def get_history(self) -> Dict:
        """获取完整历史"""
        return self.history
    
    def get_best_info(self) -> Dict:
        """获取最佳结果信息"""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }


if __name__ == "__main__":
    # 测试指标计算
    print("测试指标计算模块...\n")
    
    # 模拟预测结果
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 打印结果
    print_metrics(metrics, "测试示例")
    
    # 测试对比功能
    print("\n" + "=" * 100)
    metrics_list = [metrics, metrics]
    names = ["模型A", "模型B"]
    compare_metrics(metrics_list, names)
