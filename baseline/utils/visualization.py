"""
Visualization Module
Plot training curves, confusion matrices, etc.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from typing import Dict, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DPI, PLOT_FORMAT

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Register Times New Roman fonts
FONT_BOLD_PATH = os.path.join(SCRIPT_DIR, 'Times New Roman', 'times-new-roman-bold', 
                              'Times New Roman Bold', 'Times New Roman Bold.ttf')
FONT_REGULAR_PATH = os.path.join(SCRIPT_DIR, 'Times New Roman', 'times-new-roman-regular',
                                 'Times New Roman Regular', 'Times New Roman Regular.ttf')

# Add fonts to matplotlib
font_prop_bold = fm.FontProperties(fname=FONT_BOLD_PATH)
font_prop_regular = fm.FontProperties(fname=FONT_REGULAR_PATH)

# Set global font configuration
plt.rcParams['font.family'] = font_prop_regular.get_name()
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.title_fontsize'] = 12


def plot_training_curves(history: Dict, save_path: str = None, model_name: str = "Model"):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary containing train_loss, train_acc, val_loss, val_acc
        save_path: Save path (if None, display the plot)
        model_name: Model name
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', 
                label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', 
                label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_learning_rate(history: Dict, save_path: str = None, model_name: str = "Model"):
    """
    Plot learning rate schedule
    
    Args:
        history: Dictionary containing learning_rate
        save_path: Save path
        model_name: Model name
    """
    epochs = range(1, len(history['learning_rate']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'{model_name} - Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Learning rate curve saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix: Dict, save_path: str = None, 
                         model_name: str = "Model", class_names: List[str] = None):
    """
    Plot confusion matrix heatmap
    
    Args:
        confusion_matrix: Confusion matrix dictionary {tn, fp, fn, tp}
        save_path: Save path
        model_name: Model name
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Clean', 'Dirty']
    
    # Build matrix
    cm = np.array([
        [confusion_matrix['true_negative'], confusion_matrix['false_positive']],
        [confusion_matrix['false_negative'], confusion_matrix['true_positive']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict], save_path: str = None):
    """
    Plot metrics comparison bar chart for multiple models
    
    Args:
        metrics_dict: {model_name: metrics_dict}
        save_path: Save path
    """
    model_names = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy']
    
    # Extract data
    data = {metric: [] for metric in metric_names}
    for model_name in model_names:
        for metric in metric_names:
            data[metric].append(metrics_dict[model_name][metric] * 100)
    
    # Plot
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        offset = width * (i - 2)
        bars = ax.bar(x + offset, data[metric], width, label=label)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Comparison chart saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_radar_chart(metrics_dict: Dict[str, Dict], save_path: str = None):
    """
    Plot radar chart to compare multiple models
    
    Args:
        metrics_dict: {model_name: metrics_dict}
        save_path: Save path
    """
    model_names = list(metrics_dict.keys())
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot radar chart for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for idx, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [metrics_dict[model_name][key] * 100 for key in metric_keys]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Model Performance Radar Chart', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Radar chart saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_per_class_accuracy(metrics_dict: Dict[str, Dict], save_path: str = None):
    """
    Plot per-class accuracy comparison chart
    
    Args:
        metrics_dict: {model_name: metrics_dict}
        save_path: Save path
    """
    model_names = list(metrics_dict.keys())
    clean_accs = [metrics_dict[m]['per_class_accuracy']['clean'] * 100 for m in model_names]
    dirty_accs = [metrics_dict[m]['per_class_accuracy']['dirty'] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean', color='skyblue')
    bars2 = ax.bar(x + width/2, dirty_accs, width, label='Dirty', color='lightcoral')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy for Clean and Dirty Categories')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Per-class accuracy chart saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_tuning_heatmap(tuning_results: Dict[str, float], 
                       lr_candidates: List[float], 
                       wd_candidates: List[float], 
                       save_path: str = None, 
                       model_name: str = "Model"):
    """
    绘制超参数调优热力图 (6 LR x 2 WD)
    
    Args:
        tuning_results: 键为 "lr_wd"，值为验证集准确率的字典
        lr_candidates: 学习率候选列表
        wd_candidates: 权重衰减候选列表
        save_path: 保存路径
        model_name: 模型名称
    """
    # 转换为矩阵格式
    matrix = np.zeros((len(lr_candidates), len(wd_candidates)))
    
    for i, lr in enumerate(lr_candidates):
        for j, wd in enumerate(wd_candidates):
            key = f"{lr}_{wd}"
            matrix[i, j] = tuning_results.get(key, 0) * 100
            
    plt.figure(figsize=(10, 8))
    
    # 格式化坐标轴标签
    lr_labels = [f"{lr:.1e}" for lr in lr_candidates]
    wd_labels = [f"{wd:.1e}" for wd in wd_candidates]
    
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=wd_labels, yticklabels=lr_labels,
                cbar_kws={'label': 'Validation Accuracy (%)'})
    
    plt.xlabel('Weight Decay')
    plt.ylabel('Learning Rate')
    plt.title(f'{model_name} - Hyperparameter Tuning Landscape')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Tuning heatmap saved: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test visualization module
    print("Testing visualization module...\n")
    
    # Simulate training history
    history = {
        'train_loss': [0.6, 0.4, 0.3, 0.2, 0.15],
        'train_acc': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_loss': [0.65, 0.45, 0.35, 0.28, 0.25],
        'val_acc': [0.68, 0.78, 0.82, 0.87, 0.89],
        'learning_rate': [1e-4, 9e-5, 7e-5, 5e-5, 3e-5]
    }
    
    # Plot training curves
    plot_training_curves(history, model_name="Test Model")
    
    # Simulate confusion matrix
    cm = {
        'true_negative': 45,
        'false_positive': 5,
        'false_negative': 3,
        'true_positive': 47
    }
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, model_name="Test Model")
