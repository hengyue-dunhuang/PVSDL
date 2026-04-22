"""
基础训练器类
所有模型训练脚本的基类
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from config import *
from utils.metrics import calculate_metrics, print_metrics, MetricsTracker
from utils.visualization import (
    plot_training_curves,
    plot_learning_rate,
    plot_confusion_matrix
)


class BaseTrainer:
    """
    基础训练器类
    封装训练、验证、测试的完整流程
    """
    
    def __init__(self, model_name: str, model_config: Dict, save_checkpoints: bool = True, 
                 learning_rate: Optional[float] = None, weight_decay: Optional[float] = None):
        """
        Args:
            model_name: 模型名称（用于保存结果）
            model_config: 模型配置字典
            save_checkpoints: 是否保存检查点
            learning_rate: 学习率 (如果为None则使用config.py中的默认值)
            weight_decay: 权重衰减 (如果为None则使用config.py中的默认值)
        """
        self.model_name = model_name
        self.model_config = model_config
        self.device = torch.device(DEVICE)
        self.save_checkpoints_flag = save_checkpoints
        self.learning_rate = learning_rate if learning_rate is not None else LEARNING_RATE
        self.weight_decay = weight_decay if weight_decay is not None else WEIGHT_DECAY
        
        # 创建结果目录
        self.result_dir = os.path.join(RESULTS_DIR, model_name)
        self.checkpoint_dir = os.path.join(self.result_dir, "checkpoints")
        self.log_dir = os.path.join(self.result_dir, "logs")
        self.curve_dir = os.path.join(self.result_dir, "curves")
        
        # 使用 exist_ok=True 和 makedirs 来递归创建父目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.curve_dir, exist_ok=True)
        
        # 初始化模型
        print(f"\n{'='*80}")
        print(f"初始化模型: {model_name}")
        print(f"描述: {model_config['description']}")
        print(f"{'='*80}")
        
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n模型参数:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 指标追踪器
        self.metrics_tracker = MetricsTracker()
        
        # 早停计数器
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # 训练时间记录
        self.start_time = None
        self.total_train_time = 0
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        try:
            model = timm.create_model(
                self.model_config['name'],
                pretrained=self.model_config['pretrained'],
                num_classes=self.model_config['num_classes']
            )
            return model
        except Exception as e:
            raise RuntimeError(f"创建模型失败: {e}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if OPTIMIZER == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif OPTIMIZER == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif OPTIMIZER == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {OPTIMIZER}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if SCHEDULER == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_MAX
            )
        elif SCHEDULER == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif SCHEDULER == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % PRINT_FREQ == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """验证"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self, test_loader: DataLoader) -> Dict:
        """测试并计算详细指标"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for images, labels, paths in test_loader:
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_paths.extend(paths)
        
        # 计算指标
        metrics = calculate_metrics(all_labels, all_preds)
        
        # 添加预测详情
        metrics['predictions'] = [
            {
                'path': path,
                'true_label': true,
                'pred_label': pred
            }
            for path, true, pred in zip(all_paths, all_labels, all_preds)
        ]
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        if not self.save_checkpoints_flag:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics_tracker': self.metrics_tracker.get_history(),
            'best_info': self.metrics_tracker.get_best_info()
        }
        
        # 保存最新的检查点
        latest_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，也保存为best.pth
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最佳模型: epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"检查点已加载: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"\n{'='*80}")
        print(f"开始训练: {self.model_name}")
        print(f"{'='*80}")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"总轮数: {EPOCHS}")
        print(f"批次大小: {BATCH_SIZE}")
        print(f"学习率: {LEARNING_RATE}")
        print(f"设备: {self.device}")
        print(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            
            print(f"Epoch [{epoch}/{EPOCHS}]")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 更新指标追踪器
            self.metrics_tracker.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  用时: {epoch_time:.1f}s\n")
            
            # 保存检查点
            is_best = val_acc > self.metrics_tracker.best_val_acc
            if (epoch % SAVE_FREQ == 0 or is_best) and self.save_checkpoints_flag:
                self.save_checkpoint(epoch, is_best)
            
        self.total_train_time = time.time() - self.start_time
        print(f"\n训练完成!")
        print(f"总用时: {self.total_train_time/60:.1f}分钟")
        print(f"最佳验证准确率: {self.metrics_tracker.best_val_acc*100:.2f}% (Epoch {self.metrics_tracker.best_epoch})")
    
    def save_results(self, test_metrics: Dict):
        """保存所有结果"""
        if not self.save_checkpoints_flag:
            return
            
        # 保存训练历史
        history_path = os.path.join(self.log_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_tracker.get_history(), f, indent=2)
        print(f"训练历史已保存: {history_path}")
        
        # 保存测试结果
        test_result = {
            'model_name': self.model_name,
            'model_config': self.model_config,
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': self.total_train_time,
            'best_epoch': self.metrics_tracker.best_epoch,
            'best_val_acc': self.metrics_tracker.best_val_acc,
            'test_metrics': test_metrics
        }
        
        test_path = os.path.join(self.result_dir, "test_results.json")
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        print(f"测试结果已保存: {test_path}")
        
        # 绘制训练曲线
        curve_path = os.path.join(self.curve_dir, "training_curves.png")
        plot_training_curves(
            self.metrics_tracker.get_history(),
            save_path=curve_path,
            model_name=self.model_name
        )
        
        # 绘制学习率曲线
        lr_curve_path = os.path.join(self.curve_dir, "learning_rate.png")
        plot_learning_rate(
            self.metrics_tracker.get_history(),
            save_path=lr_curve_path,
            model_name=self.model_name
        )
        
        # 绘制混淆矩阵
        cm_path = os.path.join(self.curve_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            save_path=cm_path,
            model_name=self.model_name
        )
        
        print(f"所有结果已保存到: {self.result_dir}")
    
    def run(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """运行完整的训练和测试流程"""
        try:
            # 训练
            self.train(train_loader, val_loader)
            
            # 加载最佳模型
            if self.save_checkpoints_flag:
                best_model_path = os.path.join(self.checkpoint_dir, "best.pth")
                if os.path.exists(best_model_path):
                    self.load_checkpoint(best_model_path)
            
            # 测试
            print(f"\n{'='*80}")
            print(f"在测试集上评估最佳模型...")
            print(f"{'='*80}\n")
            
            test_metrics = self.test(test_loader)
            print_metrics(test_metrics, f"{self.model_name} - 测试集结果")
            
            # 保存结果
            self.save_results(test_metrics)
            
            return test_metrics
            
        except KeyboardInterrupt:
            print("\n训练被用户中断!")
            return None
        except Exception as e:
            print(f"\n训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 测试trainer
    print("测试BaseTrainer类...\n")
    
    from utils.dataset import create_dataloaders
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # 创建trainer并测试
    trainer = BaseTrainer("test_model", MODELS["mobilenetv2_100"])
    
    print("\nBaseTrainer初始化成功!")
