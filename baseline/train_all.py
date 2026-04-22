"""
批量训练所有benchmark模型，包含超参数调优和多种子验证
"""
import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# if u are in the china mainland, this line is very important for u
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch

from config import *
from base_trainer import BaseTrainer
from utils.dataset import create_dataloaders


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tune_hyperparameters(model_name: str, train_loader, val_loader) -> Tuple[float, float, Dict]:
    """
    超参数调优阶段 (Pilot Study)
    12种组合 (6 LR x 2 WD)，低预算运行
    """
    global LEARNING_RATE, WEIGHT_DECAY, EPOCHS
    
    print(f"\n>>> 开始超参数调优: {model_name}")
    tuning_results = {}
    best_acc = -1
    
    original_lr = LEARNING_RATE
    original_wd = WEIGHT_DECAY
    original_epochs = EPOCHS
    
    best_config = (original_lr, original_wd)
    
    # 调优循环
    for lr in LR_CANDIDATES:
        for wd in WD_CANDIDATES:
            print(f"\n测试组合: LR={lr}, WD={wd}")
            LEARNING_RATE = lr
            WEIGHT_DECAY = wd
            EPOCHS = TUNE_EPOCHS # 使用低预算
            
            # 这里的种子固定为 RANDOM_SEED 以便公平对比配置
            seed_everything(RANDOM_SEED)
            
            # 显式传递当前循环的超参数
            trainer = BaseTrainer(
                f"tuning/{model_name}_lr{lr}_wd{wd}", 
                MODELS[model_name], 
                save_checkpoints=False,
                learning_rate=lr,
                weight_decay=wd
            )
            # 仅运行训练和验证
            trainer.train(train_loader, val_loader)
            
            val_acc = trainer.metrics_tracker.best_val_acc
            tuning_results[f"{lr}_{wd}"] = val_acc
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_config = (lr, wd)
    
    # 恢复原配置（除被选中的最佳参数外）
    EPOCHS = original_epochs
    LEARNING_RATE = original_lr 
    WEIGHT_DECAY = original_wd
    
    print(f"\n调优完成! 最佳配置: LR={best_config[0]}, WD={best_config[1]}, 验证集Acc={best_acc*100:.2f}%")
    
    return best_config[0], best_config[1], tuning_results

def train_all_models(models_to_train=None, data_dir=None):
    global LEARNING_RATE, WEIGHT_DECAY
    
    if models_to_train is None:
        models_to_train = list(MODELS.keys())
    
    print("="*100)
    print(f"Benchmark 强化训练任务 (调优 + 多种子)")
    print("="*100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待处理模型总数: {len(models_to_train)}")
    print(f"模型列表: {', '.join(models_to_train)}")
    if data_dir:
        print(f"数据集路径: {data_dir}")
    print("="*100)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(data_dir=data_dir)
    
    summary_results = {}
    
    for idx, model_name in enumerate(models_to_train, 1):
        # 声明全局变量以允许修改
        global LEARNING_RATE, WEIGHT_DECAY

        print("\n" + "="*100)
        print(f"[{idx}/{len(models_to_train)}] 模型流程: {model_name}")
        print("="*100)
        
        try:
            # 1. 调优
            best_lr, best_wd, tuning_data = tune_hyperparameters(model_name, train_loader, val_loader)
            
            # 2. 多种子验证
            seed_metrics = []
            
            # 设置为本次模型的最佳参数
            LEARNING_RATE = best_lr
            WEIGHT_DECAY = best_wd
            
            for seed in RANDOM_SEEDS:
                print(f"\n--- 种子训练: Seed={seed} ---")
                seed_everything(seed)
                
                # 显式传递调优得到的最佳参数
                trainer = BaseTrainer(
                    f"{model_name}/seed_{seed}", 
                    MODELS[model_name],
                    learning_rate=best_lr,
                    weight_decay=best_wd
                )
                test_metrics = trainer.run(train_loader, val_loader, test_loader)
                
                if test_metrics:
                    seed_metrics.append(test_metrics)
            
            # 3. 统计汇总
            if seed_metrics:
                # 计算均值和标准差
                metrics_to_avg = ['accuracy', 'precision', 'recall', 'f1_score']
                final_results = {'mean': {}, 'std': {}, 'best_params': {'lr': best_lr, 'wd': best_wd}}
                
                for m in metrics_to_avg:
                    vals = [res[m] for res in seed_metrics]
                    final_results['mean'][m] = float(np.mean(vals))
                    final_results['std'][m] = float(np.std(vals))
                
                summary_results[model_name] = final_results
                
                print(f"\n✓ {model_name} 全部完成! 平均准确率: {final_results['mean']['accuracy']*100:.2f}% ± {final_results['std']['accuracy']*100:.2f}%")
                
        except Exception as e:
            print(f"\n✗ {model_name} 出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印最终大表
    if summary_results:
        print("\n" + "="*100)
        print("所有实验完成 - 最终汇总 (Mean ± Std)")
        print("="*100)
        print(f"{'模型':<20} {'最佳LR':<10} {'准确率 %':<20} {'F1分数 %':<20}")
        print("-" * 100)
        for name, res in summary_results.items():
            m = res['mean']
            s = res['std']
            print(f"{name:<20} {res['best_params']['lr']:<10.1e} "
                  f"{m['accuracy']*100:>5.2f} ± {s['accuracy']*100:>4.2f}  "
                  f"{m['f1_score']*100:>5.2f} ± {s['f1_score']*100:>4.2f}")

def main():
    parser = argparse.ArgumentParser(description='批量训练benchmark模型')
    parser.add_argument('--models', type=str, nargs='+', default=None, help='要训练的模型列表')
    parser.add_argument('--data_dir', type=str, default=None, help='数据集根目录路径')
    args = parser.parse_args()
    train_all_models(args.models, args.data_dir)

if __name__ == "__main__":
    main()