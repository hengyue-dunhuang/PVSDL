import os
import cv2
import numpy as np
import json
import random
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import *
from utils.dataset import load_from_directory
from utils.metrics import calculate_metrics, print_metrics

# --- 特征提取函数 ---
def extract_features(image_paths: List[str]) -> np.ndarray:
    """提取图像特征：颜色直方图 + 基本统计量"""
    features_list = []
    print(f"正在为 {len(image_paths)} 张图像提取特征...")
    
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"进度: {i}/{len(image_paths)}")
            
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 1. 颜色直方图 (RGB三通道)
        hist_features = []
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [16], [0, 256])
            hist_features.extend(hist.flatten())
        
        # 2. 全局统计量
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        
        # 合并特征
        feat = np.concatenate([
            np.array(hist_features) / (IMG_SIZE * IMG_SIZE), # 归一化直方图
            mean / 255.0, 
            std / 255.0
        ])
        features_list.append(feat)
        
    return np.array(features_list)

# --- 模型定义与参数空间 ---
ML_MODELS = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "svm": {
        "class": SVC,
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "probability": [True]
        }
    },
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {
            "C": [0.1, 1, 10],
            "max_iter": [1000]
        }
    }
}

if HAS_XGB:
    ML_MODELS["xgboost"] = {
        "class": XGBClassifier,
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 6],
            "use_label_encoder": [False],
            "eval_metric": ["logloss"]
        }
    }

def tune_ml_model(name: str, model_info: Dict, X_train, y_train, X_val, y_val) -> Dict:
    """网格搜索最佳参数"""
    print(f"\n>>> 调优模型: {name}")
    best_acc = -1
    best_params = {}
    
    # 简单的网格搜索实现
    import itertools
    keys = list(model_info["params"].keys())
    values = list(model_info["params"].values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in combinations:
        model = model_info["class"](**params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        
        if score > best_acc:
            best_acc = score
            best_params = params
            
    print(f"最佳配置: {best_params}, 验证集Acc={best_acc*100:.2f}%")
    return best_params

def main():
    # 确保结果目录存在
    ML_RESULTS_DIR = os.path.join(RESULTS_DIR, "ml_algorithms")
    os.makedirs(ML_RESULTS_DIR, exist_ok=True)
    
    # 1. 加载数据
    dataset_base = os.path.join(BENCHMARK_DIR, "dataset")
    train_paths, train_labels = load_from_directory(os.path.join(dataset_base, "train"))
    val_paths, val_labels = load_from_directory(os.path.join(dataset_base, "val"))
    test_paths, test_labels = load_from_directory(os.path.join(dataset_base, "test"))
    
    # 2. 提取特征
    X_train = extract_features(train_paths)
    y_train = np.array(train_labels)
    X_val = extract_features(val_paths)
    y_val = np.array(val_labels)
    X_test = extract_features(test_paths)
    y_test = np.array(test_labels)
    
    summary_results = {}
    
    # 3. 循环训练每种算法
    for name, model_info in ML_MODELS.items():
        print(f"\n" + "="*80)
        print(f"流程: {name}")
        print("="*80)
        
        # 调优
        best_params = tune_ml_model(name, model_info, X_train, y_train, X_val, y_val)
        
        # 多种子验证
        seed_metrics = []
        for seed in RANDOM_SEEDS:
            print(f"--- 种子训练: {seed} ---")
            model = model_info["class"](**best_params, random_state=seed)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            seed_metrics.append(metrics)
            
        # 汇总统计
        if seed_metrics:
            metrics_to_avg = ['accuracy', 'precision', 'recall', 'f1_score']
            final_results = {'mean': {}, 'std': {}, 'best_params': best_params}
            
            for m in metrics_to_avg:
                vals = [res[m] for res in seed_metrics]
                final_results['mean'][m] = float(np.mean(vals))
                final_results['std'][m] = float(np.std(vals))
            
            summary_results[name] = final_results
            
            # 保存报告
            save_path = os.path.join(ML_RESULTS_DIR, f"{name}_results.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2)

    # 打印最终总结
    print("\n" + "="*80)
    print("传统机器学习算法汇总 (Mean ± Std)")
    print("="*80)
    print(f"{'模型':<20} {'准确率 %':<20} {'F1分数 %':<20}")
    print("-" * 80)
    for name, res in summary_results.items():
        m = res['mean']
        s = res['std']
        print(f"{name:<20} {m['accuracy']*100:>5.2f} ± {s['accuracy']*100:>4.2f}  "
              f"{m['f1_score']*100:>5.2f} ± {s['f1_score']*100:>4.2f}")

if __name__ == "__main__":
    main()
