#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险评估模型主程序
执行简化版的风险评估模型实现方案
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from data.data_loader import DataLoader
from models.model_trainer import RiskAssessmentModels
from evaluation.custom_metrics import compare_auc_implementations, manual_logistic_regression, predict_manual_logistic_regression
from data.normalization import compare_normalization_methods, visualize_normalization_effect

def main():
    """主函数"""
    print("="*60)
    print("风险评估模型实现程序")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建必要的目录
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models/lr', exist_ok=True)
    os.makedirs('models/gbdt', exist_ok=True)
    os.makedirs('models/lightgbm', exist_ok=True)
    
    # 步骤1: 数据加载和预处理
    print("\n" + "="*40)
    print("步骤1: 数据加载和预处理")
    print("="*40)
    
    data_loader = DataLoader()
    
    # 使用指定的真实数据文件
    train_data_path = 'data/train_new.csv'
    test_data_path = 'data/test_new.csv'
    feature_info_path = 'data/feature_x.csv'
    
    print(f"加载训练数据: {train_data_path}")
    print(f"加载测试数据: {test_data_path}")
    print(f"加载特征信息: {feature_info_path}")
    
    # 加载训练数据
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # 分离特征和标签
    X_train = train_data.drop(['Y', 'id'], axis=1)
    y_train = train_data['Y']
    X_test = test_data.drop(['id'], axis=1)  # 测试数据没有Y标签
    
    # 处理缺失值（用中位数填充数值特征）
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {X_train.shape[1]}")
    
    # 保存预处理后的数据
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train_processed.csv', index=False)
    X_test.to_csv('data/processed/X_test_processed.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
    
    print("数据预处理完成，已保存到data/processed/目录")
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {X_train.shape[1]}")
    
    # 保存测试数据引用用于后续预测
    test_df = test_data.copy()
    
    # 步骤2: 基础模型训练
    print("\n" + "="*40)
    print("步骤2: 基础模型训练")
    print("="*40)
    
    model_trainer = RiskAssessmentModels()
    
    # 训练三种模型
    models = {}
    
    # 逻辑回归
    print("\n训练逻辑回归模型...")
    lr_model = model_trainer.train_logistic_regression(X_train, y_train)
    models['logistic_regression'] = {'model': lr_model}
    
    # GBDT
    print("\n训练GBDT模型...")
    gbdt_model = model_trainer.train_gbdt(X_train, y_train)
    models['gbdt'] = {'model': gbdt_model}
    
    # LightGBM
    print("\n训练LightGBM模型...")
    lgb_model = model_trainer.train_lightgbm(X_train, y_train)
    if lgb_model is not None:
        models['lightgbm'] = {'model': lgb_model}
    
    # 保存模型
    model_trainer.save_models()
    
    # 步骤3: 生成测试集预测结果
    print("\n" + "="*40)
    print("步骤3: 生成测试集预测结果")
    print("="*40)
    
    # 为所有模型生成预测结果
    for model_key, model_data in models.items():
        model = model_data['model']
        model_display_name = model_key.replace('_', ' ').title()
        
        # 生成预测概率
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # 保存预测结果
        test_results = pd.DataFrame({
            'id': test_df['id'],
            'Predicted': y_pred,
            'Probability': y_pred_proba
        })
        
        # 为每个模型生成独立的预测文件
        pred_filename = f"results/predictions/{model_key}_test_predictions.csv"
        test_results.to_csv(pred_filename, index=False)
        print(f"{model_display_name} 预测结果已保存到: {pred_filename}")
        print(f"预测结果包含 {len(test_results)} 个样本")
    
    # 步骤4: 生成汇总预测文件
    print("\n" + "="*40)
    print("步骤4: 生成汇总预测文件")
    print("="*40)
    
    # 使用逻辑回归模型生成汇总预测结果（作为默认推荐）
    lr_model = models['logistic_regression']['model']
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    y_pred = lr_model.predict(X_test)
    
    # 保存汇总预测结果
    best_test_results = pd.DataFrame({
        'id': test_df['id'],
        'Predicted': y_pred,
        'Probability': y_pred_proba
    })
    
    best_test_results.to_csv('results/predictions/best_model_test_predictions.csv', index=False)
    print(f"\n逻辑回归模型预测结果已保存到: results/predictions/best_model_test_predictions.csv")
    
    # 步骤5: 可选功能 - 交叉验证评估
    print("\n" + "="*40)
    print("步骤5: 可选功能 - 交叉验证评估")
    print("="*40)
    
    print("\n使用交叉验证评估模型性能...")
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score
        
        # 交叉验证评估
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = {}
        for model_key, model_data in models.items():
            model = model_data['model']
            model_display_name = model_key.replace('_', ' ').title()
            
            # AUC交叉验证
            auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            
            cv_results[model_display_name] = {
                'mean_auc': auc_scores.mean(),
                'std_auc': auc_scores.std(),
                'auc_scores': auc_scores.tolist()
            }
            
            print(f"{model_display_name} - 交叉验证AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std() * 2:.4f})")
        
        # 保存交叉验证结果
        import json
        cv_results_path = 'results/metrics/cross_validation_results.json'
        with open(cv_results_path, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        print(f"\n交叉验证结果已保存到: {cv_results_path}")
        
    except Exception as e:
        print(f"交叉验证评估失败: {str(e)}")
    
    # 最终总结
    print("\n" + "="*60)
    print("最终总结")
    print("="*60)
    print(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n生成的文件:")
    print("- 模型文件: models/目录下")
    print("- 预测结果: results/predictions/目录下")
    print("- 评估指标: results/metrics/目录下")
    print("- 日志文件: logs/目录下")
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()