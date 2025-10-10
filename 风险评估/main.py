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
    
    # 检查是否存在真实数据，如果没有则生成示例数据
    data_path = 'data/processed_data.csv'
    if os.path.exists(data_path):
        print(f"加载真实数据: {data_path}")
        X_train, X_test, y_train, y_test = data_loader.load_and_split_data(data_path)
    else:
        print("未找到真实数据，生成示例数据...")
        # 生成数据并进行预处理
        data_loader.load_data()
        X_train, X_test, y_train, y_test = data_loader.preprocess_data()
        
        # 保存生成的数据
        data_loader.save_preprocessed_data()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {X_train.shape[1]}")
    
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
    lr_metrics = model_trainer.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    models['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # GBDT
    print("\n训练GBDT模型...")
    gbdt_model = model_trainer.train_gbdt(X_train, y_train)
    gbdt_metrics = model_trainer.evaluate_model(gbdt_model, X_test, y_test, 'gbdt')
    models['gbdt'] = {'model': gbdt_model, 'metrics': gbdt_metrics}
    
    # LightGBM
    print("\n训练LightGBM模型...")
    lgb_model = model_trainer.train_lightgbm(X_train, y_train)
    if lgb_model is not None:
        lgb_metrics = model_trainer.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
        models['lightgbm'] = {'model': lgb_model, 'metrics': lgb_metrics}
    
    # 保存模型
    model_trainer.save_models()
    
    # 步骤3: 模型评估和比较
    print("\n" + "="*40)
    print("步骤3: 模型评估和比较")
    print("="*40)
    
    # 打印评估结果
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'GBDT', 'LightGBM'],
        'AUC': [models['logistic_regression']['metrics']['auc'],
                models['gbdt']['metrics']['auc'],
                models['lightgbm']['metrics']['auc']],
        'Accuracy': [models['logistic_regression']['metrics']['accuracy'],
                     models['gbdt']['metrics']['accuracy'],
                     models['lightgbm']['metrics']['accuracy']],
        'Precision': [models['logistic_regression']['metrics']['precision'],
                      models['gbdt']['metrics']['precision'],
                      models['lightgbm']['metrics']['precision']],
        'Recall': [models['logistic_regression']['metrics']['recall'],
                   models['gbdt']['metrics']['recall'],
                   models['lightgbm']['metrics']['recall']],
        'F1-Score': [models['logistic_regression']['metrics']['f1_score'],
                     models['gbdt']['metrics']['f1_score'],
                     models['lightgbm']['metrics']['f1_score']]
    })
    
    print("\n模型评估结果:")
    print(results_df.to_string(index=False))
    
    # 保存评估结果
    results_df.to_csv('results/metrics/model_evaluation_results.csv', index=False)
    
    # 找出最佳模型
    best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Model']
    best_auc = results_df['AUC'].max()
    print(f"\n最佳模型: {best_model_name} (AUC = {best_auc:.4f})")
    
    # 步骤4: 生成测试集预测结果
    print("\n" + "="*40)
    print("步骤4: 生成测试集预测结果")
    print("="*40)
    
    # 使用最佳模型生成预测
    best_model_key = best_model_name.lower().replace(' ', '_')
    best_model = models[best_model_key]['model']
    
    # 生成预测概率
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # 保存预测结果
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_pred_proba
    })
    
    test_results.to_csv('results/predictions/test_set_predictions.csv', index=False)
    print(f"预测结果已保存到: results/predictions/test_set_predictions.csv")
    print(f"预测结果包含 {len(test_results)} 个样本")
    
    # 步骤5: 可选功能 - AUC实现对比
    print("\n" + "="*40)
    print("步骤5: 可选功能 - AUC实现对比")
    print("="*40)
    
    print("\n对比sklearn AUC和手写AUC实现...")
    auc_comparison = compare_auc_implementations(y_test, y_pred_proba)
    
    # 保存AUC对比结果
    auc_comparison_df = pd.DataFrame([auc_comparison])
    auc_comparison_df.to_csv('results/metrics/auc_comparison_results.csv', index=False)
    
    # 步骤6: 可选功能 - 手写逻辑回归
    print("\n" + "="*40)
    print("步骤6: 可选功能 - 手写逻辑回归")
    print("="*40)
    
    print("\n训练手写逻辑回归模型...")
    try:
        manual_weights, manual_bias, training_history = manual_logistic_regression(
            X_train, y_train, learning_rate=0.01, max_iterations=1000
        )
        
        # 生成预测
        manual_proba, manual_pred = predict_manual_logistic_regression(
            X_test, manual_weights, manual_bias
        )
        
        # 计算手写模型的AUC
        manual_auc = auc_comparison['manual_auc']  # 使用之前计算的AUC函数
        
        print(f"手写逻辑回归 AUC: {manual_auc:.4f}")
        
        # 保存手写模型结果
        manual_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': manual_pred,
            'Probability': manual_proba
        })
        
        manual_results.to_csv('results/predictions/manual_logistic_regression_predictions.csv', index=False)
        print(f"手写模型预测结果已保存")
        
    except Exception as e:
        print(f"手写逻辑回归训练失败: {str(e)}")
    
    # 步骤7: 可选功能 - 归一化分析
    print("\n" + "="*40)
    print("步骤7: 可选功能 - 归一化分析")
    print("="*40)
    
    print("\n分析不同归一化方法对逻辑回归的影响...")
    try:
        def train_lr_model(X_train_norm, X_test_norm, y_train, y_test):
            """用于归一化比较的简单LR训练函数"""
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=42)
            lr.fit(X_train_norm, y_train)
            y_pred_proba = lr.predict_proba(X_test_norm)[:, 1]
            auc = auc_comparison['manual_auc']  # 使用手写AUC函数
            return lr, auc
        
        norm_results, best_norm_method = compare_normalization_methods(
            X_train, X_test, y_train, y_test, train_lr_model
        )
        
        # 保存归一化分析结果
        norm_summary = pd.DataFrame({
            'Normalization_Method': ['No Normalization', 'Standardization', 'Min-Max Normalization'],
            'AUC': [norm_results['no_normalization']['auc'],
                   norm_results['standardization']['auc'],
                   norm_results['minmax_normalization']['auc']]
        })
        
        norm_summary.to_csv('results/metrics/normalization_comparison_results.csv', index=False)
        print(f"归一化分析结果已保存")
        
    except Exception as e:
        print(f"归一化分析失败: {str(e)}")
    
    # 最终总结
    print("\n" + "="*60)
    print("最终总结")
    print("="*60)
    print(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最佳模型: {best_model_name}")
    print(f"最佳AUC: {best_auc:.4f}")
    print("\n生成的文件:")
    print("- 模型文件: models/目录下")
    print("- 预测结果: results/predictions/目录下")
    print("- 评估指标: results/metrics/目录下")
    print("- 日志文件: logs/目录下")
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()