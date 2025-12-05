import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import time

class RiskAssessmentModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, feature_names=None):
        """训练逻辑回归模型"""
        print("训练逻辑回归模型...")
        start_time = time.time()
        
        # 使用默认参数
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        self.models['lr'] = lr_model
        print(f"逻辑回归模型训练完成，耗时：{training_time:.2f}秒")
        
        return lr_model
    
    def train_gbdt(self, X_train, y_train, feature_names=None):
        """训练GBDT模型"""
        print("训练GBDT模型...")
        start_time = time.time()
        
        # 使用默认参数
        gbdt_model = GradientBoostingClassifier(random_state=42)
        gbdt_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        self.models['gbdt'] = gbdt_model
        print(f"GBDT模型训练完成，耗时：{training_time:.2f}秒")
        
        return gbdt_model
    
    def train_lightgbm(self, X_train, y_train, feature_names=None):
        """训练LightGBM模型"""
        try:
            import lightgbm as lgb
            print("训练LightGBM模型...")
            start_time = time.time()
            
            # 使用默认参数
            lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            self.models['lightgbm'] = lgb_model
            print(f"LightGBM模型训练完成，耗时：{training_time:.2f}秒")
            
            return lgb_model
            
        except ImportError:
            print("LightGBM未安装，跳过LightGBM模型训练")
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """评估模型性能"""
        print(f"评估 {model_name} 模型...")
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'auc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = results
        
        print(f"{model_name} 评估结果:")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """训练所有模型"""
        print("开始训练所有模型...")
        
        # 训练逻辑回归
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_results = self.evaluate_model(lr_model, X_test, y_test, 'lr')
        
        # 训练GBDT
        gbdt_model = self.train_gbdt(X_train, y_train)
        gbdt_results = self.evaluate_model(gbdt_model, X_test, y_test, 'gbdt')
        
        # 训练LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train)
        if lgb_model is not None:
            lgb_results = self.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
        
        print("所有模型训练完成！")
        return self.results
    
    def save_models(self, output_dir="models"):
        """保存训练好的模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/{model_name}/best_model.pkl"
            os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
            joblib.dump(model, model_path)
            print(f"模型 {model_name} 已保存到 {model_path}")
    
    def save_predictions(self, output_dir="results/predictions"):
        """保存预测结果"""
        import pandas as pd
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, results in self.results.items():
            predictions_df = pd.DataFrame({
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            })
            
            pred_path = f"{output_dir}/{model_name}_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            print(f"预测结果已保存到 {pred_path}")
    
    def save_metrics(self, output_dir="results/metrics"):
        """保存评估指标"""
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        
        # 保存每个模型的指标
        for model_name, results in self.results.items():
            metrics = {
                'model_name': results['model_name'],
                'auc': results['auc'],
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
            
            metrics_path = f"{output_dir}/{model_name}_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"评估指标已保存到 {metrics_path}")
        
        # 保存对比结果
        comparison = {}
        for model_name, results in self.results.items():
            comparison[model_name] = {
                'auc': results['auc'],
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score']
            }
        
        comparison_path = f"{output_dir}/model_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"模型对比结果已保存到 {comparison_path}")
    
    def get_best_model(self, metric='auc'):
        """获取最佳模型"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1]

if __name__ == "__main__":
    # 测试模型训练
    import sys
    sys.path.append('src')
    from data.data_loader import DataLoader
    
    # 加载数据
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.preprocess_data()
    
    # 训练模型
    trainer = RiskAssessmentModels()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # 保存结果
    trainer.save_models()
    trainer.save_predictions()
    trainer.save_metrics()
    
    # 显示最佳模型
    best_model_name, best_results = trainer.get_best_model()
    print(f"最佳模型：{best_model_name} (AUC: {best_results['auc']:.4f})")