import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

class DataLoader:
    def __init__(self, data_path=None):
        """初始化数据加载器"""
        if data_path is None:
            # 检查常见位置
            possible_paths = [
                "data/raw/data.csv",
                "data/raw/train.csv",
                "../数据预处理/data/final_processed_data.csv"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
    def generate_sample_data(self, n_samples=1000, n_features=20):
        """生成示例数据用于测试"""
        np.random.seed(42)
        
        # 生成特征数据
        X = np.random.randn(n_samples, n_features)
        
        # 添加一些分箱特征
        X_bin1 = np.random.choice(['A', 'B', 'C'], size=n_samples)
        X_bin2 = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
        
        # 生成目标变量（二分类）
        # 基于前几个特征创建一些模式
        linear_combo = 0.3 * X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2]
        prob = 1 / (1 + np.exp(-linear_combo))
        y = np.random.binomial(1, prob)
        
        # 创建DataFrame
        feature_cols = [f'X{i}' for i in range(n_features)]
        df_features = pd.DataFrame(X, columns=feature_cols)
        
        # 添加分箱特征
        df_features['X65'] = X_bin1  # 对应要求中的分箱特征
        df_features['X1_bin'] = X_bin2
        df_features['X66'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
        
        # 添加缺失值
        missing_idx = np.random.choice(df_features.index, size=int(0.05 * n_samples), replace=False)
        df_features.loc[missing_idx, 'X0'] = np.nan
        
        # 组合最终数据
        df = df_features.copy()
        df['target'] = y
        
        print(f"生成示例数据：{n_samples} 样本，{df.shape[1]-1} 特征")
        return df
    
    def load_data(self, generate_if_missing=True):
        """加载数据"""
        if self.data_path and os.path.exists(self.data_path):
            print(f"从 {self.data_path} 加载数据...")
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            else:
                # 尝试其他格式
                try:
                    self.data = pd.read_excel(self.data_path)
                except:
                    self.data = pd.read_csv(self.data_path, sep='\t')
        elif generate_if_missing:
            print("未找到数据文件，生成示例数据...")
            self.data = self.generate_sample_data()
            # 保存示例数据
            os.makedirs("data/raw", exist_ok=True)
            self.data.to_csv("data/raw/sample_data.csv", index=False)
            print("示例数据已保存到 data/raw/sample_data.csv")
        else:
            raise FileNotFoundError("未找到数据文件")
        
        print(f"数据加载完成，形状：{self.data.shape}")
        return self.data
    
    def preprocess_data(self, target_col='target', test_size=0.3, random_state=42):
        """数据预处理"""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # 分离特征和目标
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            # 假设最后一列是目标
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # 处理缺失值
        X = self.handle_missing_values(X)
        
        # 处理分箱特征
        X = self.handle_categorical_features(X)
        
        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"数据预处理完成:")
        print(f"训练集：{self.X_train.shape}")
        print(f"测试集：{self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_missing_values(self, X):
        """处理缺失值"""
        # 数值特征用中位数填充
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # 类别特征用众数填充
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def handle_categorical_features(self, X):
        """处理类别特征"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            # 对分箱特征进行编码
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            return X_encoded
        
        return X
    
    def get_feature_names(self):
        """获取特征名称"""
        if self.X_train is not None:
            return self.X_train.columns.tolist()
        return None
    
    def save_preprocessed_data(self, output_dir="data/processed"):
        """保存预处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集和测试集
        self.X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        self.X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        self.y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        self.y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        print(f"预处理数据已保存到 {output_dir}")

if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader()
    loader.load_data()
    loader.preprocess_data()
    loader.save_preprocessed_data()