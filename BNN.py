# -*- coding: utf-8 -*-
"""
贝叶斯神经网络经济变量建模 - 三层网络结构
包含网络结构调整、元数据结构修复和完整逆标准化
"""
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
from pyro.infer.predictive import Predictive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import time
import joblib
import pickle
import os
from collections import namedtuple

# 全局配置
pyro.set_rng_seed(42)
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

# 修改为8个特征，根据你的要求
NUM_FEATURES = 8
HIDDEN_DIM = 5

class BayesianEconomicModel:
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM):
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
    def model(self, X, y=None):
        # 类型检查与转换
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
            
        # 定义三层网络参数
        # 第一层：输入层→隐藏层1
        w1 = pyro.sample("w1", dist.Normal(0, 1.0).expand([self.hidden_dim, self.num_features]).to_event(2))
        b1 = pyro.sample("b1", dist.Normal(0, 0.5).expand([self.hidden_dim]).to_event(1))
        
        # 第二层：隐藏层1→隐藏层2
        w2 = pyro.sample("w2", dist.Normal(0, 1.0).expand([self.hidden_dim, self.hidden_dim]).to_event(2))
        b2 = pyro.sample("b2", dist.Normal(0, 0.5).expand([self.hidden_dim]).to_event(1))
        
        # 第三层：隐藏层2→输出层
        w3 = pyro.sample("w3", dist.Normal(0, 1.0).expand([1, self.hidden_dim]).to_event(2))
        b3 = pyro.sample("b3", dist.Normal(0, 0.5).expand([1]).to_event(1))
        
        sigma = pyro.sample("sigma", dist.HalfNormal(0.5))

        # 网络前向传播
        with pyro.plate("data", X.shape[0]):
            # 第一层 + sigmoid激活
            hidden1 = torch.sigmoid(torch.mm(X, w1.t()) + b1)  # 确保不为负
            # 第二层 + sigmoid激活
            hidden2 = torch.sigmoid(torch.mm(hidden1, w2.t()) + b2)  # 确保不为负
             # 第三层（输出层）
            mu = torch.mm(hidden2, w3.t()) + b3
    
            if y is not None:
                pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
            else:
                pyro.sample("obs", dist.Normal(mu, sigma).to_event(1))

    def train(self, X, y, num_samples=1000, warmup=300):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
        
        nuts_kernel = NUTS(
            self.model,
            jit_compile=True,
            target_accept_prob=0.8,
            max_tree_depth=8
        )
        
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup,
            num_chains=1,
            disable_progbar=False  # 显示采样进度条
        )
        mcmc.run(X_tensor, y_tensor)
        
        self.samples = {
            k: v.detach().to(dtype=torch.float32) 
            for k, v in mcmc.get_samples().items()
        }
        return self
    
    def predict(self, X, num_samples=1000):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        predictive = Predictive(
            self.model,
            posterior_samples=self.samples,
            return_sites=["obs"]
        )
        
        pred_samples = predictive(X_tensor)["obs"].detach().numpy()
        return pred_samples

def describe_parameter_distributions(samples):
    """详细描述所有参数分布（均值和标准差）"""
    print("\n" + "="*50)
    print("参数分布统计量")
    print("="*50)
    
    # 排除观测值
    param_names = [k for k in samples.keys() if k != "obs"]
    
    for i, param_name in enumerate(param_names):
        param_data = samples[param_name]
        
        # 处理高维参数
        if len(param_data.shape) > 1:
            # 展平高维参数
            param_data = param_data.reshape(-1)
            
        # 计算统计量
        mean_val = param_data.mean()
        std_val = param_data.std()
        
        print(f"\n➜ 参数: {param_name}")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        print(f"  形状: {samples[param_name].shape}")
        
        if len(samples[param_name].shape) > 1:
            # 对于矩阵参数，显示每列的平均值
            mat_mean = samples[param_name].mean(axis=0)
            print(f"  每列均值: {mat_mean}")
    
    print("="*50 + "\n")
    return True

def plot_parameter_distributions(samples, param_names=None):
    """绘制所有参数的分布"""
    if param_names is None:
        param_names = [k for k in samples.keys() if k != "obs"]
    
    columns = 4
    rows = int(np.ceil(len(param_names) / columns))
    
    plt.figure(figsize=(15, 4 * rows))
    for i, param_name in enumerate(param_names):
        plt.subplot(rows, columns, i+1)
        param_data = samples[param_name]
        
        # 处理高维参数
        if len(param_data.shape) > 1:
            # 展平高维参数
            param_data = param_data.reshape(-1)
            display_name = param_name + " (flattened)"
        else:
            display_name = param_name
        
        # 计算统计量
        mean_val = param_data.mean()
        std_val = param_data.std()
        
        sns.histplot(x=param_data, kde=True)
        plt.title(f"{display_name}\n均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
        plt.xlabel("参数值")
    
    plt.tight_layout()
    plt.show()
    return True

def plot_prediction_intervals(pred_samples, y_true=None, title="Predictions", xlabel="Time Step", ylabel="Value"):
    """绘制预测区间"""
    mean_pred = pred_samples.mean(0).flatten()  # 确保一维
    lower_bound = np.percentile(pred_samples, 2.5, axis=0).flatten()  # 确保一维
    upper_bound = np.percentile(pred_samples, 97.5, axis=0).flatten()  # 确保一维
    
    plt.figure(figsize=(10, 5))
    if y_true is not None:
        plt.plot(y_true.flatten(), label="实际值", color="blue", alpha=0.7)  # 展平为一维
    
    plt.plot(mean_pred, label="预测值", color="red")
    
    # 填充95%置信区间
    plt.fill_between(
        range(len(mean_pred)), 
        lower_bound, 
        upper_bound, 
        color="gray", alpha=0.3, label="95% 置信区间"
    )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} 与95%置信区间")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title}_prediction_intervals.png")
    plt.show()
    print(f"✅ 预测区间图已保存为: {title}_prediction_intervals.png")
    return True

def inverse_transform_target(meta_data, predictions, target_idx):
    """
    将目标变量预测值完整逆转换到原始尺度
    按顺序匹配: 假设meta_data中的顺序与预测顺序一致
    
    步骤:
    1. 获取目标变量的元数据（按顺序）
    2. 逆标准化：原始差分值 = 标准化值 * scale + mean
    3. 逆差分：累加差分值得到水平值 (从初始值开始)
    
    参数:
    predictions: 形状为 (n_samples, n_timesteps) 的预测值
    target_idx: 目标变量的索引（0,1或2）
    
    返回:
    逆转换后的原始尺度值，形状为 (n_samples, n_timesteps)
    """
    # 1. 通过索引获取目标变量名称
    target_name = meta_data['target_names'][target_idx]
    
    # 2. 获取元数据
    initial_value = meta_data['initial_values'][target_name]
    scaler_params = meta_data['scaler_params'][target_name]
    
    # 确保有正确的参数
    mean = scaler_params['mean']
    scale = scaler_params['scale']
    
    # 3. 逆标准化：恢复差分值
    inv_std_preds = predictions * scale + mean
    
    # 4. 逆差分：从初始值开始累加差分值
    raw_preds = np.zeros_like(inv_std_preds)
    
    # 对每个样本进行逆差分
    for i in range(inv_std_preds.shape[0]):
        cumulative = initial_value
        for j in range(inv_std_preds.shape[1]):
            cumulative += inv_std_preds[i, j]
            raw_preds[i, j] = cumulative
    
    return raw_preds

# =============== 严格的元数据加载函数 ===============
def critical_load_metadata(path):
    """严格加载元数据，出错时直接报错"""
    try:
        print(f"\n⩶⩶⩶⩶⩶⩶ 开始加载元数据: {path} ⩶⩶⩶⩶⩶⩶")
        with open(path, 'rb') as f:
            meta_data = pickle.load(f)
        
        print("✅ 文件加载成功")
        print(f"元数据类型: {type(meta_data)}")
        
        # 验证必需字段
        required_keys = ['target_names', 'initial_values', 'scaler_params']
        for key in required_keys:
            if key not in meta_data:
                raise ValueError(f"元数据缺少必需字段: {key}")
        print("✅ 必需字段验证通过")
        
        # 确保所有目标都有初始值和标准化参数
        all_targets = set(meta_data['target_names'])
        
        # 验证初始值
        if not isinstance(meta_data['initial_values'], dict):
            raise TypeError(f"initial_values应为字典, 实际为 {type(meta_data['initial_values'])}")
        
        # 验证scaler_params
        if not isinstance(meta_data['scaler_params'], dict):
            raise TypeError(f"scaler_params应为字典, 实际为 {type(meta_data['scaler_params'])}")
        
        # 验证所有目标是否都有初始值
        missing_init = all_targets - set(meta_data['initial_values'].keys())
        if missing_init:
            raise ValueError(f"目标缺失初始值: {', '.join(missing_init)}")
        
        # 验证所有目标是否都有scaler_params
        missing_scaler = all_targets - set(meta_data['scaler_params'].keys())
        if missing_scaler:
            raise ValueError(f"目标缺失标准化参数: {', '.join(missing_scaler)}")
        
        print("✅ 初始值和标准化参数完整性验证通过")
        
        # 确保原始值和标准化参数都是标量
        for target_name in all_targets:
            # 确保初始值是标量
            init_val = meta_data['initial_values'][target_name]
            if isinstance(init_val, (np.ndarray, list, tuple)):
                if len(init_val) == 1:
                    meta_data['initial_values'][target_name] = float(init_val[0])
                else:
                    raise ValueError(f"初始值应该是标量: {target_name} 的形状是 {np.array(init_val).shape}")
            
            # 确保标准化参数都是标量
            scaler_params = meta_data['scaler_params'][target_name]
            for param in ['mean', 'scale']:
                param_val = scaler_params.get(param)
                if param_val is None:
                    raise ValueError(f"缺少必需参数 '{param}' 对于目标 {target_name}")
                if isinstance(param_val, (np.ndarray, list, tuple)):
                    if len(param_val) == 1:
                        scaler_params[param] = float(param_val[0])
                    else:
                        raise ValueError(f"标准化参数 '{param}' 应该是标量: {target_name} 的形状是 {np.array(param_val).shape}")
        
        # 打印元数据表格
        print("\n" + "="*70)
        print("| {:12} | {:>14} | {:>14} | {:>14} | {:>14} |".format(
            "目标变量", "初始值", "均值(mean)", "缩放系数(scale)", "标准化初始值"))
        print("="*70)
        for target_name in meta_data['target_names']:
            mean_val = meta_data['scaler_params'][target_name]['mean']
            scale_val = meta_data['scaler_params'][target_name]['scale']
            normalized = (meta_data['initial_values'][target_name] - mean_val) / scale_val
            
            print("| {:12} | {:14.4f} | {:14.6f} | {:14.6f} | {:14.6f} |".format(
                target_name, meta_data['initial_values'][target_name], mean_val, 
                scale_val, normalized))
        print("="*70)
        
        return meta_data
    
    except Exception as e:
        print(f"\n❌ 严重: 元数据加载失败 ❌\n{'-'*50}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        print("-"*50)
        print("需要确保元数据文件包含:")
        print("- 'target_names': 目标变量名称列表")
        print("- 'initial_values': 每个目标的初始值字典")
        print("- 'scaler_params': 每个目标的标准化参数字典 (包含'mean'和'scale')")
        print("元数据结构应类似:")
        print({
            'target_names': ['TARGET1', 'TARGET2'],
            'initial_values': {'TARGET1': 100.0, 'TARGET2': 50.0},
            'scaler_params': {
                'TARGET1': {'mean': 90.0, 'scale': 10.0},
                'TARGET2': {'mean': 40.0, 'scale': 5.0}
            }
        })
        print("="*70)
        raise RuntimeError("元数据加载失败导致程序终止") from e


if __name__ == "__main__":
    # 1. 加载预处理后的数据
    print(f"➜ 加载数据，预期特征数量: {NUM_FEATURES}")
    pca_df = pd.read_csv(r"/mnt/pca_components.csv", parse_dates=['DATE'])
    
    # 关键修改：只读取前8个数值列（跳过第一列DATE）
    print("➜ 直接选择前8个数值特征（跳过日期列）")
    
    # 获取所有数值列（跳过第一列DATE）
    numeric_cols = [col for col in pca_df.columns if col != 'DATE'][:NUM_FEATURES]
    print(f"使用的前{NUM_FEATURES}个特征: {numeric_cols}")
    
    X = pca_df[numeric_cols].values.astype(np.float32)
    
    targets_df = pd.read_csv(r"/mnt/standardized_targets.csv")
    
    # 只保留前3列目标变量（就业、CPI、工业产出）
    # 跳过第一列DATE，取接下来的3列
    if len(targets_df.columns) >= 4:
        target_columns = targets_df.columns[1:4].tolist()
    else:
        # 如果列数不足，取所有数值列
        target_columns = [col for col in targets_df.columns if col != 'DATE'][:3]
    
    y = targets_df[target_columns].values.astype(np.float32)
    
    print(f"\n数据形状 - 解释变量: {X.shape}, 目标变量: {y.shape}")
    print(f"目标列名: {target_columns}")
    
    # 处理y中的缺失值
    for i in range(y.shape[1]):
        col = y[:, i]
        col_mean = np.nanmean(col)
        col = np.where(np.isnan(col), col_mean, col)
        y[:, i] = col
    
    # 2. 加载并简化元数据
    metadata_path = r"/mnt/targets_metadata.pkl"
    try:
        meta_data = critical_load_metadata(metadata_path)
    except Exception as e:
        print(f"\n❌ 无法加载元数据: {str(e)}")
        print("程序终止")
        exit(1)
    
    # 3. 分别对每个目标变量进行处理
    date_index = pca_df['DATE']
    
    for target_idx in range(y.shape[1]):
        # 获取目标变量名称
        target_name = meta_data['target_names'][target_idx]
        
        print(f"\n{'='*60}")
        print(f"★ 处理目标变量 [索引: {target_idx}] - {target_name} ")
        print(f"{'='*60}")
        
        # 使用全部数据进行训练 - 特征维度修改为8
        model = BayesianEconomicModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM)
        start_time = time.time()
        print(f"➜ 训练三层网络模型... (特征数 = {NUM_FEATURES}, 隐藏单元: {HIDDEN_DIM})")
        model.train(X, y[:, target_idx], num_samples=1000, warmup=300)
        print(f"✓ 训练完成! 耗时: {time.time()-start_time:.2f}秒")
        
        # 1. 输出参数详细分布
        describe_parameter_distributions(model.samples)
        plot_parameter_distributions(model.samples)
        
        # 2. 在所有数据上进行预测
        print("➜ 在完整数据集上预测...")
        pred_samples = model.predict(X, num_samples=1000)  # 形状: (1000, n_timesteps)
        
        # 计算标准化尺度上的RMSE
        mean_pred = pred_samples.mean(0)
        rmse = mean_squared_error(y[:, target_idx], mean_pred, squared=False)
        print(f"✧ 全样本RMSE (标准化尺度): {rmse:.4f}")
        
        # 3. 完整逆变换到原始尺度
        print("➜ 完整逆变换预测值到原始尺度...")
        raw_preds = inverse_transform_target(meta_data, pred_samples, target_idx)
        
        # 4. 逆变换真实值（用于比较）
        # 注意：y是标准化的差分值，需要复制与预测相同的结构进行逆变换
        y_true_expanded = np.tile(y[:, target_idx], (pred_samples.shape[0], 1))
        raw_true = inverse_transform_target(meta_data, y_true_expanded, target_idx)
        raw_true_mean = raw_true.mean(axis=0)  # 原始真实值（均值）
        
        # 计算原始尺度上的RMSE
        mean_raw_pred = raw_preds.mean(axis=0)
        rmse_raw = mean_squared_error(raw_true_mean, mean_raw_pred, squared=False)
        print(f"✧ 全样本RMSE (原始尺度): {rmse_raw:.4f}")
        
        # 5. 绘制标准化尺度预测区间
        print("➜ 绘制标准化尺度预测区间...")
        plot_prediction_intervals(
            pred_samples, 
            y[:, target_idx], 
            title=f"标准化尺度预测 - {target_name}",
            ylabel="标准化值"
        )
        # 6. 绘制原始尺度预测区间（完整逆变换后）
        print("➜ 绘制原始尺度预测区间（含95%置信区间）...")
        raw_preds_reshaped = raw_preds.reshape(1000, -1)  # 确保正确的形状
        plot_prediction_intervals(
            raw_preds_reshaped, 
            raw_true_mean, 
            title=f"原始尺度预测 - {target_name}",
            ylabel="原始值"
        )
        
        # 7. 预测下一个月的值
        print("➜ 预测下个月的值...")
        last_month_features = X[-1].reshape(1, -1)
        next_month_pred = model.predict(last_month_features, num_samples=1000)  # 形状: (1000, 1)
        
        # 8. 逆变换下个月的预测值到原始尺度
        next_month_raw = inverse_transform_target(meta_data, next_month_pred[:, 0:1], target_idx).flatten()
        # 获取最后观测值作为参考
        initial_value = meta_data['initial_values'][target_name]
        current_val = raw_true_mean[-1] if len(raw_true_mean) > 0 else 0
        # 计算逆变换后的统计量
        mean_next = next_month_raw.mean() + current_val - initial_value
        std_next = next_month_raw.std()
        lower = np.percentile(next_month_raw, 2.5) + current_val - initial_value
        upper = np.percentile(next_month_raw, 97.5) + current_val - initial_value
        
        print(f"\n✧ {target_name} - 原始初始值: {initial_value:.4f}")
        print(f"✧ {target_name} - 当前原始值: {current_val:.4f}")
        print(f"✧ {target_name} - 下个月预测值 (原始尺度):")
        print(f"  均值预测: {mean_next:.4f}")
        print(f"  标准差: {std_next:.4f}")
        print(f"  95%置信区间: [{lower:.4f}, {upper:.4f}]")
        if current_val != 0:
            change_percent = 100 * (mean_next - current_val) / current_val
            print(f"  相比当前值变化: {mean_next - current_val:.4f} ({change_percent:.2f}%)")
            print(f"  相比初始值变化: {mean_next - initial_value:.4f} ({(100 * (mean_next - initial_value)/initial_value):.2f}%)")

                        
        
        # 10. 可视化历史数据与预测
        plt.figure(figsize=(12, 6))
        
        # 最后12个月的实际数据
        lookback = min(12, len(date_index))
        
        # 确保只展示有效数据
        if lookback <= 0:
            print("⚠️ 警告: 没有足够的历史数据用于绘图")
            continue
            
        # 日期索引
        plot_dates = date_index.iloc[-lookback:]
        plot_raw_true = raw_true_mean[-lookback:]
        
        # 预测的历史值（取最后12个月）
        plot_pred_mean = mean_raw_pred[-lookback:].flatten()  # 确保是一维数组
        pred_lower = np.percentile(raw_preds, 2.5, axis=0)[-lookback:].flatten()  # 展平为一维
        pred_upper = np.percentile(raw_preds, 97.5, axis=0)[-lookback:].flatten()  # 展平为一维
        
        # 真实值
        plt.plot(
            plot_dates, 
            plot_raw_true, 
            'o-', label='历史真实值', color='blue', markersize=6
        )
        
        # 预测值（历史部分）
        plt.plot(
            plot_dates, 
            plot_pred_mean, 
            's--', label='模型回测', color='red', alpha=0.8, markersize=5
        )
        
        # 预测值（历史部分）的置信区间
        plt.fill_between(
            plot_dates, 
            pred_lower, 
            pred_upper, 
            color='red', alpha=0.15, label='95%置信区间'
        )
        
        # 下个月预测
        next_date = date_index.iloc[-1] + pd.DateOffset(months=1)
        plt.plot(
            next_date, 
            mean_next, 
            'D', markersize=8, 
            label=f'下月预测: {mean_next:.2f}', 
            color='green'
        )
        
        # 置信区间
        plt.errorbar(
            x=[next_date], 
            y=[mean_next], 
            yerr=[[mean_next - lower], [upper - mean_next]], 
            fmt='none', 
            ecolor='black', 
            elinewidth=2.0,
            capsize=8,
            capthick=2,
            label='预测区间'
        )
        
        plt.title(f"{target_name} - 三层模型预测")
        plt.xlabel("日期")
        plt.ylabel("原始值")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{target_name}_model_prediction.png")
        print(f"📈 预测图保存为: {target_name}_model_prediction.png")
        plt.show()
