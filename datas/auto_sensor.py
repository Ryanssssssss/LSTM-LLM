#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 定义基本路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
DATA_PATH = BASE_PATH  # 数据文件在当前目录
OUTPUT_PATH_RAW = os.path.join(BASE_PATH, "con_raw")  # 原始数据结果文件夹
OUTPUT_PATH_NORM = os.path.join(BASE_PATH, "con_normalized")  # 归一化数据结果文件夹

# 定义传感器文件
SENSOR_FILES = [
    "mq7b_no_normal.xlsx",
    "mq2_no_normal.xlsx",
    "mp801_no_normal.xlsx",
    "mp503_no_normal.xlsx",
    "mp3b_no_normal.xlsx",
    "mp2_no_normal.xlsx",
    "mp3b_normal_1.xlsx"
]

# 定义气体类型映射关系
GAS_TYPE_MAPPING = {
    # 乙醇 (1-6) -> 1
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    # 丙酮 (7-12) -> 2
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,
    # 甲醛 (13-18) -> 3
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
    # 甲苯 (19-24) -> 4
    19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4
}

# 定义分割比例
SPLIT_RATE = 0.5

def create_directory(directory):
    """创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def load_labels():
    """从标签文件加载标签"""
    print("正在加载标签...")
    label_file = "mp3b_normal_1.xlsx"
    file_path = os.path.join(DATA_PATH, label_file)
    print(f"读取标签文件: {file_path}")
    
    df = pd.read_excel(file_path, header=None)
    original_labels = df.iloc[:, -1].values
    
    # 将原始标签转换为气体类型标签 (1-4)
    gas_type_labels = np.array([GAS_TYPE_MAPPING[label] for label in original_labels])
    
    print(f"✓ 标签加载完成，共 {len(original_labels)} 个样本")
    return original_labels, gas_type_labels

def load_single_sensor(file_path):
    """加载单个传感器文件数据并进行前向填充+后向填充"""
    df = pd.read_excel(file_path, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if df.isnull().values.any():
        null_count = df.isnull().sum().sum()
        null_positions = df.isnull()
        null_rows = null_positions.any(axis=1)
        null_row_indices = null_rows[null_rows].index.tolist()
        print(f"  发现 {null_count} 个缺失值，分布在 {len(null_row_indices)} 行")
        print(f"  使用前向填充 (ffill) 处理缺失值")
        df = df.ffill(axis=1)
        
        # 如果前向填充后仍有 NaN（第一列的情况），使用后向填充
        if df.isnull().values.any():
            remaining_nulls = df.isnull().sum().sum()
            print(f"  ⚠️  前向填充后仍有 {remaining_nulls} 个缺失值，使用后向填充 (bfill)")
            df = df.bfill(axis=1)
            
            # 最终检查
            final_nulls = df.isnull().sum().sum()
            if final_nulls > 0:
                print(f"  ❌ 警告：填充后仍有 {final_nulls} 个缺失值，将用 0 填充")
                df = df.fillna(0)
    
    return df.iloc[:, :361].values

def get_concentration_samples(original_labels):
    """获取每个浓度对应的样本索引"""
    concentration_indices = {}
    
    # 获取每个浓度的样本索引
    for con in range(1, 7):
        # 获取乙醇的当前浓度索引
        ethanol_indices = np.where(original_labels == con)[0]
        # 获取丙酮的当前浓度索引
        acetone_indices = np.where(original_labels == con + 6)[0]
        # 获取甲醛的当前浓度索引
        formaldehyde_indices = np.where(original_labels == con + 12)[0]
        # 获取甲苯的当前浓度索引
        toluene_indices = np.where(original_labels == con + 18)[0]
        
        # 合并所有当前浓度的样本索引
        concentration_indices[con] = np.concatenate([
            ethanol_indices, acetone_indices, formaldehyde_indices, toluene_indices
        ])
    
    return concentration_indices

def split_train_test(concentration_indices):
    """为每个浓度划分训练集和测试集 (50:50)"""
    train_test_indices = {}
    
    for con, indices in concentration_indices.items():
        # 使用50%作为测试集
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=SPLIT_RATE, 
            random_state=42,
            shuffle=True
        )
        
        train_test_indices[con] = {
            'train': train_idx,
            'test': test_idx
        }
        
        print(f"浓度 {con}: 训练集={len(train_idx)}, 测试集={len(test_idx)}")
    
    return train_test_indices

def normalize_sensor_data(train_data, test_data):
    """归一化单个传感器数据 (2D数组，按列归一化)"""
    # train_data, test_data 的形状是 (samples, features)
    train_norm = np.zeros_like(train_data)
    test_norm = np.zeros_like(test_data)
    
    features = train_data.shape[1]
    
    # 对每个特征列分别进行归一化
    for feat in range(features):
        train_col = train_data[:, feat].reshape(-1, 1)
        test_col = test_data[:, feat].reshape(-1, 1)
        
        # 创建归一化器并在训练集上拟合
        scaler = StandardScaler()
        scaler.fit(train_col)
        
        # 对训练集和测试集进行归一化
        train_norm[:, feat] = scaler.transform(train_col).flatten()
        test_norm[:, feat] = scaler.transform(test_col).flatten()
    
    return train_norm, test_norm

def prepare_and_save_data(gas_type_labels, train_test_indices, concentration_indices):
    """准备并保存数据：先分割，再归一化，最后堆叠"""
    print("\n=== 开始准备数据 ===")
    
    # 创建结果文件夹
    create_directory(OUTPUT_PATH_RAW)
    create_directory(OUTPUT_PATH_NORM)
    
    # 获取传感器文件列表（排除标签文件）
    sensor_files = [f for f in SENSOR_FILES if f != "mp3b_normal_1.xlsx"]
    
    # 为每个源域-目标域组合准备数据
    for source_con in range(1, 7):  # 源域浓度 (1-6)
        for target_con in range(1, 7):  # 目标域浓度 (1-6)
            experiment_name = f"con{source_con}con{target_con}Sensor"
            print(f"\n处理实验: {experiment_name}")
            
            # 创建输出目录（在两个结果文件夹下都创建）
            output_dir_raw = os.path.join(OUTPUT_PATH_RAW, experiment_name)
            output_dir_norm = os.path.join(OUTPUT_PATH_NORM, experiment_name)
            create_directory(output_dir_raw)
            create_directory(output_dir_norm)
            
            # === 第1步: 确定训练集和测试集的样本索引 ===
            train_indices = train_test_indices[source_con]['train']
            
            if source_con == target_con:
                # 如果源域和目标域是同一个浓度，使用该浓度的测试部分
                test_indices = train_test_indices[target_con]['test']
            else:
                # 如果源域和目标域不同，使用目标域的全部样本
                test_indices = concentration_indices[target_con]
            
            # 准备标签
            y_train = gas_type_labels[train_indices]
            y_test = gas_type_labels[test_indices]
            
            # === 第2步: 对每个传感器分别处理（分割 + 归一化） ===
            train_channels_raw = []
            test_channels_raw = []
            train_channels_norm = []
            test_channels_norm = []
            
            for sensor_file in sensor_files:
                file_path = os.path.join(DATA_PATH, sensor_file)
                print(f"  处理传感器: {sensor_file}")
                
                # 加载传感器数据
                sensor_data = load_single_sensor(file_path)
                
                # 分割数据
                train_data = sensor_data[train_indices]
                test_data = sensor_data[test_indices]
                
                # 保存原始分割数据
                train_channels_raw.append(train_data)
                test_channels_raw.append(test_data)
                
                # 归一化数据
                train_norm, test_norm = normalize_sensor_data(train_data, test_data)
                
                # 保存归一化数据
                train_channels_norm.append(train_norm)
                test_channels_norm.append(test_norm)
            
            # === 第3步: 堆叠所有传感器数据为3D数组 ===
            X_train_raw = np.stack(train_channels_raw, axis=1).astype(np.float64)
            X_test_raw = np.stack(test_channels_raw, axis=1).astype(np.float64)
            X_train_norm = np.stack(train_channels_norm, axis=1).astype(np.float64)
            X_test_norm = np.stack(test_channels_norm, axis=1).astype(np.float64)
            
            print(f"  数据形状 - 训练集: {X_train_raw.shape}, 测试集: {X_test_raw.shape}")
            
            # === 第4步: 保存原始数据到 results_raw 文件夹 ===
            np.save(os.path.join(output_dir_raw, f"{experiment_name}_train_x.npy"), X_train_raw)
            np.save(os.path.join(output_dir_raw, f"{experiment_name}_train_y.npy"), y_train)
            np.save(os.path.join(output_dir_raw, f"{experiment_name}_test_x.npy"), X_test_raw)
            np.save(os.path.join(output_dir_raw, f"{experiment_name}_test_y.npy"), y_test)
            print(f"  ✓ 已保存原始数据到: {output_dir_raw}")
            
            # === 第5步: 保存归一化数据到 results_normalized 文件夹 ===
            np.save(os.path.join(output_dir_norm, f"{experiment_name}_train_x.npy"), X_train_norm)
            np.save(os.path.join(output_dir_norm, f"{experiment_name}_train_y.npy"), y_train)
            np.save(os.path.join(output_dir_norm, f"{experiment_name}_test_x.npy"), X_test_norm)
            np.save(os.path.join(output_dir_norm, f"{experiment_name}_test_y.npy"), y_test)
            print(f"  ✓ 已保存归一化数据到: {output_dir_norm}")
    
    print("\n=== 数据准备完成! ===")

def main():
    """主函数"""
    print("=" * 60)
    print("传感器数据预处理 - 先分割 → 归一化 → 堆叠")
    print("=" * 60)
    
    # 1. 加载标签
    original_labels, gas_type_labels = load_labels()
    
    # 2. 获取每个浓度的样本索引
    concentration_indices = get_concentration_samples(original_labels)
    
    # 3. 为每个浓度划分训练集和测试集 (50:50)
    print("\n=== 划分训练集和测试集 (50:50) ===")
    train_test_indices = split_train_test(concentration_indices)
    
    # 4. 准备并保存数据（先分割，再归一化，最后堆叠）
    prepare_and_save_data(gas_type_labels, train_test_indices, concentration_indices)
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n原始数据保存至: {OUTPUT_PATH_RAW}")
    print(f"归一化数据保存至: {OUTPUT_PATH_NORM}")
    print("\n每个文件夹下包含 36 个实验子文件夹 (con1con1Sensor ~ con6con6Sensor)")
    print("每个子文件夹包含 4 个文件:")
    print("  - *_train_x.npy (训练集特征, 形状: (n_samples, 6, 361))")
    print("  - *_train_y.npy (训练集标签, 形状: (n_samples,))")
    print("  - *_test_x.npy (测试集特征, 形状: (n_samples, 6, 361))")
    print("  - *_test_y.npy (测试集标签, 形状: (n_samples,))")

if __name__ == "__main__":
    main()
