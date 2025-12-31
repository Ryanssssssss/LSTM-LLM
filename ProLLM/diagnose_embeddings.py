#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：检查 offline embedding 数据中的 NaN 问题
特别关注 batch 81 和相关索引
"""

import os
import h5py
import numpy as np
import torch
from pathlib import Path
import argparse

def check_single_embedding(file_path, idx, representation='sequence'):
    """检查单个 embedding 文件"""
    issues = []
    
    try:
        with h5py.File(file_path, 'r') as hf:
            if 'embeddings' not in hf:
                issues.append(f"❌ 缺少 'embeddings' 数据集")
                return issues, None
            
            data = hf['embeddings'][:]
            
            # 基本信息
            print(f"  索引 {idx}: 形状={data.shape}, dtype={data.dtype}")
            
            # 检查 NaN
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                issues.append(f"🔴 包含 {nan_count} 个 NaN 值 (总共 {data.size} 个元素, 占比 {nan_count/data.size*100:.2f}%)")
            
            # 检查 Inf
            inf_count = np.isinf(data).sum()
            if inf_count > 0:
                issues.append(f"🔴 包含 {inf_count} 个 Inf 值")
            
            # 检查极端值
            if nan_count == 0 and inf_count == 0:
                data_min, data_max = data.min(), data.max()
                data_mean, data_std = data.mean(), data.std()
                print(f"    值范围: [{data_min:.4f}, {data_max:.4f}]")
                print(f"    均值/标准差: {data_mean:.4f} / {data_std:.4f}")
                
                # 检查异常极端值
                if abs(data_max) > 1000 or abs(data_min) > 1000:
                    issues.append(f"⚠️  存在极端值: min={data_min:.2f}, max={data_max:.2f}")
            
            return issues, data
            
    except Exception as e:
        issues.append(f"❌ 读取文件失败: {str(e)}")
        return issues, None

def diagnose_dataset_embeddings(dataset_name, representation='sequence', batch_size=32):
    """诊断指定数据集的 embeddings"""
    print(f"\n{'='*80}")
    print(f"诊断数据集: {dataset_name}")
    print(f"表示方式: {representation}")
    print(f"{'='*80}")
    
    base_dir = f"embeddings/{dataset_name}"
    if representation != 'sequence':
        base_dir = f"{base_dir}/{representation}"
    
    splits = ['train', 'test']
    all_issues = {}
    
    for split in splits:
        print(f"\n📂 检查 {split} 集:")
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"  ❌ 目录不存在: {split_dir}")
            continue
        
        # 获取所有 embedding 文件
        files = sorted([f for f in os.listdir(split_dir) if f.endswith('.h5')], 
                      key=lambda x: int(x.replace('.h5', '')))
        
        print(f"  找到 {len(files)} 个 embedding 文件")
        
        if len(files) == 0:
            continue
        
        # 检查特定索引（batch 81 相关）
        # batch_size=32, batch_81 包含索引 2592-2623
        batch_81_start = 81 * batch_size
        batch_81_end = batch_81_start + batch_size
        
        print(f"\n  🎯 重点检查 Batch 81 (索引 {batch_81_start}-{batch_81_end-1}):")
        
        batch_81_issues = []
        for idx in range(batch_81_start, min(batch_81_end, len(files))):
            file_path = os.path.join(split_dir, f"{idx}.h5")
            if os.path.exists(file_path):
                issues, data = check_single_embedding(file_path, idx, representation)
                if issues:
                    batch_81_issues.append((idx, issues))
                    print(f"    {'  '.join(issues)}")
        
        # 随机抽查其他样本
        print(f"\n  📊 随机抽查其他样本:")
        sample_indices = np.random.choice(len(files), min(10, len(files)), replace=False)
        sample_indices = [idx for idx in sample_indices if idx < batch_81_start or idx >= batch_81_end]
        
        other_issues = []
        for idx in sorted(sample_indices):
            file_path = os.path.join(split_dir, f"{idx}.h5")
            if os.path.exists(file_path):
                issues, data = check_single_embedding(file_path, idx, representation)
                if issues:
                    other_issues.append((idx, issues))
                    print(f"    {'  '.join(issues)}")
        
        # 统计汇总
        print(f"\n  📈 {split} 集统计:")
        print(f"    Batch 81 问题数: {len(batch_81_issues)}")
        print(f"    其他样本问题数: {len(other_issues)} / {len(sample_indices)} (抽样)")
        
        all_issues[split] = {
            'batch_81': batch_81_issues,
            'others': other_issues
        }
    
    return all_issues

def check_raw_data(dataset_name, batch_size=32):
    """检查原始输入数据是否有问题"""
    print(f"\n{'='*80}")
    print(f"检查原始数据: {dataset_name}")
    print(f"{'='*80}")
    
    train_path = f"npydata/{dataset_name}/{dataset_name}_train_x.npy"
    test_path = f"npydata/{dataset_name}/{dataset_name}_test_x.npy"
    
    for split, path in [('train', train_path), ('test', test_path)]:
        print(f"\n📂 {split} 集:")
        
        if not os.path.exists(path):
            print(f"  ❌ 文件不存在: {path}")
            continue
        
        data = np.load(path)
        print(f"  形状: {data.shape}")
        print(f"  dtype: {data.dtype}")
        
        # 检查 NaN/Inf
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        print(f"  NaN 数量: {nan_count}")
        print(f"  Inf 数量: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print(f"  值范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  均值/标准差: {data.mean():.4f} / {data.std():.4f}")
        
        # 检查 batch 81
        batch_81_start = 81 * batch_size
        batch_81_end = min(batch_81_start + batch_size, len(data))
        
        if batch_81_start < len(data):
            print(f"\n  🎯 Batch 81 数据 (索引 {batch_81_start}-{batch_81_end-1}):")
            batch_data = data[batch_81_start:batch_81_end]
            batch_nan = np.isnan(batch_data).sum()
            batch_inf = np.isinf(batch_data).sum()
            
            print(f"    形状: {batch_data.shape}")
            print(f"    NaN 数量: {batch_nan}")
            print(f"    Inf 数量: {batch_inf}")
            
            if batch_nan == 0 and batch_inf == 0:
                print(f"    值范围: [{batch_data.min():.4f}, {batch_data.max():.4f}]")
                print(f"    均值/标准差: {batch_data.mean():.4f} / {batch_data.std():.4f}")
            else:
                print(f"    🔴 原始数据中 Batch 81 就有问题!")

def main():
    parser = argparse.ArgumentParser(description="诊断 offline embedding 数据")
    parser.add_argument('--dataset', type=str, default='Sensor', help='数据集名称')
    parser.add_argument('--representation', type=str, default='pooled_last_token', 
                       choices=['sequence', 'pooled_last_token'],
                       help='Embedding 表示方式')
    parser.add_argument('--batch_size', type=int, default=32, help='训练时使用的 batch size')
    parser.add_argument('--check_raw', action='store_true', help='是否检查原始数据')
    
    args = parser.parse_args()
    
    print(f"\n🔍 Offline Embedding 诊断工具")
    print(f"数据集: {args.dataset}")
    print(f"表示方式: {args.representation}")
    print(f"Batch Size: {args.batch_size}")
    
    # 检查原始数据
    if args.check_raw:
        check_raw_data(args.dataset, args.batch_size)
    
    # 检查 embeddings
    issues = diagnose_dataset_embeddings(args.dataset, args.representation, args.batch_size)
    
    # 汇总报告
    print(f"\n{'='*80}")
    print(f"📝 诊断报告汇总")
    print(f"{'='*80}")
    
    total_issues = 0
    for split, split_issues in issues.items():
        batch_81_count = len(split_issues['batch_81'])
        others_count = len(split_issues['others'])
        total = batch_81_count + others_count
        total_issues += total
        
        print(f"\n{split} 集:")
        print(f"  Batch 81 问题: {batch_81_count}")
        print(f"  其他问题: {others_count}")
        print(f"  总计: {total}")
        
        if batch_81_count > 0:
            print(f"\n  🔴 Batch 81 有问题的索引:")
            for idx, issue_list in split_issues['batch_81']:
                print(f"    - 索引 {idx}: {issue_list[0]}")
    
    if total_issues == 0:
        print(f"\n✅ 未发现问题!")
    else:
        print(f"\n⚠️  发现 {total_issues} 个问题，需要重新生成 embeddings!")
        print(f"\n💡 建议:")
        print(f"  1. 检查原始数据是否有 NaN/Inf")
        print(f"  2. 检查 prompt_handler.py 中的 generate_prompt() 是否产生异常")
        print(f"  3. 重新运行 offline_embedding_generator.py 生成 embeddings")

if __name__ == "__main__":
    main()
