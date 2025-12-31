#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_results():
    """加载所有实验结果"""
    csv_file = 'experiment_results/all_results.csv'
    if not os.path.exists(csv_file):
        print("❌ 未找到实验结果文件!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"✅ 加载了 {len(df)} 条实验结果")
    return df

def generate_comprehensive_heatmaps(df):
    """生成综合热力图分析"""
    print("\n=== 生成综合热力图分析 ===")
    
    model_types = df['model_type'].unique()
    
    # 为每种模型类型生成热力图
    for model_type in model_types:
        model_df = df[df['model_type'] == model_type]
        
        # 按seed分组，计算平均准确率和标准差
        stats = model_df.groupby(['source_con', 'target_con'])['best_accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        # 创建热力图数据
        mean_data = stats.pivot(index='source_con', columns='target_con', values='mean')
        std_data = stats.pivot(index='source_con', columns='target_con', values='std')
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 平均准确率热力图
        sns.heatmap(mean_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Accuracy'}, ax=axes[0])
        axes[0].set_title(f'{model_type} - Average Accuracy')
        axes[0].set_xlabel('Target Concentration')
        axes[0].set_ylabel('Source Concentration')
        
        # 标准差热力图
        sns.heatmap(std_data, annot=True, fmt='.4f', cmap='Blues', 
                   cbar_kws={'label': 'Standard Deviation'}, ax=axes[1])
        axes[1].set_title(f'{model_type} - Standard Deviation')
        axes[1].set_xlabel('Target Concentration')
        axes[1].set_ylabel('Source Concentration')
        
        plt.tight_layout()
        
        # 保存热力图
        heatmap_file = f'experiment_results/{model_type}_comprehensive_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 保存热力图: {heatmap_file}")

def generate_model_comparison(df):
    """生成模型对比分析"""
    print("\n=== 生成模型对比分析 ===")
    
    if len(df['model_type'].unique()) < 2:
        print("⚠️  只有一种模型类型，跳过对比分析")
        return
    
    # 计算每个模型的平均性能
    model_performance = df.groupby(['model_type', 'source_con', 'target_con'])['best_accuracy'].mean().reset_index()
    
    # 创建对比热力图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    model_types = df['model_type'].unique()
    
    for i, model_type in enumerate(model_types):
        model_data = model_performance[model_performance['model_type'] == model_type]
        heatmap_data = model_data.pivot(index='source_con', columns='target_con', values='best_accuracy')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Accuracy'}, ax=axes[i])
        axes[i].set_title(f'{model_type}')
        axes[i].set_xlabel('Target Concentration')
        axes[i].set_ylabel('Source Concentration')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存对比图
    comparison_file = 'experiment_results/model_comparison_heatmap.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存模型对比图: {comparison_file}")
    
    # 计算性能差异
    if len(model_types) == 2:
        generate_performance_difference(df, model_types)

def generate_performance_difference(df, model_types):
    """生成性能差异分析"""
    print("\n=== 生成性能差异分析 ===")
    
    # 计算两个模型的平均性能
    model1_data = df[df['model_type'] == model_types[0]].groupby(['source_con', 'target_con'])['best_accuracy'].mean().reset_index()
    model2_data = df[df['model_type'] == model_types[1]].groupby(['source_con', 'target_con'])['best_accuracy'].mean().reset_index()
    
    # 合并数据
    merged = pd.merge(model1_data, model2_data, on=['source_con', 'target_con'], suffixes=('_1', '_2'))
    merged['difference'] = merged['best_accuracy_2'] - merged['best_accuracy_1']
    
    # 创建差异热力图
    diff_data = merged.pivot(index='source_con', columns='target_con', values='difference')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_data, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
               cbar_kws={'label': f'Accuracy Difference\n({model_types[1]} - {model_types[0]})'})
    plt.title(f'Performance Difference: {model_types[1]} vs {model_types[0]}')
    plt.xlabel('Target Concentration')
    plt.ylabel('Source Concentration')
    
    # 保存差异图
    diff_file = 'experiment_results/performance_difference_heatmap.png'
    plt.savefig(diff_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存性能差异图: {diff_file}")
    
    # 打印统计摘要
    print(f"\n📊 性能差异统计:")
    print(f"   平均差异: {merged['difference'].mean():.4f}")
    print(f"   最大提升: {merged['difference'].max():.4f}")
    print(f"   最大下降: {merged['difference'].min():.4f}")
    print(f"   标准差: {merged['difference'].std():.4f}")

def generate_concentration_analysis(df):
    """生成浓度分析"""
    print("\n=== 生成浓度分析 ===")
    
    model_types = df['model_type'].unique()
    
    fig, axes = plt.subplots(len(model_types), 2, figsize=(15, 6*len(model_types)))
    if len(model_types) == 1:
        axes = axes.reshape(1, -1)
    
    for i, model_type in enumerate(model_types):
        model_df = df[df['model_type'] == model_type]
        
        # 源域浓度分析
        source_stats = model_df.groupby('source_con')['best_accuracy'].agg(['mean', 'std']).reset_index()
        axes[i, 0].bar(source_stats['source_con'], source_stats['mean'], 
                      yerr=source_stats['std'], capsize=5, alpha=0.7, color='skyblue')
        axes[i, 0].set_xlabel('Source Concentration')
        axes[i, 0].set_ylabel('Average Accuracy')
        axes[i, 0].set_title(f'{model_type} - Source Domain Performance')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 目标域浓度分析
        target_stats = model_df.groupby('target_con')['best_accuracy'].agg(['mean', 'std']).reset_index()
        axes[i, 1].bar(target_stats['target_con'], target_stats['mean'], 
                      yerr=target_stats['std'], capsize=5, alpha=0.7, color='lightcoral')
        axes[i, 1].set_xlabel('Target Concentration')
        axes[i, 1].set_ylabel('Average Accuracy')
        axes[i, 1].set_title(f'{model_type} - Target Domain Performance')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存浓度分析图
    concentration_file = 'experiment_results/concentration_analysis.png'
    plt.savefig(concentration_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存浓度分析图: {concentration_file}")

def generate_summary_report(df):
    """生成总结报告"""
    print("\n=== 生成总结报告 ===")
    
    report_lines = []
    report_lines.append("# ProLLM 实验结果总结报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 基本统计
    report_lines.append("## 基本统计")
    report_lines.append(f"- 总实验数: {len(df)}")
    report_lines.append(f"- 模型类型: {', '.join(df['model_type'].unique())}")
    report_lines.append(f"- 数据集数量: {len(df['dataset'].unique())}")
    report_lines.append(f"- 随机种子数量: {len(df['seed'].unique())}")
    report_lines.append("")
    
    # 每个模型的性能统计
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        report_lines.append(f"## {model_type} 性能统计")
        report_lines.append(f"- 平均准确率: {model_df['best_accuracy'].mean():.4f} ± {model_df['best_accuracy'].std():.4f}")
        report_lines.append(f"- 最高准确率: {model_df['best_accuracy'].max():.4f}")
        report_lines.append(f"- 最低准确率: {model_df['best_accuracy'].min():.4f}")
        report_lines.append("")
        
        # 浓度分析
        source_stats = model_df.groupby('source_con')['best_accuracy'].mean()
        target_stats = model_df.groupby('target_con')['best_accuracy'].mean()
        
        report_lines.append("### 源域浓度性能")
        for con, acc in source_stats.items():
            report_lines.append(f"- Con{con}: {acc:.4f}")
        report_lines.append("")
        
        report_lines.append("### 目标域浓度性能")
        for con, acc in target_stats.items():
            report_lines.append(f"- Con{con}: {acc:.4f}")
        report_lines.append("")
    
    # 保存报告
    report_file = 'experiment_results/summary_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ 保存总结报告: {report_file}")

def main():
    """主函数"""
    print("🚀 开始分析实验结果...")
    
    # 创建结果目录
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    
    # 加载数据
    df = load_results()
    if df is None:
        return
    
    # 生成各种分析
    generate_comprehensive_heatmaps(df)
    generate_model_comparison(df)
    generate_concentration_analysis(df)
    generate_summary_report(df)
    
    print("\n🎉 结果分析完成!")
    print("📁 所有分析结果保存在 'experiment_results' 目录中")

if __name__ == "__main__":
    main()