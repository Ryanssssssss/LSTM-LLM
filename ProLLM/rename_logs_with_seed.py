#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

def extract_seed_from_log(log_file_path):
    """从LOG文件中提取seed值"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            # 只读前几行就够了，seed通常在开头
            for line in f:
                # 匹配类似 "Using random seed: 3407" 或 "seed: 3407" 的模式
                match = re.search(r'(?:Using random seed|seed):\s*(\d+)', line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"❌ 读取文件 {log_file_path} 时出错: {e}")
    return None

def rename_log_file(log_file_path):
    """重命名LOG文件，在日期时间戳前插入seed信息"""
    # 提取seed
    seed = extract_seed_from_log(log_file_path)
    
    if seed is None:
        print(f"⚠️  无法从文件中提取seed: {log_file_path.name}")
        return False
    
    # 获取原文件名
    original_name = log_file_path.name
    
    # 检查文件名是否已经包含seed
    if re.search(r'_seed\d+_', original_name):
        print(f"✓ 文件名已包含seed，跳过: {original_name}")
        return False
    
    # 匹配日期时间戳模式 (YYYYMMDD_HHMMSS)
    # 例如: 20251102_214756
    match = re.search(r'_(\d{8}_\d{6})(\.log)$', original_name)
    
    if not match:
        print(f"⚠️  文件名格式不符合预期: {original_name}")
        return False
    
    # 构造新文件名：在日期时间戳前插入 _seedXXXX
    timestamp = match.group(1)
    extension = match.group(2)
    prefix = original_name[:match.start(1)]
    
    new_name = f"{prefix}seed{seed}_{timestamp}{extension}"
    new_path = log_file_path.parent / new_name
    
    # 检查新文件名是否已存在
    if new_path.exists():
        print(f"⚠️  目标文件已存在，跳过: {new_name}")
        return False
    
    # 执行重命名
    try:
        log_file_path.rename(new_path)
        print(f"✅ 重命名成功:")
        print(f"   {original_name}")
        print(f"   → {new_name}")
        return True
    except Exception as e:
        print(f"❌ 重命名失败: {e}")
        return False

def main():
    """主函数：批量处理logs目录下的所有.log文件"""
    print("🚀 开始批量重命名LOG文件...\n")
    
    # logs目录路径
    logs_dir = Path(__file__).parent / 'logs'
    
    if not logs_dir.exists():
        print(f"❌ logs目录不存在: {logs_dir}")
        return
    
    # 查找所有.log文件
    log_files = list(logs_dir.glob('*.log'))
    
    if not log_files:
        print(f"⚠️  在 {logs_dir} 中未找到.log文件")
        return
    
    print(f"📁 找到 {len(log_files)} 个LOG文件\n")
    
    # 统计
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 逐个处理
    for log_file in sorted(log_files):
        result = rename_log_file(log_file)
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            fail_count += 1
        print()  # 空行分隔
    
    # 打印统计信息
    print("=" * 60)
    print(f"✅ 成功重命名: {success_count} 个")
    print(f"⏭️  跳过: {skip_count} 个")
    print(f"❌ 失败: {fail_count} 个")
    print(f"📊 总计: {len(log_files)} 个")
    print("=" * 60)

if __name__ == "__main__":
    main()
