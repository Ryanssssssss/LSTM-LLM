#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 mp3b_no_normal.xlsx 的文字标签转换为数字标签
"""

import os
import pandas as pd
import numpy as np

# 获取当前脚本所在目录
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 输入输出文件路径
input_file = os.path.join(BASE_PATH, "mp3b_no_normal.xlsx")
output_file = os.path.join(BASE_PATH, "mp3b_no_normal_1.xlsx")

print(f"读取文件: {input_file}")

# 读取Excel文件
df = pd.read_excel(input_file, header=None)

# 定义标签映射字典（文字 -> 数字）
label_mapping = {
    '乙醇1': 1, '乙醇2': 2, '乙醇3': 3, '乙醇4': 4, '乙醇5': 5, '乙醇6': 6,
    '丙酮1': 7, '丙酮2': 8, '丙酮3': 9, '丙酮4': 10, '丙酮5': 11, '丙酮6': 12,
    '甲醛1': 13, '甲醛2': 14, '甲醛3': 15, '甲醛4': 16, '甲醛5': 17, '甲醛6': 18,
    '甲苯1': 19, '甲苯2': 20, '甲苯3': 21, '甲苯4': 22, '甲苯5': 23, '甲苯6': 24
}

# 获取最后一列的标签
labels = df.iloc[:, -1]

print(f"\n原始标签类型: {labels.dtype}")
print(f"原始标签的唯一值:\n{labels.unique()}")

# 使用映射转换标签
y = labels.map(label_mapping)

print(f"\n转换后标签的唯一值:\n{sorted(y.unique())}")

# 将转换后的标签替换回DataFrame的最后一列
df.iloc[:, -1] = y

# 保存修改后的文件
df.to_excel(output_file, index=False, header=False)

print(f"\n✓ 数据已保存到: {output_file}")
print(f"✓ 标签转换完成，共 {len(df)} 行数据")
