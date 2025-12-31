"""快速测试ETDataset能否正常加载和处理"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("="*60)
print("🔍 快速测试ETDataset")
print("="*60)

# 1. 加载数据
df = pd.read_csv("ETDataset/ETT-small/ETTh1.csv")
print(f"\n✓ 数据加载成功: {df.shape}")
print(f"  时间范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
print(f"  列名: {df.columns.tolist()}")

# 2. 数据预处理
feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
target_col = 'OT'
data = df[feature_cols + [target_col]].values

print(f"\n✓ 特征提取完成: {data.shape}")
print(f"  数值范围:")
for i, col in enumerate(feature_cols + [target_col]):
    print(f"    {col}: {data[:, i].min():.2f} ~ {data[:, i].max():.2f}")

# 3. 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(f"\n✓ 标准化完成")
print(f"  缩放后范围: {data_scaled.min():.2f} ~ {data_scaled.max():.2f}")

# 4. 创建序列
LOOK_BACK = 96
N_FUTURE = 24

X, y = [], []
for i in range(len(data) - LOOK_BACK - N_FUTURE + 1):
    X.append(data_scaled[i:i+LOOK_BACK, :])
    y.append(data_scaled[i+LOOK_BACK:i+LOOK_BACK+N_FUTURE, -1])

X = np.array(X)
y = np.array(y)

print(f"\n✓ 序列构建完成")
print(f"  X shape: {X.shape} (样本数, 时间步, 特征数)")
print(f"  y shape: {y.shape} (样本数, 预测步数)")

# 5. 测试Prompt生成
def generate_electricity_prompt(sequence):
    hufl = sequence[:, 0]
    ot = sequence[:, 6]
    
    recent_ot = ot[-24:].mean()
    total_load_recent = (sequence[-24:, 0] + sequence[-24:, 2] + sequence[-24:, 4]).mean()
    
    prompt = (
        f"<|electricity_forecasting|>电力变压器运行监测。"
        f"近24小时负荷{total_load_recent:.2f}，油温{recent_ot:.2f}。"
        f"预测未来24小时油温变化。<|endoftext|>"
    )
    return prompt

test_prompt = generate_electricity_prompt(X[0])
print(f"\n✓ Prompt生成测试")
print(f"  示例: {test_prompt}")
print(f"  长度: {len(test_prompt)}字符")

print(f"\n✅ 所有测试通过！数据集可以正常使用")
print(f"   建议配置: LOOK_BACK={LOOK_BACK}, N_FUTURE={N_FUTURE}")
print(f"   总样本数: {len(X)}")
print("="*60)
