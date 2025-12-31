import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定义文件路径
base_path = "/home/wuwujian/LXY/sensor_process/LLM-few/npydata/sensor/"
files = [
    "mq7b_normal.xlsx",
    "mq2_normal.xlsx",
    "mp801_normal.xlsx",
    "mp503_normal.xlsx",
    "mp3b_normal_1.xlsx",
    "mp2_normal.xlsx"
]

# 创建空列表存储各通道数据
data_channels = []

# 读取所有数据
for file in files:
    df = pd.read_excel(base_path + file, header=None)
    
    if file == "mp3b_normal_1.xlsx":
        # 从mp3b文件中获取标签
        labels = df.iloc[:, -1].values
        # 获取特征（前361列）
        features = df.iloc[:, :361].values
    else:
        # 其他文件全部作为特征
        features = df.values
        
    data_channels.append(features)

# 将所有通道的数据堆叠成3D数组 (samples, channels, features)
X = np.stack(data_channels, axis=1).astype(np.float64)
# 将标签转换为要求的格式
y = labels.astype('<U4')

# 划分训练集和测试集，保持标签分布
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 验证数据shape和dtype
print("数据形状:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\n数据类型:")
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)

# 保存为npy文件
np.save(base_path + 'sensor_x_train.npy', X_train)
np.save(base_path + 'sensor_y_train.npy', y_train)
np.save(base_path + 'sensor_x_test.npy', X_test)
np.save(base_path + 'sensor_y_test.npy', y_test)

print("\n文件保存完成！")