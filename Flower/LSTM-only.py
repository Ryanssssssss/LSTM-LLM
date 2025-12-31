"""
消融实验：纯LSTM模型（无LLM）
用于对比LSTM-LLM混合模型的效果提升
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置（可调整） ====================
# 时序窗口设置
LOOK_BACK = 48          # 历史时间步数（输入）
N_FUTURE = 36            # 预测未来步数（输出）

# 数据划分比例
TRAIN_RATIO = 0.7       # 训练集比例
VAL_RATIO = 0.1         # 验证集比例
TEST_RATIO = 0.2        # 测试集比例

# LSTM模型结构（更深更宽以匹配混合模型容量）
LSTM_HIDDEN_SIZE = 256  # LSTM隐藏层维度
LSTM_NUM_LAYERS = 3     # LSTM层数
LSTM_DROPOUT = 0.3      # LSTM Dropout率

# 训练超参数
BATCH_SIZE = 32         # 批次大小
LEARNING_RATE = 0.001   # 初始学习率
WEIGHT_DECAY = 5e-5     # 权重衰减（L2正则化）
EPOCHS = 150            # 最大训练轮数
EARLY_STOP_PATIENCE = 30  # 早停耐心值（验证集loss不降的轮数）
GRAD_CLIP_NORM = 1.0    # 梯度裁剪阈值

# 其他配置
RANDOM_SEED = 42        # 随机种子
DATA_FILE = "data/数据列表（20240317~20240505）.xlsx"  # 数据文件路径
MODEL_SAVE_PATH = "checkpoints/best_lstm_only_model.pth"  # 模型保存路径
# ==================== ⚙️ 配置结束 ====================

# 配置日志系统
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
log_filename = f"logs/lstm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")
logger.info(f"日志文件: {log_filename}")
logger.info("\n" + "="*60)
logger.info("⚙️  超参数配置（纯LSTM消融实验）")
logger.info("="*60)
logger.info(f"历史窗口: {LOOK_BACK}步 ({LOOK_BACK*20}分钟) | 预测窗口: {N_FUTURE}步 ({N_FUTURE*20}分钟)")
logger.info(f"数据划分: 训练{TRAIN_RATIO*100}% | 验证{VAL_RATIO*100}% | 测试{TEST_RATIO*100}%")
logger.info(f"LSTM结构: {LSTM_NUM_LAYERS}层 × {LSTM_HIDDEN_SIZE}维 (Dropout={LSTM_DROPOUT})")
logger.info(f"训练参数: Batch={BATCH_SIZE} | LR={LEARNING_RATE} | Epochs={EPOCHS} | 早停={EARLY_STOP_PATIENCE}")

# ==================== 1. 数据加载与预处理 ====================
logger.info("="*60)
logger.info("1. 数据加载与预处理")
logger.info("="*60)

df = pd.read_excel(DATA_FILE)
df = df.iloc[::-1].reset_index(drop=True)
df = df.loc[:,['土壤温度','空气温度','空气湿度']]

for i in range(len(df)):
    df.iloc[i,0] = float(df.iloc[i,0][:-1])
    df.iloc[i,1] = float(df.iloc[i,1][:-1])
    df.iloc[i,2] = float(df.iloc[i,2][:-1])

logger.info(f"数据形状: {df.shape}")
logger.info(f"数据统计:\n{df.describe()}")

if df.isnull().sum().sum() > 0:
    logger.info(f"⚠️ 发现缺失值: {df.isnull().sum()}")
    df = df.fillna(method='ffill').fillna(method='bfill')
    logger.info("已使用前向/后向填充处理缺失值")

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)
logger.info(f"归一化后数据范围: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")

# ==================== 2. 时序数据构建 ====================
logger.info("\n" + "="*60)
logger.info("2. 时序数据构建")
logger.info("="*60)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

look_back = LOOK_BACK
n_future = N_FUTURE

supervised_data = series_to_supervised(scaled_data, n_in=look_back, n_out=n_future)
supervised_data = supervised_data.reset_index(drop=True)

input_cols = [col for col in supervised_data.columns if '(t-' in col or col == 'var1(t)']
output_cols = [col for col in supervised_data.columns if 'var1(t+' in col or col == 'var1(t)']

X = supervised_data[input_cols].values
y = supervised_data[output_cols].values

logger.info(f"输入特征形状 X: {X.shape}")
logger.info(f"输出标签形状 y: {y.shape}")

# ==================== 3. 数据集划分 ====================
logger.info("\n" + "="*60)
logger.info("3. 数据集划分")
logger.info("="*60)

train_size = int(len(X) * TRAIN_RATIO)
val_size = int(len(X) * VAL_RATIO)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

logger.info(f"训练集: X_train {X_train.shape}, y_train {y_train.shape}")
logger.info(f"验证集: X_val {X_val.shape}, y_val {y_val.shape}")
logger.info(f"测试集: X_test {X_test.shape}, y_test {y_test.shape}")

# ==================== 4. 纯LSTM模型架构 ====================
logger.info("\n" + "="*60)
logger.info("4. 纯LSTM模型架构")
logger.info("="*60)

class PureLSTMModel(nn.Module):
    """纯LSTM模型（无LLM组件）"""
    def __init__(self, input_size, hidden_size, num_layers, output_steps, dropout=0.2):
        super(PureLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 预测头（更深的网络以弥补无LLM的损失）
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_seq):
        """
        x_seq: (batch, 31) - 需要reshape为 (batch, 11, 3)
        """
        batch_size = x_seq.shape[0]
        
        # Reshape为LSTM输入格式
        x_reshaped = []
        for i in range(batch_size):
            sample_steps = []
            for step in range(10):
                start_idx = step * 3
                sample_steps.append(x_seq[i, start_idx:start_idx+3].unsqueeze(0))
            # 最后一步：土壤温度 + 补0
            last_step = torch.cat([
                x_seq[i, -1:],
                torch.zeros(2, device=x_seq.device)
            ]).unsqueeze(0)
            sample_steps.append(last_step)
            x_reshaped.append(torch.cat(sample_steps, dim=0))
        
        x_reshaped = torch.stack(x_reshaped)  # (batch, 11, 3)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
        
        # 取最后一层的隐状态
        features = h_n[-1]  # (batch, hidden_size)
        features = self.dropout(features)
        
        # 多步预测
        predictions = self.predictor(features)  # (batch, output_steps)
        
        return predictions

# 初始化模型
model = PureLSTMModel(
    input_size=3,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_NUM_LAYERS,
    output_steps=n_future,
    dropout=LSTM_DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"总参数量: {total_params:,}")
logger.info(f"可训练参数量: {trainable_params:,}")

# ==================== 5. 训练配置 ====================
logger.info("\n" + "="*60)
logger.info("5. 训练配置")
logger.info("="*60)

# 转换为Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
criterion = nn.MSELoss()

logger.info(f"批次大小: {BATCH_SIZE}")
logger.info(f"训练轮数: {EPOCHS}")
logger.info(f"学习率: {LEARNING_RATE}")

# ==================== 6. 训练循环 ====================
logger.info("\n" + "="*60)
logger.info("6. 开始训练（纯LSTM）")
logger.info("="*60)

os.makedirs('checkpoints', exist_ok=True)
model_save_path = MODEL_SAVE_PATH

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_start_time = time.time()
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_time = time.time() - train_start_time
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_time = time.time() - val_start_time
    epoch_time = time.time() - epoch_start_time
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        save_status = "✓ 已保存"
        patience_counter = 0
    else:
        save_status = f"未保存 (patience: {patience_counter + 1}/{EARLY_STOP_PATIENCE})"
        patience_counter += 1
    
    logger.info(
        f"Epoch [{epoch+1:3d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"Time: {epoch_time:.2f}s (Train: {train_time:.2f}s, Val: {val_time:.2f}s) | "
        f"{save_status}"
    )
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        logger.info(f"Early stopping at epoch {epoch+1}")
        break

# ==================== 7. 测试评估 ====================
logger.info("\n" + "="*60)
logger.info("7. 测试集评估")
logger.info("="*60)

model.load_state_dict(torch.load(model_save_path))
logger.info(f"✓ 已加载最佳模型: {model_save_path}")
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_x)
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

predictions = np.vstack(all_predictions)
targets = np.vstack(all_targets)

# 反归一化到原始尺度
predictions_real = np.zeros_like(predictions)
targets_real = np.zeros_like(targets)

for step in range(n_future):
    step_pred = np.hstack([
        predictions[:, step:step+1],
        np.zeros((predictions.shape[0], 2))
    ])
    step_target = np.hstack([
        targets[:, step:step+1],
        np.zeros((targets.shape[0], 2))
    ])
    
    predictions_real[:, step] = scaler.inverse_transform(step_pred)[:, 0]
    targets_real[:, step] = scaler.inverse_transform(step_target)[:, 0]

# 计算指标
mse = mean_squared_error(targets_real, predictions_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_real, predictions_real)
r2 = r2_score(targets_real.flatten(), predictions_real.flatten())

logger.info(f"测试集结果（纯LSTM）:")
logger.info(f"  MSE:  {mse:.4f}")
logger.info(f"  RMSE: {rmse:.4f}")
logger.info(f"  MAE:  {mae:.4f}")
logger.info(f"  R²:   {r2:.4f}")

# ==================== 8. 保存训练日志 ====================
logger.info("\n" + "="*60)
logger.info("8. 保存训练日志")
logger.info("="*60)

import json

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_json_filename = f"logs/lstm_only_{timestamp}.json"

log_data = {
    'experiment_type': 'LSTM-only (Ablation Study)',
    'timestamp': timestamp,
    'hyperparameters': {
        'look_back': look_back,
        'n_future': n_future,
        'batch_size': BATCH_SIZE,
        'epochs': len(train_losses),
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'lstm_hidden_size': 256,
        'lstm_num_layers': 3,
        'dropout': 0.2
    },
    'data_split': {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2
    },
    'training_history': {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'best_train_loss': float(min(train_losses)),
        'best_val_loss': float(min(val_losses))
    },
    'test_results': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    },
    'model_info': {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_path': model_save_path
    }
}

with open(log_json_filename, 'w', encoding='utf-8') as f:
    json.dump(log_data, f, indent=2, ensure_ascii=False)

logger.info(f"训练日志已保存至: {log_json_filename}")

# ==================== 9. 输出最终结果 ====================
logger.info("\n" + "="*60)
logger.info("训练完成！最终结果（纯LSTM）")
logger.info("="*60)
logger.info(f"\n【模型信息】")
logger.info(f"  模型类型: 纯LSTM（消融实验）")
logger.info(f"  总参数量: {total_params:,}")
logger.info(f"  可训练参数量: {trainable_params:,}")
logger.info(f"  最佳模型: {model_save_path}")

logger.info(f"\n【训练信息】")
logger.info(f"  训练轮数: {len(train_losses)} epochs")
logger.info(f"  最佳训练Loss: {min(train_losses):.6f} (Epoch {train_losses.index(min(train_losses))+1})")
logger.info(f"  最佳验证Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses))+1})")

logger.info(f"\n【测试集结果】")
logger.info(f"  MSE:  {mse:.6f}")
logger.info(f"  RMSE: {rmse:.6f}℃")
logger.info(f"  MAE:  {mae:.6f}℃")
logger.info(f"  R²:   {r2:.6f}")

logger.info(f"\n【各步预测MAE】")
for step in range(n_future):
    step_mae = mean_absolute_error(targets_real[:, step], predictions_real[:, step])
    logger.info(f"  Step {step+1}: {step_mae:.6f}℃")

logger.info("\n" + "="*60)
logger.info("💡 提示：与 LSTM-LLM 混合模型对比以评估LLM组件的贡献")
logger.info("="*60)
