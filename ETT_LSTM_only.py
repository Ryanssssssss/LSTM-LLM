"""
ETDataset电力负荷预测：纯LSTM模型（对照组）
用于对比LSTM-LLM的性能提升
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置 ====================
LOOK_BACK = 96
N_FUTURE = 24
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

LSTM_HIDDEN_SIZE = 128  # 与LSTM-LLM保持一致
LSTM_NUM_LAYERS = 2     # 与LSTM-LLM保持一致
LSTM_DROPOUT = 0.2      # 与LSTM-LLM保持一致

BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0

RANDOM_SEED = 42
DATA_FILE = "ETDataset/ETT-small/ETTh1.csv"
MODEL_SAVE_PATH = "checkpoints/best_ett_lstm_only.pth"
# ==================== ⚙️ 配置结束 ====================

os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
log_filename = f"logs/ett_lstm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info("="*60)
logger.info("⚡ ETDataset电力负荷预测 - 纯LSTM模型（对照组）")
logger.info("="*60)
logger.info(f"设备: {device}")

# 数据加载
df = pd.read_csv(DATA_FILE)
feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
target_col = 'OT'
data = df[feature_cols + [target_col]].values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i+look_back, :])
        y.append(data[i+look_back:i+look_back+n_future, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, LOOK_BACK, N_FUTURE)

train_size = int(len(X) * TRAIN_RATIO)
val_size = int(len(X) * VAL_RATIO)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logger.info(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# Dataset
class ETTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ETTDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ETTDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(ETTDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 纯LSTM模型
class PureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, num_layers=3, dropout=0.3):
        super(PureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_steps)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.predictor(lstm_out[:, -1, :])
        return output

model = PureLSTM(
    input_size=7,
    hidden_size=LSTM_HIDDEN_SIZE,
    output_steps=N_FUTURE,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT
).to(device)

logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

logger.info("\n开始训练...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
        save_status = "✓ 已保存"
    else:
        patience_counter += 1
        save_status = f"未保存 ({patience_counter}/{EARLY_STOP_PATIENCE})"
    
    if (epoch + 1) % 5 == 0:
        logger.info(f"Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {save_status}")
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        logger.info(f"早停触发，最佳val loss: {best_val_loss:.6f}")
        break

# 测试
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds, all_trues = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y_batch.cpu().numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_trues)

# 反标准化
ot_scaler = StandardScaler()
ot_scaler.mean_ = scaler.mean_[-1]
ot_scaler.scale_ = scaler.scale_[-1]

y_pred_original = ot_scaler.inverse_transform(y_pred)
y_true_original = ot_scaler.inverse_transform(y_true)

mse = mean_squared_error(y_true_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_original, y_pred_original)
r2 = r2_score(y_true_original.flatten(), y_pred_original.flatten())

logger.info("\n" + "="*60)
logger.info("📊 测试集性能（纯LSTM）")
logger.info("="*60)
logger.info(f"  RMSE: {rmse:.4f}°C")
logger.info(f"  MAE:  {mae:.4f}°C")
logger.info(f"  R²:   {r2:.4f}")
logger.info("="*60)

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(14, 6))
plt.plot(y_true_original[:200, 0], label='真实值', alpha=0.7)
plt.plot(y_pred_original[:200, 0], label='预测值', alpha=0.7)
plt.title(f'ETT电力负荷预测 (纯LSTM) | RMSE={rmse:.4f}°C, R²={r2:.4f}')
plt.xlabel('时间步')
plt.ylabel('油温 (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/ett_lstm_only_results.png', dpi=150)
logger.info(f"\n✓ 结果图保存至: results/ett_lstm_only_results.png")
