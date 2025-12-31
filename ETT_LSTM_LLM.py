"""
ETDataset电力负荷预测：LSTM-LLM混合模型（离线Embeddings版本）
使用电力变压器数据展示LLM在理解多特征相关性方面的优势
需要先运行 generate_embeddings_ett.py 生成embeddings
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
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置 ====================
LOOK_BACK = 96          # 历史窗口：96小时（4天）
N_FUTURE = 24           # 预测窗口：24小时（1天）
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0

RANDOM_SEED = 42
DATA_FILE = "ETDataset/ETT-small/ETTh1.csv"
EMBEDDING_DIR = "embeddings/ett"  # 预生成的embeddings目录
MODEL_SAVE_PATH = "checkpoints/best_ett_lstm_llm.pth"
# ==================== ⚙️ 配置结束 ====================

# 配置日志
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
log_filename = f"logs/ett_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info("="*60)
logger.info("⚡ ETDataset电力负荷预测 - LSTM-LLM混合模型（离线版）")
logger.info("="*60)
logger.info(f"设备: {device}")
logger.info(f"日志: {log_filename}")
logger.info(f"历史窗口: {LOOK_BACK}小时 | 预测窗口: {N_FUTURE}小时")
logger.info(f"LSTM结构: {LSTM_NUM_LAYERS}层 × {LSTM_HIDDEN_SIZE}维")

# ==================== 1. 数据加载 ====================
logger.info("\n" + "="*60)
logger.info("1. 加载ETDataset数据")
logger.info("="*60)

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
logger.info(f"✓ 数据加载完成: {df.shape[0]}条记录")

# 提取特征（去除日期列）
feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
target_col = 'OT'  # 油温作为预测目标

data = df[feature_cols + [target_col]].values

# 加载预保存的scaler
scaler_mean = np.load(f"{EMBEDDING_DIR}/scaler_mean.npy")
scaler_scale = np.load(f"{EMBEDDING_DIR}/scaler_scale.npy")
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

data_scaled = scaler.transform(data)

logger.info(f"✓ 数据标准化完成（使用预保存的scaler）")

# ==================== 2. 构建时序数据集 ====================
def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i+look_back, :])
        y.append(data[i+look_back:i+look_back+n_future, -1])  # 只预测OT
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, LOOK_BACK, N_FUTURE)
logger.info(f"\n✓ 时序数据构建完成")
logger.info(f"  X shape: {X.shape} (样本数, 时间步, 特征数)")
logger.info(f"  y shape: {y.shape} (样本数, 预测步数)")

# 数据划分
train_size = int(len(X) * TRAIN_RATIO)
val_size = int(len(X) * VAL_RATIO)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

logger.info(f"✓ 数据集划分完成")
logger.info(f"  训练集: {len(X_train)}样本 ({TRAIN_RATIO*100:.0f}%)")
logger.info(f"  验证集: {len(X_val)}样本 ({VAL_RATIO*100:.0f}%)")
logger.info(f"  测试集: {len(X_test)}样本 ({TEST_RATIO*100:.0f}%)")

# ==================== 3. 加载预生成的Embeddings ====================
logger.info("\n" + "="*60)
logger.info("3. 加载预生成的GPT-2 Embeddings")
logger.info("="*60)

def load_embeddings(save_dir):
    """加载分块保存的embeddings"""
    embeddings = []
    i = 0
    while os.path.exists(f"{save_dir}/embeddings_{i}.npy"):
        chunk = np.load(f"{save_dir}/embeddings_{i}.npy")
        embeddings.append(chunk)
        i += 1
    if len(embeddings) == 0:
        raise FileNotFoundError(
            f"未找到embeddings文件！请先运行: python generate_embeddings_ett.py"
        )
    return np.vstack(embeddings)

train_embeddings = load_embeddings(f"{EMBEDDING_DIR}/train")
val_embeddings = load_embeddings(f"{EMBEDDING_DIR}/val")
test_embeddings = load_embeddings(f"{EMBEDDING_DIR}/test")

logger.info(f"✓ Embeddings加载完成")
logger.info(f"  训练集: {train_embeddings.shape}")
logger.info(f"  验证集: {val_embeddings.shape}")
logger.info(f"  测试集: {test_embeddings.shape}")

# 验证尺寸匹配
assert len(train_embeddings) == len(X_train), "训练集尺寸不匹配！"
assert len(val_embeddings) == len(X_val), "验证集尺寸不匹配！"
assert len(test_embeddings) == len(X_test), "测试集尺寸不匹配！"
logger.info(f"✓ 数据尺寸验证通过")

# ==================== 4. 数据集类 ====================
class ETTDataset(Dataset):
    def __init__(self, X, y, embeddings):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.embeddings[idx], self.y[idx]

train_dataset = ETTDataset(X_train, y_train, train_embeddings)
val_dataset = ETTDataset(X_val, y_val, val_embeddings)
test_dataset = ETTDataset(X_test, y_test, test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 5. 模型定义 ====================
class LSTMLLM_ETT(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, llm_hidden_size, 
                 output_steps, num_layers=2, dropout=0.2):
        super(LSTMLLM_ETT, self).__init__()
        
        # LSTM分支：编码时序模式
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, 
                           num_layers=num_layers, dropout=dropout, 
                           batch_first=True)
        
        # LLM投影层：将768维降到128维
        self.llm_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 门控融合单元（改进版）：使用可学习的权衡机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.Sigmoid()
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_steps)
        )
        
        # 使用Xavier初始化来避免梯度问题
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，避免偏向某一分支"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_lstm, x_llm):
        # 1. LSTM编码时序特征
        lstm_out, _ = self.lstm(x_lstm)  # (batch, seq_len, 128)
        lstm_feat = lstm_out[:, -1, :]   # (batch, 128)
        
        # 2. LLM投影到相同维度
        llm_feat = self.llm_projector(x_llm)  # (batch, 128)
        
        # 3. 特征级别的门控融合
        # 计算每个维度的权重（而非全局单一权重）
        combined = torch.cat([lstm_feat, llm_feat], dim=1)  # (batch, 256)
        gate = self.fusion_gate(combined)  # (batch, 128)
        
        # 逐维度加权融合
        fused = gate * lstm_feat + (1 - gate) * llm_feat  # (batch, 128)
        
        # 4. 再次融合原始特征（跳跃连接）
        fused_enhanced = self.fusion_layer(
            torch.cat([fused, lstm_feat], dim=1)
        )  # (batch, 128)
        
        # 5. 预测
        output = self.predictor(fused_enhanced)  # (batch, output_steps)
        
        # 返回平均gate权重作为可解释性指标
        avg_gate = gate.mean(dim=1, keepdim=True)  # (batch, 1)
        return output, avg_gate

model = LSTMLLM_ETT(
    lstm_input_size=7,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    llm_hidden_size=768,
    output_steps=N_FUTURE,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT
).to(device)

logger.info(f"\n✓ 模型初始化完成")
logger.info(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 6. 训练 ====================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

logger.info("\n" + "="*60)
logger.info("开始训练")
logger.info("="*60)

best_val_loss = float('inf')
patience_counter = 0
gate_weights_history = []

for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0
    for X_batch, emb_batch, y_batch in train_loader:
        X_batch, emb_batch, y_batch = X_batch.to(device), emb_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred, gate_weight = model(X_batch, emb_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # 验证
    model.eval()
    val_loss = 0
    epoch_gate_weights = []
    with torch.no_grad():
        for X_batch, emb_batch, y_batch in val_loader:
            X_batch, emb_batch, y_batch = X_batch.to(device), emb_batch.to(device), y_batch.to(device)
            pred, gate_weight = model(X_batch, emb_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
            epoch_gate_weights.append(gate_weight.mean().item())
    
    val_loss /= len(val_loader)
    avg_gate_weight = np.mean(epoch_gate_weights)
    gate_weights_history.append(avg_gate_weight)
    
    scheduler.step(val_loss)
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
        save_status = f"✓ 已保存"
    else:
        patience_counter += 1
        save_status = f"未保存 (patience: {patience_counter}/{EARLY_STOP_PATIENCE})"
    
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Gate权重: {avg_gate_weight:.3f} | "
            f"{save_status}"
        )
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        logger.info(f"\n早停触发，最佳验证loss: {best_val_loss:.6f}")
        break

# ==================== 7. 测试 ====================
logger.info("\n" + "="*60)
logger.info("7. 测试集评估")
logger.info("="*60)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_trues = []
test_gate_weights = []

with torch.no_grad():
    for X_batch, emb_batch, y_batch in test_loader:
        X_batch, emb_batch, y_batch = X_batch.to(device), emb_batch.to(device), y_batch.to(device)
        pred, gate_weight = model(X_batch, emb_batch)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y_batch.cpu().numpy())
        test_gate_weights.append(gate_weight.cpu().numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_trues)
test_gate_weights = np.vstack(test_gate_weights)

# 反标准化（只针对OT列）
ot_scaler = StandardScaler()
ot_scaler.mean_ = scaler.mean_[-1]
ot_scaler.scale_ = scaler.scale_[-1]

y_pred_original = ot_scaler.inverse_transform(y_pred)
y_true_original = ot_scaler.inverse_transform(y_true)

# 计算指标
mse = mean_squared_error(y_true_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_original, y_pred_original)
r2 = r2_score(y_true_original.flatten(), y_pred_original.flatten())

logger.info("="*60)
logger.info("📊 测试集性能指标")
logger.info("="*60)
logger.info(f"  RMSE: {rmse:.4f}°C")
logger.info(f"  MAE:  {mae:.4f}°C")
logger.info(f"  R²:   {r2:.4f}")
logger.info(f"  平均Gate权重: {test_gate_weights.mean():.3f} (0.5表示LSTM与LLM贡献相当)")
logger.info("="*60)

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 预测对比
axes[0].plot(y_true_original[:200, 0], label='真实值', alpha=0.7)
axes[0].plot(y_pred_original[:200, 0], label='预测值', alpha=0.7)
axes[0].set_title(f'ETT电力负荷预测 (LSTM-LLM) | RMSE={rmse:.4f}°C, R²={r2:.4f}')
axes[0].set_xlabel('时间步')
axes[0].set_ylabel('油温 (°C)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gate权重分布
axes[1].hist(test_gate_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(test_gate_weights.mean(), color='red', linestyle='--', 
                label=f'均值={test_gate_weights.mean():.3f}')
axes[1].set_title('特征级Gate权重分布（0.5表示LSTM与LLM贡献相当）')
axes[1].set_xlabel('Gate权重')
axes[1].set_ylabel('频数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/ett_lstm_llm_results.png', dpi=150, bbox_inches='tight')
logger.info(f"\n✓ 结果图保存至: results/ett_lstm_llm_results.png")

logger.info(f"\n🎉 训练完成！最佳模型已保存至: {MODEL_SAVE_PATH}")
