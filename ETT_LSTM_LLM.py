"""
ETDataset电力负荷预测：LSTM-LLM混合模型（ProLLM架构版本）
核心改进：将融合特征通过GPT-2 forward传播，利用Transformer进行语义推理
参考ProLLM架构，使用LLM的动态处理能力而非静态embeddings
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import GPT2Model, GPT2Config
import warnings
import logging
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置 ====================
LOOK_BACK = 96          # 历史窗口：96小时（4天）
N_FUTURE = 24           # 预测窗口：24小时（1天）
TRAIN_RATIO = 0.7       # 改为70/30划分
TEST_RATIO = 0.3

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.4      # 增强正则化：从0.2增加到0.4

BATCH_SIZE = 32
LEARNING_RATE = 0.0003  # 大幅降低学习率：从0.001降至0.0003
WEIGHT_DECAY = 5e-4     # 增强L2正则：从1e-5增加到5e-4
EPOCHS = 50             # 减少epochs防止过拟合
EARLY_STOP_PATIENCE = 8 # 更激进的早停：从15缩短到8
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
logger.info("⚡ ETDataset电力负荷预测 - LSTM-LLM混合模型（ProLLM架构）")
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

# 数据划分（70/30）
train_size = int(len(X) * TRAIN_RATIO)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

logger.info(f"✓ 数据集划分完成")
logger.info(f"  训练集: {len(X_train)}样本 ({TRAIN_RATIO*100:.0f}%)")
logger.info(f"  测试集: {len(X_test)}样本 ({TEST_RATIO*100:.0f}%)")

# ==================== 3. 加载预生成的Embeddings ====================
logger.info("\n" + "="*60)
logger.info("3. 加载预生成的GPT-2 Prompt Embeddings")
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
test_embeddings = load_embeddings(f"{EMBEDDING_DIR}/test")

logger.info(f"✓ Embeddings加载完成")
logger.info(f"  训练集: {train_embeddings.shape}")
logger.info(f"  测试集: {test_embeddings.shape}")

# 验证尺寸匹配
assert len(train_embeddings) == len(X_train), "训练集尺寸不匹配！"
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
test_dataset = ETTDataset(X_test, y_test, test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 5. 模型定义（ProLLM架构） ====================
class LSTMLLM_ETT(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, llm_hidden_size, 
                 output_steps, num_layers=2, dropout=0.2):
        super(LSTMLLM_ETT, self).__init__()
        
        # LSTM分支：编码时序模式
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, 
                           num_layers=num_layers, dropout=dropout, 
                           batch_first=True)
        
        # LLM prompt embeddings投影：768维 → 128维
        self.llm_prompt_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 门控融合单元：决定LSTM与LLM prompt的融合比例
        self.fusion_gate = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.Sigmoid()
        )
        
        # 【核心改进】反向投影层：128维 → 768维（准备输入GPT-2）
        self.to_llm_projector = nn.Sequential(
            nn.Linear(lstm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 【核心改进】加载预训练GPT-2模型（冻结参数）
        logger.info("  正在加载GPT-2模型...")
        config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2', config=config)
        # 冻结GPT-2参数，只训练投影层和融合层
        for param in self.gpt2.parameters():
            param.requires_grad = False
        logger.info("  ✓ GPT-2加载完成（参数已冻结）")
        
        # 预测头：从LLM输出到最终预测
        self.predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_steps)
        )
        
        # 使用Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，避免偏向某一分支"""
        for name, m in self.named_modules():
            # 跳过GPT-2的参数（已预训练）
            if 'gpt2' in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_lstm, x_llm_prompt):
        """
        ProLLM架构的forward流程：
        1. LSTM编码时序特征
        2. LLM prompt降维并通过门控融合
        3. 【关键】融合特征投影到LLM空间，通过GPT-2 forward
        4. GPT-2输出用于最终预测
        
        Args:
            x_lstm: (batch, seq_len, input_size) 时序输入
            x_llm_prompt: (batch, 768) 预生成的prompt embeddings
        """
        batch_size = x_lstm.size(0)
        
        # 1. LSTM编码时序特征
        lstm_out, _ = self.lstm(x_lstm)  # (batch, seq_len, 128)
        lstm_feat = lstm_out[:, -1, :]   # (batch, 128)
        
        # 2. LLM prompt投影到相同维度
        llm_prompt_feat = self.llm_prompt_projector(x_llm_prompt)  # (batch, 128)
        
        # 3. 特征级门控融合
        combined = torch.cat([lstm_feat, llm_prompt_feat], dim=1)  # (batch, 256)
        gate = self.fusion_gate(combined)  # (batch, 128)
        
        # 数值稳定性处理（参考ProLLM）
        gate = torch.clamp(gate, min=0.0, max=1.0)
        gate = torch.nan_to_num(gate, nan=0.5)
        
        # 逐维度加权融合
        fused = gate * lstm_feat + (1 - gate) * llm_prompt_feat  # (batch, 128)
        
        # 数值稳定性检查
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. 【核心改进】投影到LLM维度并通过GPT-2 forward
        fused_llm_input = self.to_llm_projector(fused)  # (batch, 768)
        fused_llm_input = fused_llm_input.unsqueeze(1)  # (batch, 1, 768) 作为单个token
        
        # 通过GPT-2进行语义推理（利用12层Transformer的self-attention）
        gpt2_output = self.gpt2(inputs_embeds=fused_llm_input)
        llm_semantic_feat = gpt2_output.last_hidden_state[:, -1, :]  # (batch, 768)
        
        # 数值稳定性处理
        llm_semantic_feat = torch.nan_to_num(llm_semantic_feat, nan=0.0)
        
        # 5. 预测
        output = self.predictor(llm_semantic_feat)  # (batch, output_steps)
        
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
criterion = nn.MSELoss()
# 添加学习率衰减
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

logger.info("\n" + "="*60)
logger.info("开始训练（70/30划分，增强正则化）")
logger.info("="*60)

best_train_loss = float('inf')
patience_counter = 0
gate_weights_history = []

for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0
    train_preds, train_trues = [], []
    epoch_gate_weights = []
    
    for X_batch, emb_batch, y_batch in train_loader:
        X_batch, emb_batch, y_batch = X_batch.to(device), emb_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred, gate_weight = model(X_batch, emb_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.append(pred.detach().cpu().numpy())
        train_trues.append(y_batch.detach().cpu().numpy())
        epoch_gate_weights.append(gate_weight.mean().item())
    
    train_loss /= len(train_loader)
    train_preds = np.vstack(train_preds)
    train_trues = np.vstack(train_trues)
    train_r2 = r2_score(train_trues.flatten(), train_preds.flatten())
    avg_gate_weight = np.mean(epoch_gate_weights)
    gate_weights_history.append(avg_gate_weight)
    
    # 学习率衰减
    scheduler.step(train_loss)
    
    # 保存最佳模型并早停
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
        save_status = "✓"
    else:
        patience_counter += 1
        save_status = ""
    
    # 每轮都打印
    logger.info(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | R²: {train_r2:.4f} | "
        f"Gate: {avg_gate_weight:.3f} | {save_status}"
    )
    
    # 早停检查
    if patience_counter >= EARLY_STOP_PATIENCE:
        logger.info(f"\n早停触发！连续{EARLY_STOP_PATIENCE}轮无改善")
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
axes[0].set_title(f'ETT电力负荷预测 (LSTM-LLM ProLLM架构) | RMSE={rmse:.4f}°C, R²={r2:.4f}')
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
