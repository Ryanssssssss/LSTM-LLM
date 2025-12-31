"""
LSTM-LLM混合模型用于温室土壤温度预测
参考ProLLM架构，结合LSTM时序建模和GPT-2语义理解能力
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
from transformers import GPT2Model, GPT2Tokenizer
import warnings
import logging
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# 配置日志系统
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到终端
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")
logger.info(f"日志文件: {log_filename}")

# ==================== 1. 数据加载与预处理 ====================
logger.info("="*60)
logger.info("1. 数据加载与预处理")
logger.info("="*60)

# 读取数据
df = pd.read_excel("data/数据列表（20240317~20240505）.xlsx")
df = df.iloc[::-1].reset_index(drop=True)  # 反转时间序列
df = df.loc[:,['土壤温度','空气温度','空气湿度']]

# 清洗数据：移除温度单位符号
for i in range(len(df)):
    df.iloc[i,0] = float(df.iloc[i,0][:-1])
    df.iloc[i,1] = float(df.iloc[i,1][:-1])
    df.iloc[i,2] = float(df.iloc[i,2][:-1])

logger.info(f"数据形状: {df.shape}")
logger.info(f"数据统计:\n{df.describe()}")

# 检查缺失值
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
    """
    将时序数据转换为监督学习格式
    参数:
        data: 输入数据 (DataFrame或ndarray)
        n_in: 历史时间步数 (look_back)
        n_out: 未来预测步数
        dropnan: 是否删除NaN行
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    
    # 预测序列 (t, t+1, ... t+n)
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

# 参数设置
look_back = 10  # 历史时间步
n_future = 6    # 预测未来6步

# 构建监督学习数据
supervised_data = series_to_supervised(scaled_data, n_in=look_back, n_out=n_future)
supervised_data = supervised_data.reset_index(drop=True)

# 只保留土壤温度的未来预测列（其他特征只作为输入）
# 输入: 所有特征的历史10步 (3*10=30列)
# 输出: 土壤温度的未来6步 (6列)
input_cols = [col for col in supervised_data.columns if '(t-' in col or col == 'var1(t)']
output_cols = [col for col in supervised_data.columns if 'var1(t+' in col or col == 'var1(t)']

X = supervised_data[input_cols].values
y = supervised_data[output_cols].values

logger.info(f"输入特征形状 X: {X.shape}")  # (样本数, 31) - 10步*3特征+当前步土壤温度
logger.info(f"输出标签形状 y: {y.shape}")  # (样本数, 6) - 未来6步土壤温度

# ==================== 3. 数据集划分 ====================
logger.info("\n" + "="*60)
logger.info("3. 数据集划分")
logger.info("="*60)

# 训练集:验证集:测试集 = 7:1:2
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

logger.info(f"训练集: X_train {X_train.shape}, y_train {y_train.shape}")
logger.info(f"验证集: X_val {X_val.shape}, y_val {y_val.shape}")
logger.info(f"测试集: X_test {X_test.shape}, y_test {y_test.shape}")

# ==================== 4. Prompt生成器 ====================
logger.info("\n" + "="*60)
logger.info("4. Prompt生成器")
logger.info("="*60)

class PromptGenerator:
    """生成描述性prompt用于LLM理解"""
    def __init__(self, feature_names, look_back=10):
        self.feature_names = feature_names
        self.look_back = look_back
    
    def generate_prompt(self, x_batch, scaler):
        """
        为批次数据生成prompt
        x_batch: (batch_size, 31) - 包含10步历史+当前步=11步，每步3特征
        """
        batch_size = x_batch.shape[0]
        prompts = []
        
        for i in range(batch_size):
            # 31个特征 = 10步*3特征 + 1步*1特征（只有土壤温度）
            # 重新组织为 (10, 3) 的历史数据矩阵
            sample_features = []
            for step in range(self.look_back):
                start_idx = step * 3
                sample_features.append(x_batch[i, start_idx:start_idx+3])
            sample = np.array(sample_features)
            
            # 反归一化以获得真实值
            sample_real = scaler.inverse_transform(sample)  # 历史10步
            
            # 计算统计特征
            soil_temp = sample_real[:, 0]
            air_temp = sample_real[:, 1]
            humidity = sample_real[:, 2]
            
            soil_trend = np.polyfit(np.arange(len(soil_temp)), soil_temp, 1)[0]
            air_trend = np.polyfit(np.arange(len(air_temp)), air_temp, 1)[0]
            
            # 构建prompt
            prompt = (
                f"<|greenhouse_prediction|>地理位置：广东省云浮市（北纬22°22′-23°19′，东经111°03′-112°31′）。"
                f"历史10步观测：土壤温度从{soil_temp[0]:.2f}℃变化到{soil_temp[-1]:.2f}℃，"
                f"趋势为{soil_trend:.4f}℃/步；"
                f"空气温度从{air_temp[0]:.2f}℃变化到{air_temp[-1]:.2f}℃，"
                f"趋势为{air_trend:.4f}℃/步；"
                f"空气湿度均值{humidity.mean():.2f}%。"
                f"任务：基于亚热带温室气候特征预测未来6步土壤温度。"
            )
            prompts.append(prompt)
        
        return prompts

prompt_gen = PromptGenerator(['土壤温度', '空气温度', '空气湿度'], look_back=look_back)

# 测试prompt生成
sample_prompts = prompt_gen.generate_prompt(X_train[:2], scaler)
logger.info(f"样例Prompt:\n{sample_prompts[0]}\n")

# ==================== 5. 自定义数据集（使用离线embeddings） ====================
import h5py

class TimeSeriesDatasetWithEmbeddings(Dataset):
    """加载时序数据和预生成的embeddings"""
    def __init__(self, X, y, split_name, embedding_dir='embeddings/greenhouse_soil'):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.split_name = split_name
        self.embedding_path = os.path.join(embedding_dir, split_name)
        
        # 检查embeddings是否存在
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(
                f"Embedding目录不存在: {self.embedding_path}\n"
                f"请先运行 generate_embeddings.py 生成embeddings"
            )
        
        logger.info(f"{split_name}集embedding路径: {self.embedding_path}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 读取时序特征
        x = self.X[idx]
        y = self.y[idx]
        
        # 读取预生成的embedding
        h5_path = os.path.join(self.embedding_path, f"{idx}.h5")
        try:
            with h5py.File(h5_path, 'r') as hf:
                embedding = torch.from_numpy(hf['embedding'][:]).float()
        except Exception as e:
            logger.error(f"读取embedding失败 {h5_path}: {e}")
            # 返回零向量作为fallback
            embedding = torch.zeros(768)
        
        return x, y, embedding

# ==================== 6. 模型架构 ====================
logger.info("\n" + "="*60)
logger.info("6. 模型架构构建")
logger.info("="*60)

class LSTMEncoder(nn.Module):
    """LSTM时序编码器"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一层的隐状态
        out = h_n[-1]  # (batch, hidden_size)
        return self.dropout(out)

class LSTMLLMOffline(nn.Module):
    """LSTM-LLM混合模型（使用离线embeddings）"""
    def __init__(self, lstm_input_size, lstm_hidden_size, llm_hidden_size, 
                 output_steps, dropout=0.2):
        super(LSTMLLMOffline, self).__init__()
        
        self.lstm_encoder = LSTMEncoder(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        # 维度对齐
        self.lstm_proj = nn.Linear(lstm_hidden_size, llm_hidden_size)
        
        # 门控融合机制 (参考ProLLM)
        self.fusion_gate = nn.Sequential(
            nn.Linear(llm_hidden_size * 2, llm_hidden_size),
            nn.Sigmoid()
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size // 2, output_steps)
        )
        
        self.layer_norm = nn.LayerNorm(llm_hidden_size)
    
    def forward(self, x_seq, prompt_embeddings):
        """
        x_seq: (batch, 31) - 10步*3特征 + 1步*1特征
        prompt_embeddings: (batch, 768) - 预生成的GPT-2 embeddings
        """
        batch_size = x_seq.shape[0]
        
        # Reshape为LSTM输入格式: (batch, 11, 3)
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
        lstm_features = self.lstm_encoder(x_reshaped)  # (batch, lstm_hidden)
        lstm_features = self.lstm_proj(lstm_features)  # (batch, llm_hidden)
        
        # 直接使用预生成的prompt embeddings
        prompt_features = prompt_embeddings  # (batch, llm_hidden=768)
        
        # 门控融合
        concat_features = torch.cat([lstm_features, prompt_features], dim=-1)
        gate = self.fusion_gate(concat_features)
        
        fused_features = gate * prompt_features + (1 - gate) * lstm_features
        fused_features = self.layer_norm(fused_features)
        
        # 多步预测
        predictions = self.predictor(fused_features)  # (batch, output_steps)
        
        return predictions

# 初始化模型（使用离线embeddings版本）
model = LSTMLLMOffline(
    lstm_input_size=3,  # 3个特征
    lstm_hidden_size=128,
    llm_hidden_size=768,  # GPT-2隐藏层维度
    output_steps=n_future,
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"总参数量: {total_params:,}")
logger.info(f"可训练参数量: {trainable_params:,}")

# ==================== 7. 训练配置 ====================
logger.info("\n" + "="*60)
logger.info("7. 训练配置")
logger.info("="*60)

# 超参数
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# 数据加载器（加载预生成的embeddings）
logger.info("创建数据加载器，使用预生成的embeddings...")
train_dataset = TimeSeriesDatasetWithEmbeddings(X_train, y_train, 'train')
val_dataset = TimeSeriesDatasetWithEmbeddings(X_val, y_val, 'val')
test_dataset = TimeSeriesDatasetWithEmbeddings(X_test, y_test, 'test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
logger.info("✓ 数据加载器创建完成")

# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
criterion = nn.MSELoss()

logger.info(f"批次大小: {BATCH_SIZE}")
logger.info(f"训练轮数: {EPOCHS}")
logger.info(f"学习率: {LEARNING_RATE}")
logger.info(f"优化器: Adam")
logger.info(f"损失函数: MSE")

# ==================== 8. 训练循环 ====================
logger.info("\n" + "="*60)
logger.info("8. 开始训练")
logger.info("="*60)

# 创建checkpoints目录
os.makedirs('checkpoints', exist_ok=True)
model_save_path = 'checkpoints/best_lstm_llm_model.pth'

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

import time

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_start_time = time.time()
    
    for batch_x, batch_y, batch_embeddings in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_embeddings = batch_embeddings.to(device)
        
        # 前向传播（直接使用预生成的embeddings）
        outputs = model(batch_x, batch_embeddings)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        for batch_x, batch_y, batch_embeddings in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_embeddings = batch_embeddings.to(device)
            
            outputs = model(batch_x, batch_embeddings)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_time = time.time() - val_start_time
    epoch_time = time.time() - epoch_start_time
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        save_status = "✓ 已保存"
        patience_counter = 0
    else:
        save_status = f"未保存 (patience: {patience_counter + 1}/{early_stop_patience})"
        patience_counter += 1
    
    # 每轮都打印详细的训练进度
    logger.info(
        f"Epoch [{epoch+1:3d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"Time: {epoch_time:.2f}s (Train: {train_time:.2f}s, Val: {val_time:.2f}s) | "
        f"{save_status}"
    )
    
    if patience_counter >= early_stop_patience:
        logger.info(f"Early stopping at epoch {epoch+1}")
        break

# ==================== 9. 测试评估 ====================
logger.info("\n" + "="*60)
logger.info("9. 测试集评估")
logger.info("="*60)

# 加载最佳模型
model.load_state_dict(torch.load(model_save_path))
logger.info(f"✓ 已加载最佳模型: {model_save_path}")
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y, batch_embeddings in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_embeddings = batch_embeddings.to(device)
        
        outputs = model(batch_x, batch_embeddings)
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

predictions = np.vstack(all_predictions)
targets = np.vstack(all_targets)

# 反归一化到原始尺度
# predictions: (N, 6) - 只有土壤温度的6步预测
# 需要补齐为 (N, 6, 3) 才能正确反归一化每一步
predictions_real = np.zeros_like(predictions)
targets_real = np.zeros_like(targets)

for step in range(n_future):
    # 为每一步构造 (N, 3) 的矩阵：[土壤温度, 0, 0]
    step_pred = np.hstack([
        predictions[:, step:step+1],
        np.zeros((predictions.shape[0], 2))
    ])
    step_target = np.hstack([
        targets[:, step:step+1],
        np.zeros((targets.shape[0], 2))
    ])
    
    # 反归一化后只取第一列（土壤温度）
    predictions_real[:, step] = scaler.inverse_transform(step_pred)[:, 0]
    targets_real[:, step] = scaler.inverse_transform(step_target)[:, 0]

# 计算指标
mse = mean_squared_error(targets_real, predictions_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_real, predictions_real)
r2 = r2_score(targets_real.flatten(), predictions_real.flatten())

logger.info(f"测试集结果:")
logger.info(f"  MSE:  {mse:.4f}")
logger.info(f"  RMSE: {rmse:.4f}")
logger.info(f"  MAE:  {mae:.4f}")
logger.info(f"  R²:   {r2:.4f}")

# ==================== 10. 保存训练日志 ====================
logger.info("\n" + "="*60)
logger.info("10. 保存训练日志")
logger.info("="*60)

import os
import json
from datetime import datetime

# 创建logs目录
os.makedirs('logs', exist_ok=True)

# 生成日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"logs/lstm_llm_{timestamp}.json"

# 构建日志数据
log_data = {
    'timestamp': timestamp,
    'hyperparameters': {
        'look_back': look_back,
        'n_future': n_future,
        'batch_size': BATCH_SIZE,
        'epochs': len(train_losses),
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'lstm_hidden_size': 128,
        'llm_hidden_size': 768,
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
        'best_epoch': train_losses.index(min(train_losses)) + 1,
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

# 保存日志
with open(log_filename, 'w', encoding='utf-8') as f:
    json.dump(log_data, f, indent=2, ensure_ascii=False)

logger.info(f"训练日志已保存至: {log_filename}")

# ==================== 11. 输出最终结果 ====================
logger.info("\n" + "="*60)
logger.info("训练完成！最终结果")
logger.info("="*60)
logger.info(f"\n【模型信息】")
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