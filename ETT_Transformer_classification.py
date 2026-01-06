"""
纯Transformer分类模型 - Benchmark基准模型
使用标准Transformer Encoder进行时序分类
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
import logging
import os
import time
import math
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置 ====================
import argparse

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="数据集名称，不指定则训练所有con*Sensor")
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="计算设备")
args = parser.parse_args()

DATA_DIR = "ProLLM/con_normalized"

# Transformer参数
D_MODEL = 128  # 模型维度
N_HEADS = 8    # 注意力头数（必须能整除d_model）
N_LAYERS = 4   # Transformer层数
DIM_FEEDFORWARD = 512  # FFN维度
DROPOUT = 0.2

BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 
EPOCHS = 100
GRAD_CLIP_NORM = 1.0

# 早停配置
EARLY_STOP_PATIENCE = 15
EARLY_STOP_THRESHOLD = 0.995

RANDOM_SEED = 42
# ==================== ⚙️ 配置结束 ====================

os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# ==================== 获取数据集列表 ====================
SOURCE_CONS = range(1, 7)
TARGET_CONS = range(1, 7)

if args.dataset:
    datasets = [args.dataset]
else:
    datasets = []
    for src in SOURCE_CONS:
        for tgt in TARGET_CONS:
            datasets.append(f"con{src}con{tgt}Sensor")
    
    print(f"\n🎯 训练范围: {len(datasets)} 个数据集")
    print(f"  源浓度: con{min(SOURCE_CONS)}-con{max(SOURCE_CONS)}")
    print(f"  目标浓度: con{min(TARGET_CONS)}-con{max(TARGET_CONS)}")
    print()

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================== 训练每个数据集 ====================
for DATASET_NAME in datasets:
    print("\n" + "="*80)
    print(f"开始训练数据集: {DATASET_NAME}")
    print("="*80)
    
    MODEL_SAVE_PATH = f"checkpoints/best_transformer_{DATASET_NAME}.pth"
    log_filename = f"logs/{DATASET_NAME}_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 重新配置logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info(f"⚡ 纯Transformer分类模型（Benchmark）- {DATASET_NAME}")
    logger.info("="*60)
    logger.info(f"设备: {device}")
    
    # 数据加载
    train_x_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_train_x.npy"
    train_y_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_train_y.npy"
    test_x_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_test_x.npy"
    test_y_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_test_y.npy"
    
    X_train = np.load(train_x_path)  # (N, channels, length)
    y_train = np.load(train_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    
    # 将标签映射到0-based索引
    y_train = y_train - 1
    y_test = y_test - 1
    
    num_classes = len(np.unique(y_train))
    
    logger.info(f"✓ 数据加载完成")
    logger.info(f"  训练集: {X_train.shape}")
    logger.info(f"  测试集: {X_test.shape}")
    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  序列长度: {X_train.shape[2]}")
    
    # Dataset
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            # X: (N, channels, length) -> (N, length, channels)
            self.X = torch.FloatTensor(X).permute(0, 2, 1)
            self.y = torch.LongTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Transformer模型
    class Transformer_Classification(nn.Module):
        def __init__(self, input_size, d_model, n_heads, n_layers, 
                     dim_feedforward, num_classes, dropout=0.3):
            super(Transformer_Classification, self).__init__()
            
            # 输入投影层（将input_size投影到d_model）
            self.input_projection = nn.Linear(input_size, d_model)
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
            
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True  # 🔥 重要：使用batch_first
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=n_layers
            )
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            # x: (batch, seq_len, input_size)
            
            # 1. 投影到d_model维度
            x = self.input_projection(x)  # (batch, seq_len, d_model)
            
            # 2. 添加位置编码
            x = self.pos_encoder(x)
            
            # 3. Transformer编码
            x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
            
            # 4. 全局平均池化（或取最后一个token）
            x = x.mean(dim=1)  # (batch, d_model)
            # 或者取最后一个: x = x[:, -1, :]
            
            # 5. 分类
            logits = self.classifier(x)
            return logits
    
    model = Transformer_Classification(
        input_size=X_train.shape[1],  # channels
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n✓ 模型初始化完成")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  模型维度: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    logger.info("\n" + "="*60)
    logger.info("开始训练（Transformer Benchmark）")
    logger.info(f"  学习率: {LEARNING_RATE}")
    logger.info(f"  早停策略: 准确率>{EARLY_STOP_THRESHOLD:.3f}且连续{EARLY_STOP_PATIENCE}轮不提升")
    logger.info("="*60)
    
    best_test_acc = 0
    no_improve_count = 0
    training_start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_preds, train_trues = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_trues.extend(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_trues, train_preds)
        
        # 测试
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                test_preds.extend(logits.argmax(dim=1).cpu().numpy())
                test_trues.extend(y_batch.cpu().numpy())
        
        test_acc = accuracy_score(test_trues, test_preds)
        test_f1 = f1_score(test_trues, test_preds, average='weighted')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_status = "✓"
            no_improve_count = 0
        else:
            save_status = ""
            if test_acc >= EARLY_STOP_THRESHOLD:
                no_improve_count += 1
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} | "
            f"TeAcc: {test_acc:.4f} | F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.2f}s | {save_status}"
        )
        
        # 早停检查
        if test_acc >= 1.0:
            logger.info(f"\n🎉 完美准确率达成！提前结束 (Epoch {epoch+1}/{EPOCHS})")
            break
        
        if test_acc >= EARLY_STOP_THRESHOLD and no_improve_count >= EARLY_STOP_PATIENCE:
            logger.info(f"\n⚡ 早停触发！连续 {no_improve_count} 个epoch无提升")
            logger.info(f"  提前结束训练 (Epoch {epoch+1}/{EPOCHS})")
            break
        
        scheduler.step()
    
    # 最终评估
    total_training_time = time.time() - training_start_time
    logger.info(f"\n✓ 训练完成！总耗时: {total_training_time:.2f}s ({total_training_time/60:.2f}min)")
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_trues.extend(y_batch.cpu().numpy())
    
    test_acc = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds, average='weighted')
    
    logger.info("\n" + "="*60)
    logger.info("📊 最终性能（Transformer Benchmark）")
    logger.info("="*60)
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  测试F1: {test_f1:.4f}")
    logger.info(f"\n{classification_report(test_trues, test_preds)}")
    
    print(f"\n✅ 数据集 {DATASET_NAME} 训练完成！最佳准确率: {best_test_acc:.4f}\n")

print("\n" + "="*80)
print("🎉 所有数据集训练完成！")
print("="*80)
