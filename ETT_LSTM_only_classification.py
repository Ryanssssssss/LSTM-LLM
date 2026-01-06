"""
纯LSTM分类模型 - ProLLM数据集对照组
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

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2 

BATCH_SIZE = 4  # 对齐ProLLM：16 -> 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 
EPOCHS = 100
GRAD_CLIP_NORM = 1.0

RANDOM_SEED = 42
# ==================== ⚙️ 配置结束 ====================

os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# ==================== 获取数据集列表 ====================
if args.dataset:
    datasets = [args.dataset]
else:
    datasets = [d for d in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, d)) and 'Sensor' in d]
    datasets.sort()
    print(f"\n将训练 {len(datasets)} 个数据集:")
    for ds in datasets:
        print(f"  - {ds}")
    print()

# ==================== 训练每个数据集 ====================
for DATASET_NAME in datasets:
    print("\n" + "="*80)
    print(f"开始训练数据集: {DATASET_NAME}")
    print("="*80)
    
    MODEL_SAVE_PATH = f"checkpoints/best_lstm_only_{DATASET_NAME}.pth"
    log_filename = f"logs/{DATASET_NAME}_lstm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    logger.info(f"⚡ 纯LSTM分类模型 - {DATASET_NAME}")
    logger.info("="*60)
    
    # 数据加载
    train_x_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_train_x.npy"
    train_y_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_train_y.npy"
    test_x_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_test_x.npy"
    test_y_path = f"{DATA_DIR}/{DATASET_NAME}/{DATASET_NAME}_test_y.npy"
    
    X_train = np.load(train_x_path)  # (N, channels, length)
    y_train = np.load(train_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    
    # 将标签映射到0-based索引 (CrossEntropyLoss要求)
    y_train = y_train - 1
    y_test = y_test - 1
    
    num_classes = len(np.unique(y_train))
    
    logger.info(f"✓ 数据加载完成")
    logger.info(f"  训练集: {X_train.shape}")
    logger.info(f"  测试集: {X_test.shape}")
    logger.info(f"  类别数: {num_classes}")
    
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
    
    # 纯LSTM模型
    class PureLSTM_Classification(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
            super(PureLSTM_Classification, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                               dropout=dropout, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            logits = self.classifier(lstm_out[:, -1, :])
            return logits
    
    model = PureLSTM_Classification(
        input_size=X_train.shape[1],
        hidden_size=LSTM_HIDDEN_SIZE,
        num_classes=num_classes,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    # 对齐ProLLM：使用StepLR替代ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8
    )
    
    logger.info("\n开始训练...")
    best_test_acc = 0
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
        else:
            save_status = ""
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} | "
            f"TeAcc: {test_acc:.4f} | F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.2f}s | {save_status}"
        )
        
        # 学习率衰减（对齐ProLLM：在epoch结束时调用）
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
    logger.info("📊 最终性能（纯LSTM）")
    logger.info("="*60)
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  测试F1: {test_f1:.4f}")
    logger.info(f"\n{classification_report(test_trues, test_preds)}")
    
    print(f"\n✅ 数据集 {DATASET_NAME} 训练完成！最佳准确率: {best_test_acc:.4f}\n")

print("\n" + "="*80)
print("🎉 所有数据集训练完成！")
print("="*80)
