"""
纯LLM分类模型 - 消融实验（只用Prompt Embeddings）
不使用LSTM时序编码，直接用预训练的Prompt Embeddings进行分类

⚠️ 注意：由于使用pooled_last_token模式（单token），移除了RoBERTa层
- Prompt Embeddings已经是RoBERTa编码后的池化结果
- 再输入RoBERTa(单token)会退化为恒等映射，无法学习
- 改为直接在embeddings上构建分类器
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import RobertaModel, RobertaConfig
import warnings
import logging
import os
import sys
import time
from datetime import datetime

# 导入根目录的 PromptHandler
from prompt_handler import PromptHandler

warnings.filterwarnings('ignore')

# ==================== ⚙️ 超参数配置 ====================
import argparse

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="数据集名称，不指定则训练所有con*Sensor")
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="计算设备")
args = parser.parse_args()

DATA_DIR = "ProLLM/con_normalized"

BATCH_SIZE = 4  # 对齐ProLLM：16 -> 4
LEARNING_RATE = 0.001  # 🔥 提高学习率（从0.0001到0.001），因为只训练轻量分类器
WEIGHT_DECAY = 1e-4  
EPOCHS = 50
GRAD_CLIP_NORM = 1.0
DROPOUT = 0.3  # 🔥 增加dropout（从0.1到0.3），防止过拟合

# 早停配置
EARLY_STOP_PATIENCE = 10
EARLY_STOP_THRESHOLD = 0.995

RANDOM_SEED = 42
PROMPT_REPRESENTATION = "pooled_last_token"
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
# 🎯 硬编码指定训练范围（修改循环范围即可）
SOURCE_CONS = range(1, 7)  # 修改这里：range(1, 7) 表示con1-con6
TARGET_CONS = range(1, 7)  # 修改这里：range(1, 7) 表示con1-con6

if args.dataset:
    datasets = [args.dataset]
else:
    # 自动生成数据集列表
    datasets = []
    for src in SOURCE_CONS:
        for tgt in TARGET_CONS:
            datasets.append(f"con{src}con{tgt}Sensor")
    
    print(f"\n🎯 训练范围: {len(datasets)} 个数据集")
    print(f"  源浓度: con{min(SOURCE_CONS)}-con{max(SOURCE_CONS)}")
    print(f"  目标浓度: con{min(TARGET_CONS)}-con{max(TARGET_CONS)}")
    print(f"  数据集列表: {datasets}")
    print()

# ==================== 训练每个数据集 ====================
for DATASET_NAME in datasets:
    print("\n" + "="*80)
    print(f"开始训练数据集: {DATASET_NAME}")
    print("="*80)
    
    MODEL_SAVE_PATH = f"checkpoints/best_llm_only_{DATASET_NAME}.pth"
    log_filename = f"logs/{DATASET_NAME}_llm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    logger.info(f"⚡ 纯LLM分类模型（消融实验）- {DATASET_NAME}")
    logger.info("="*60)
    logger.info(f"设备: {device}")
    logger.info(f"日志: {log_filename}")
    
    # ==================== 1. 数据加载 ====================
    logger.info("\n" + "="*60)
    logger.info("1. 加载ProLLM数据")
    logger.info("="*60)
    
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
    
    # 获取类别数
    num_classes = len(np.unique(y_train))
    
    logger.info(f"✓ 数据加载完成")
    logger.info(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    logger.info(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  通道数: {X_train.shape[1]}, 序列长度: {X_train.shape[2]}")
    
    # ==================== 2. 加载离线Embeddings ====================
    logger.info("\n" + "="*60)
    logger.info("2. 初始化PromptHandler并预加载embeddings")
    logger.info("="*60)
    
    prompt_handler = PromptHandler(
        tokenizer_path="FacebookAI/roberta-base",
        llm_path="FacebookAI/roberta-base",
        device=device,
        max_length=768,
        representation=PROMPT_REPRESENTATION
    )
    
    logger.info(f"✓ PromptHandler初始化完成")
    logger.info(f"  表示类型: {PROMPT_REPRESENTATION}")
    
    # 🚀 预加载所有embeddings到内存（显著加速训练）
    train_embeddings = prompt_handler.preload_all_embeddings(DATASET_NAME, is_training=True)
    test_embeddings = prompt_handler.preload_all_embeddings(DATASET_NAME, is_training=False)
    
    # 移到GPU（如果可用）
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    
    logger.info(f"✓ Embeddings预加载完成")
    logger.info(f"  训练集embeddings: {train_embeddings.shape}")
    logger.info(f"  测试集embeddings: {test_embeddings.shape}")
    
    # ==================== 3. 数据集类 ====================
    class LLMOnlyDataset(Dataset):
        def __init__(self, y, indices):
            # 只需要标签和索引，不需要X（时序数据）
            self.y = torch.LongTensor(y)
            self.indices = torch.LongTensor(indices)
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return self.y[idx], self.indices[idx]
    
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))
    
    train_dataset = LLMOnlyDataset(y_train, train_indices)
    test_dataset = LLMOnlyDataset(y_test, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"\n✓ 数据集构建完成")
    logger.info(f"  训练批次: {len(train_loader)}")
    logger.info(f"  测试批次: {len(test_loader)}")
    
    # ==================== 4. 模型定义 ====================
    class LLMOnly_Classification(nn.Module):
        def __init__(self, num_classes, llm_hidden_size=768, dropout=0.3):
            super(LLMOnly_Classification, self).__init__()
            self.d_model = llm_hidden_size  # 768
            
            logger.info("  构建LLM-Only分类器（直接使用Prompt Embeddings）")
            
            # 🔥 移除RoBERTa层，直接在pooled embeddings上分类
            # 原因：pooled_last_token已经是RoBERTa编码+池化的结果
            # 再输入RoBERTa(单token)会退化为恒等映射
            
            # 投影层：提取判别特征
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.d_model // 2),
                nn.LayerNorm(self.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # 分类头
            self.classifier = nn.Linear(self.d_model // 2, num_classes)
            
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"  ✓ 分类器构建完成（可训练参数：{trainable_params:,}）")
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x_llm_prompt):
            """
            直接使用Prompt Embeddings进行分类
            
            Args:
                x_llm_prompt: (batch, 1, 768) 离线prompt embeddings
            """
            # 1. 确保维度 (batch, 1, 768) -> (batch, 768)
            if x_llm_prompt.dim() == 3:
                x_llm_prompt = x_llm_prompt.squeeze(1)
            
            # 2. 特征提取
            features = self.feature_extractor(x_llm_prompt)  # (batch, 384)
            
            # 3. 分类
            logits = self.classifier(features)  # (batch, num_classes)
            
            return logits
    
    model = LLMOnly_Classification(
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n✓ 模型初始化完成")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # ==================== 5. 训练 ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()
    # 对齐ProLLM：使用StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8
    )
    
    logger.info("\n" + "="*60)
    logger.info("开始训练（LLM-Only分类任务）")
    logger.info(f"  学习率: {LEARNING_RATE}")
    logger.info(f"  早停策略: 准确率>{EARLY_STOP_THRESHOLD:.3f}且连续{EARLY_STOP_PATIENCE}轮不提升")
    logger.info("="*60)
    
    best_test_acc = 0
    no_improve_count = 0  # 早停计数器
    training_start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        # 训练
        model.train()
        train_loss = 0
        train_preds, train_trues = [], []
        
        for y_batch, indices_batch in train_loader:
            y_batch = y_batch.to(device)
            
            # 🚀 直接索引预加载的embeddings（超快！）
            embeddings = train_embeddings[indices_batch]  # (batch, 1, d_model)
            
            optimizer.zero_grad()
            logits = model(embeddings)
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
            for y_batch, indices_batch in test_loader:
                y_batch = y_batch.to(device)
                
                # 🚀 直接索引预加载的embeddings
                embeddings = test_embeddings[indices_batch]
                
                logits = model(embeddings)
                test_preds.extend(logits.argmax(dim=1).cpu().numpy())
                test_trues.extend(y_batch.cpu().numpy())
        
        test_acc = accuracy_score(test_trues, test_preds)
        test_f1 = f1_score(test_trues, test_preds, average='weighted')
        
        # 保存最佳模型 & 早停逻辑
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_status = "✓"
            no_improve_count = 0  # 重置计数器
        else:
            save_status = ""
            # 只有达到阈值后才开始计数
            if test_acc >= EARLY_STOP_THRESHOLD:
                no_improve_count += 1
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} | "
            f"TeAcc: {test_acc:.4f} | F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.2f}s | {save_status}"
        )
        
        # 🎯 早停检查1: 达到完美准确率，直接停止
        if test_acc >= 1.0:
            logger.info(f"\n🎉 完美准确率达成！")
            logger.info(f"  测试准确率: {test_acc:.4f} (100%)")
            logger.info(f"  无需继续训练，提前结束 (Epoch {epoch+1}/{EPOCHS})")
            break
        
        # 🎯 早停检查2: 高准确率但连续不提升
        if test_acc >= EARLY_STOP_THRESHOLD and no_improve_count >= EARLY_STOP_PATIENCE:
            logger.info(f"\n⚡ 早停触发！")
            logger.info(f"  当前准确率: {test_acc:.4f} (>{EARLY_STOP_THRESHOLD:.3f})")
            logger.info(f"  连续 {no_improve_count} 个epoch无提升")
            logger.info(f"  最佳准确率: {best_test_acc:.4f}")
            logger.info(f"  提前结束训练 (Epoch {epoch+1}/{EPOCHS})")
            break
        
        # 学习率衰减（对齐ProLLM：在epoch结束时调用）
        scheduler.step()
    
    # ==================== 6. 最终评估 ====================
    total_training_time = time.time() - training_start_time
    logger.info(f"\n✓ 训练完成！总耗时: {total_training_time:.2f}s ({total_training_time/60:.2f}min)")
    
    logger.info("\n" + "="*60)
    logger.info("最终测试集评估")
    logger.info("="*60)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
        for y_batch, indices_batch in test_loader:
            # 🚀 直接索引预加载的embeddings
            embeddings = test_embeddings[indices_batch]
            logits = model(embeddings)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_trues.extend(y_batch.cpu().numpy())
    
    test_acc = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds, average='weighted')

    logger.info(f"\n📊 最终性能（LLM-Only）")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  测试F1: {test_f1:.4f}")
    logger.info(f"\n{classification_report(test_trues, test_preds)}")
    logger.info(f"\n✓ 模型保存至: {MODEL_SAVE_PATH}")
    
    print(f"\n✅ 数据集 {DATASET_NAME} 训练完成！最佳准确率: {best_test_acc:.4f}\n")

print("\n" + "="*80)
print("🎉 所有数据集训练完成！")
print("="*80)
