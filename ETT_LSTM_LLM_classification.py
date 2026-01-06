"""
LSTM-LLM分类模型 - 使用ProLLM的数据集
保持LSTM-LLM架构，改为分类任务
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

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1  # 对齐ProLLM：0.3 -> 0.1

# Patch参数（学习ProLLM）
PATCH_LEN = 16
STRIDE = 8

BATCH_SIZE = 4  # 对齐ProLLM：16 -> 4
LEARNING_RATE = 0.001
LLM_LEARNING_RATE = 0.0001  # RoBERTa层专用学习率
WEIGHT_DECAY = 1e-4  
EPOCHS = 50
GRAD_CLIP_NORM = 1.0

# 早停配置
EARLY_STOP_PATIENCE = 10  # 测试准确率连续N个epoch不提升则停止
EARLY_STOP_THRESHOLD = 0.995  # 准确率达到此阈值后才开始计数

RANDOM_SEED = 42
USE_OFFLINE_EMBEDDINGS = True
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
    
    MODEL_SAVE_PATH = f"checkpoints/best_lstm_llm_{DATASET_NAME}.pth"
    log_filename = f"logs/{DATASET_NAME}_lstm_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    logger.info(f"⚡ LSTM-LLM分类模型 - {DATASET_NAME}")
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
    class ProLLMDataset(Dataset):
        def __init__(self, X, y, indices):
            # X: (N, channels, length) -> 保持原格式，不permute
            # Patch Embedding需要 (batch, channels, seq_len)
            self.X = torch.FloatTensor(X)  # (N, channels, length)
            self.y = torch.LongTensor(y)
            self.indices = torch.LongTensor(indices)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.indices[idx]
    
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))
    
    train_dataset = ProLLMDataset(X_train, y_train, train_indices)
    test_dataset = ProLLMDataset(X_test, y_test, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"\n✓ 数据集构建完成")
    logger.info(f"  训练批次: {len(train_loader)}")
    logger.info(f"  测试批次: {len(test_loader)}")
    
    # ==================== 4. Patch Embedding（学习ProLLM）====================
    class PatchEmbedding(nn.Module):
        """将时序数据分割成patch并编码"""
        def __init__(self, d_model, patch_len, stride, dropout):
            super(PatchEmbedding, self).__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            
            # 1D卷积编码patch
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.value_embedding = nn.Conv1d(
                in_channels=patch_len, 
                out_channels=d_model,
                kernel_size=3, 
                padding=padding, 
                padding_mode='circular', 
                bias=False
            )
            
            # 位置编码
            self.position_embedding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            """
            Args:
                x: (batch, channels, seq_len)
            Returns:
                embedded patches, num_patches
            """
            n_vars = x.shape[1]
            
            # Padding
            x = self.padding_patch_layer(x)
            
            # Unfold成patches: (batch, channels, num_patches, patch_len)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            
            # 重塑: (batch*channels, num_patches, patch_len)
            B, C, num_patches, patch_len = x.shape
            x = x.reshape(B * C, num_patches, patch_len)
            
            # 卷积编码: (batch*channels, num_patches, patch_len) -> (batch*channels, d_model, num_patches)
            x = x.permute(0, 2, 1)  # (B*C, patch_len, num_patches)
            x = self.value_embedding(x)  # (B*C, d_model, num_patches)
            x = x.permute(0, 2, 1)  # (B*C, num_patches, d_model)
            
            # 位置编码
            x = x + self.position_embedding[:, :num_patches, :]
            
            return self.dropout(x), n_vars, num_patches
    
    # ==================== 5. 模型定义 ====================
    class LSTMLLM_Classification(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, 
                     seq_len, patch_len, stride,
                     llm_hidden_size=768, num_layers=2, dropout=0.3):
            super(LSTMLLM_Classification, self).__init__()
            self.d_model = llm_hidden_size  # 768
            self.seq_len = seq_len
            self.input_size = input_size
            
            # Patch Embedding（学习ProLLM）
            self.patch_embedding = PatchEmbedding(
                d_model=self.d_model,
                patch_len=patch_len,
                stride=stride,
                dropout=dropout
            )
            
            # 计算patch数量
            self.patch_nums = int((seq_len - patch_len) / stride + 2)
            
            # LSTM编码器（处理patch序列）
            self.lstm = nn.LSTM(
                input_size=self.d_model,  # 输入是patch embedding
                hidden_size=llm_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout, 
                batch_first=True
            )
            
            # 维度调整层（需要考虑所有通道）
            total_lstm_feats = input_size * self.patch_nums * llm_hidden_size
            self.dim_adjust = nn.Linear(total_lstm_feats, self.d_model)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.LeakyReLU()
            
            # ✨ 门控融合模块（完全学习ProLLM）
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Sigmoid()
            )
            
            # 加载RoBERTa（学习ProLLM）
            logger.info("  正在加载RoBERTa模型...")
            config = RobertaConfig.from_pretrained('roberta-base')
            # 🚀 使用bfloat16加速（对齐ProLLM）
            self.llm_model = RobertaModel.from_pretrained(
                'roberta-base', 
                config=config,
                torch_dtype=torch.bfloat16
            )
            
            # 冻结大部分层，只解冻最后2层
            for name, param in self.llm_model.named_parameters():
                if 'encoder.layer.10' in name or 'encoder.layer.11' in name or 'pooler' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in self.llm_model.parameters() if p.requires_grad)
            logger.info(f"  ✓ RoBERTa加载完成（解冻最后2层，可训练参数：{trainable_params:,}）")
            
            # LayerNorm + 分类头（完全学习ProLLM）
            self.ln_proj = nn.LayerNorm(self.d_model)
            self.mapping = nn.Sequential(
                nn.Linear(self.d_model, num_classes),
                nn.Dropout(dropout)
            )
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            for name, m in self.named_modules():
                if 'llm_model' in name:
                    continue
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x_input, x_llm_prompt):
            """
            1. Patch Embedding分割时序
            2. LSTM处理patch序列
            3. 门控融合prompt和LSTM特征
            4. 输入RoBERTa
            5. 残差连接 + 分类
            
            Args:
                x_input: (batch, channels, seq_len) 时序输入
                x_llm_prompt: (batch, 1, 768) 离线prompt embeddings
            """
            B = x_input.size(0)
            
            # 1. Patch Embedding
            patch_embeds, n_vars, num_patches = self.patch_embedding(x_input)
            # patch_embeds: (batch*channels, num_patches, d_model)
            
            # 2. LSTM编码每个通道的patch序列
            # 重塑以便LSTM处理
            patch_embeds = patch_embeds.view(B, n_vars, num_patches, self.d_model)
            
            # 对每个通道分别LSTM编码，然后拼接
            lstm_outs = []
            for c in range(n_vars):
                channel_patches = patch_embeds[:, c, :, :]  # (batch, num_patches, d_model)
                lstm_out, _ = self.lstm(channel_patches)  # (batch, num_patches, d_model)
                lstm_outs.append(lstm_out)
            
            # 拼接所有通道
            output_x = torch.stack(lstm_outs, dim=1)  # (batch, channels, num_patches, d_model)
            output_x = output_x.reshape(B, -1)  # (batch, channels*num_patches*d_model)
            
            # 维度调整
            output_x = self.dim_adjust(output_x)  # (batch, d_model)
            output_x = self.dropout(output_x)
            output_x = output_x.unsqueeze(1)  # (batch, 1, d_model)
            
            # 保存残差
            output_x_residual = output_x.clone()
            
            # 3. 确保prompt embeddings维度正确
            if x_llm_prompt.dim() == 2:
                x_llm_prompt = x_llm_prompt.unsqueeze(1)
            prompt_embeddings = x_llm_prompt
            
            # 4. ✨ 门控融合
            concat_feats = torch.cat([prompt_embeddings, output_x], dim=-1)
            gate = self.fusion_gate(concat_feats)
            
            # dtype管理
            llm_dtype = next(self.llm_model.parameters()).dtype
            prompt_embeddings_llm = prompt_embeddings.to(dtype=llm_dtype)
            output_x_llm = output_x.to(dtype=llm_dtype)
            gate_llm = gate.to(dtype=llm_dtype)
            
            # 门控融合
            fused_embeds = gate_llm * prompt_embeddings_llm + (1 - gate_llm) * output_x_llm
            
            # 5. 输入RoBERTa
            llm_out = self.llm_model(inputs_embeds=fused_embeds.contiguous()).last_hidden_state
            time_series_out = llm_out  # (batch, 1, d_model)
            
            # 转回float32
            time_series_out_f32 = time_series_out.float()
            
            # 数值稳定性处理
            if torch.isnan(time_series_out_f32).any() or torch.isinf(time_series_out_f32).any():
                time_series_out_f32 = torch.nan_to_num(time_series_out_f32, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 6. 残差连接 + 激活
            outputs = self.relu(time_series_out_f32.squeeze(1) + output_x_residual.squeeze(1))
            outputs = torch.clamp(outputs, min=-50, max=50)
            
            # 7. LayerNorm + 分类
            outputs = self.ln_proj(outputs)
            logits = self.mapping(outputs)
            
            # 最终输出保护
            logits = torch.clamp(logits, min=-100, max=100)
            
            return logits, gate.mean()
    
    model = LSTMLLM_Classification(
        input_size=X_train.shape[1],  # channels
        hidden_size=LSTM_HIDDEN_SIZE,
        num_classes=num_classes,
        seq_len=X_train.shape[2],  # 序列长度361
        patch_len=PATCH_LEN,
        stride=STRIDE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n✓ 模型初始化完成")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # ==================== 5. 训练 ====================
    # 差异化学习率
    llm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'llm_model' in name:
                llm_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': LEARNING_RATE},
        {'params': llm_params, 'lr': LLM_LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()
    # 对齐ProLLM：使用StepLR替代ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8
    )
    
    logger.info("\n" + "="*60)
    logger.info("开始训练（LSTM-LLM分类任务）")
    logger.info(f"  主学习率: {LEARNING_RATE} | RoBERTa学习率: {LLM_LEARNING_RATE}")
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
        train_gate_weights = []  # 收集门控参数
        
        for X_batch, y_batch, indices_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 🚀 直接索引预加载的embeddings（超快！）
            embeddings = train_embeddings[indices_batch]  # (batch, 1, d_model)
            
            optimizer.zero_grad()
            logits, gate_weight = model(X_batch, embeddings)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_trues.extend(y_batch.cpu().numpy())
            train_gate_weights.append(gate_weight.mean().item())  # 记录平均门控值
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_trues, train_preds)
        avg_gate = np.mean(train_gate_weights)  # 计算epoch平均门控
        
        # 测试
        model.eval()
        test_preds, test_trues = [], []
        
        with torch.no_grad():
            for X_batch, y_batch, indices_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # 🚀 直接索引预加载的embeddings
                embeddings = test_embeddings[indices_batch]
                
                logits, _ = model(X_batch, embeddings)
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
            f"Gate: {avg_gate:.4f} | "
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
        for X_batch, y_batch, indices_batch in test_loader:
            X_batch = X_batch.to(device)
            # 🚀 直接索引预加载的embeddings
            embeddings = test_embeddings[indices_batch]
            logits, _ = model(X_batch, embeddings)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_trues.extend(y_batch.cpu().numpy())
    
    test_acc = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds, average='weighted')

    logger.info(f"\n📊 最终性能")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  测试F1: {test_f1:.4f}")
    logger.info(f"\n{classification_report(test_trues, test_preds)}")
    logger.info(f"\n✓ 模型保存至: {MODEL_SAVE_PATH}")
    
    print(f"\n✅ 数据集 {DATASET_NAME} 训练完成！最佳准确率: {best_test_acc:.4f}\n")

print("\n" + "="*80)
print("🎉 所有数据集训练完成！")
print("="*80)
