"""
为ETDataset预生成GPT-2 Embeddings（离线模式）
只需运行一次，后续训练直接加载
"""
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
LOOK_BACK = 96
N_FUTURE = 24
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DATA_FILE = "ETDataset/ETT-small/ETTh1.csv"
SAVE_DIR = "embeddings/ett"
# ==================== 配置结束 ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建保存目录
os.makedirs(f"{SAVE_DIR}/train", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/val", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/test", exist_ok=True)

# ==================== 1. 数据加载 ====================
print("\n" + "="*60)
print("1. 加载ETDataset数据")
print("="*60)

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
print(f"✓ 数据加载: {df.shape[0]}条记录")

feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
target_col = 'OT'
data = df[feature_cols + [target_col]].values

# 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(f"✓ 数据标准化完成")

# 保存scaler（训练时需要反标准化）
np.save(f"{SAVE_DIR}/scaler_mean.npy", scaler.mean_)
np.save(f"{SAVE_DIR}/scaler_scale.npy", scaler.scale_)
print(f"✓ Scaler已保存至 {SAVE_DIR}/")

# ==================== 2. 构建时序序列 ====================
def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i+look_back, :])
        y.append(data[i+look_back:i+look_back+n_future, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, LOOK_BACK, N_FUTURE)
print(f"\n✓ 时序序列构建完成: X{X.shape}, y{y.shape}")

# 数据划分
train_size = int(len(X) * TRAIN_RATIO)
val_size = int(len(X) * VAL_RATIO)

X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]

print(f"✓ 数据集划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}")

# ==================== 3. 定义Prompt生成函数 ====================
def generate_electricity_prompt(sequence):
    """
    基于模式识别的领域知识检索
    核心思想：识别当前样本的运行模式，然后检索对应的专家经验和物理规律
    """
    hufl = sequence[:, 0]  # 高压侧有功负荷
    hull = sequence[:, 1]  # 高压侧无功负荷
    mufl = sequence[:, 2]  # 中压侧有功负荷
    mull = sequence[:, 3]  # 中压侧无功负荷
    lufl = sequence[:, 4]  # 低压侧有功负荷
    lull = sequence[:, 5]  # 低压侧无功负荷
    ot = sequence[:, 6]    # 油温
    
    # ========== 模式识别（高层次特征，LSTM难以直接学习）==========
    
    # 1. 负荷波动模式识别
    total_load = hufl + mufl + lufl
    load_std = total_load.std()
    load_cv = load_std / (total_load.mean() + 1e-8)  # 变异系数
    
    if load_cv > 0.5:
        volatility_pattern = "剧烈波动型"
        volatility_knowledge = "剧烈波动工况下，油温响应呈现非线性特征，需关注峰值负荷的累积热效应和快速散热能力。"
    elif load_cv > 0.25:
        volatility_pattern = "中度波动型"
        volatility_knowledge = "中度波动工况属于典型城市负荷模式，油温变化具有明显滞后性（2-3小时），可参考热惯性模型。"
    else:
        volatility_pattern = "平稳型"
        volatility_knowledge = "平稳负荷下油温主要受环境温度影响，散热效率稳定，适合线性外推预测。"
    
    # 2. 负荷-温度耦合强度分析
    recent_load_change = total_load[-24:].mean() - total_load[-48:-24].mean()
    recent_temp_change = ot[-24:].mean() - ot[-48:-24].mean()
    
    if abs(recent_load_change) < 0.1 and abs(recent_temp_change) > 0.15:
        coupling_pattern = "温度异常型"
        coupling_knowledge = "负荷稳定但温度异常变化，可能存在散热系统故障或环境突变，需警惕设备异常。"
    elif abs(recent_load_change) > 0.3:
        coupling_pattern = "负荷主导型"
        coupling_knowledge = "负荷大幅变化是温度变化的主要驱动力，遵循铜损与负荷平方成正比的物理规律（P_loss ∝ I²R）。"
    else:
        coupling_pattern = "正常耦合型"
        coupling_knowledge = "负荷与温度呈现正常耦合关系，符合标准热力学模型，预测精度依赖于历史相似模式。"
    
    # 3. 多级负荷协同模式
    high_ratio = hufl.mean() / (total_load.mean() + 1e-8)
    mid_ratio = mufl.mean() / (total_load.mean() + 1e-8)
    low_ratio = lufl.mean() / (total_load.mean() + 1e-8)
    
    if high_ratio > 0.6:
        load_dist_pattern = "高压集中型"
        load_dist_knowledge = "高压侧主导（>60%），铁损占比高，油温对高压负荷变化敏感度约为中低压的2-3倍。"
    elif max(high_ratio, mid_ratio, low_ratio) < 0.45:
        load_dist_pattern = "均衡分布型"
        load_dist_knowledge = "三级负荷均衡分布，热量产生较为分散，整体热平衡稳定性好，适合多元线性回归预测。"
    else:
        load_dist_pattern = "双级主导型"
        load_dist_knowledge = "中高压协同主导，需关注两级负荷的交互影响，叠加效应可能导致温升加速。"
    
    # 4. 功率因数与无功影响
    pf_high = abs(hufl.mean() / (hull.mean() + 1e-8))
    
    if pf_high < 1.5:
        pf_pattern = "低功率因数型"
        pf_knowledge = "功率因数低（PF<0.8），无功电流导致额外铜损约15-25%，油温预测需上调5-10%。"
    elif pf_high > 3.0:
        pf_pattern = "高功率因数型"
        pf_knowledge = "功率因数优秀（PF>0.95），设备运行效率高，温升主要来自有功负荷，损耗计算可简化。"
    else:
        pf_pattern = "正常功率因数型"
        pf_knowledge = "功率因数正常（0.8<PF<0.95），符合电网运行规范，采用标准损耗模型即可。"
    
    # 5. 时间模式识别（周期性）
    load_autocorr_24h = np.corrcoef(total_load[:-24], total_load[24:])[0, 1]
    
    if load_autocorr_24h > 0.7:
        time_pattern = "强周期型"
        time_knowledge = "24小时周期性强（相关系数>0.7），可利用昨日同期数据，适合ARIMA类方法。"
    elif load_autocorr_24h < 0.3:
        time_pattern = "弱周期型"
        time_knowledge = "周期性弱，可能为非工作日或特殊事件，历史模式参考价值有限，需依赖实时趋势。"
    else:
        time_pattern = "中等周期型"
        time_knowledge = "周期性中等，建议结合趋势分析和周期分解方法（如STL）提高预测精度。"
    
    # 6. 温度状态评估
    current_temp = ot[-1]
    temp_percentile = (ot < current_temp).sum() / len(ot)
    
    if temp_percentile > 0.9:
        temp_state = "高温运行区"
        temp_knowledge = "当前温度处于历史高位（>90分位），散热能力接近饱和，温升速率可能加快，注意85°C报警阈值。"
    elif temp_percentile < 0.1:
        temp_state = "低温运行区"
        temp_knowledge = "当前温度处于历史低位（<10分位），设备冷启动或低负荷状态，温升速率遵循指数上升规律。"
    else:
        temp_state = "正常运行区"
        temp_knowledge = "温度处于正常区间，热平衡稳定，预测误差主要来源于负荷波动和环境扰动。"
    
    # ========== 构建结构化Prompt ==========
    prompt = f"""<|electricity_forecasting|>电力变压器油温预测任务。

【运行模式识别】
- 负荷波动特征: {volatility_pattern}
- 负荷-温度耦合: {coupling_pattern}
- 多级负荷分布: {load_dist_pattern}
- 功率因数状态: {pf_pattern}
- 时间周期特征: {time_pattern}
- 当前温度状态: {temp_state}

【领域知识与预测策略】
{volatility_knowledge}
{coupling_knowledge}
{load_dist_knowledge}
{pf_knowledge}
{time_knowledge}
{temp_knowledge}

【物理约束】
热平衡方程: dT/dt = α·ΔP_loss - β·(T - T_amb)
其中α为热容系数，β为散热系数，ΔP_loss为负荷变化引起的损耗变化。
滞后时间常数约为2-4小时，峰值负荷后需持续监测3-6小时。<|endoftext|>"""
    
    return prompt

print("\n" + "="*60)
print("3. 生成Prompts（基于模式识别的领域知识）")
print("="*60)

train_prompts = [generate_electricity_prompt(seq) for seq in tqdm(X_train, desc="训练集")]
val_prompts = [generate_electricity_prompt(seq) for seq in tqdm(X_val, desc="验证集")]
test_prompts = [generate_electricity_prompt(seq) for seq in tqdm(X_test, desc="测试集")]

print(f"\n✓ Prompt生成完成")
print(f"  示例（前300字符）:\n{train_prompts[0][:300]}...")
print(f"\n  包含的模式识别维度：")
print(f"    - 负荷波动特征（平稳/中度/剧烈）")
print(f"    - 负荷-温度耦合模式")
print(f"    - 多级负荷分布模式")
print(f"    - 功率因数状态")
print(f"    - 时间周期性")
print(f"    - 温度状态分区")

# ==================== 4. 生成GPT-2 Embeddings ====================
print("\n" + "="*60)
print("4. 生成GPT-2 Embeddings")
print("="*60)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2Model.from_pretrained('gpt2').to(device)
gpt2_model.eval()

def generate_embeddings_batch(prompts, batch_size=32, desc="Processing"):
    """批量生成embeddings"""
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors='pt', 
                             padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = gpt2_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, -1, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

print("正在生成训练集embeddings...")
train_embeddings = generate_embeddings_batch(train_prompts, desc="训练集")

print("正在生成验证集embeddings...")
val_embeddings = generate_embeddings_batch(val_prompts, desc="验证集")

print("正在生成测试集embeddings...")
test_embeddings = generate_embeddings_batch(test_prompts, desc="测试集")

print(f"\n✓ Embeddings生成完成")
print(f"  Embedding维度: {train_embeddings.shape[1]}")

# ==================== 5. 保存Embeddings ====================
print("\n" + "="*60)
print("5. 保存Embeddings到本地")
print("="*60)

# 分块保存（每1000个样本一个文件）
def save_embeddings_in_chunks(embeddings, save_dir, chunk_size=1000):
    num_chunks = (len(embeddings) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings))
        chunk_data = embeddings[start_idx:end_idx]
        np.save(f"{save_dir}/embeddings_{i}.npy", chunk_data)
    return num_chunks

train_chunks = save_embeddings_in_chunks(train_embeddings, f"{SAVE_DIR}/train")
val_chunks = save_embeddings_in_chunks(val_embeddings, f"{SAVE_DIR}/val")
test_chunks = save_embeddings_in_chunks(test_embeddings, f"{SAVE_DIR}/test")

print(f"✓ 训练集: {train_chunks}个文件保存至 {SAVE_DIR}/train/")
print(f"✓ 验证集: {val_chunks}个文件保存至 {SAVE_DIR}/val/")
print(f"✓ 测试集: {test_chunks}个文件保存至 {SAVE_DIR}/test/")

# 保存元数据
metadata = {
    'look_back': LOOK_BACK,
    'n_future': N_FUTURE,
    'train_size': len(train_embeddings),
    'val_size': len(val_embeddings),
    'test_size': len(test_embeddings),
    'embedding_dim': train_embeddings.shape[1],
    'data_file': DATA_FILE
}

import json
with open(f"{SAVE_DIR}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ 元数据已保存至 {SAVE_DIR}/metadata.json")

print("\n" + "="*60)
print("🎉 ETDataset Embeddings生成完成！")
print("="*60)
print(f"总样本数: {len(train_embeddings) + len(val_embeddings) + len(test_embeddings)}")
print(f"存储路径: {SAVE_DIR}/")
print(f"现在可以运行训练脚本了：python ETT_LSTM_LLM_offline.py")
print("="*60)
