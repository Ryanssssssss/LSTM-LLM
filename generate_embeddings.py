"""
🌟 离线生成Prompt Embeddings（测试修改）
参考ProLLM架构，预先计算所有样本的GPT-2 embeddings，避免训练时重复编码
"""
import numpy as np
import pandas as pd
import torch
import os
import h5py
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 当前使用设备: {device}")

# ==================== 1. 数据加载（与主训练脚本一致） ====================
print("="*80)
print("1. 数据加载与预处理")
print("="*80)

df = pd.read_excel("data/数据列表（20240317~20240505）.xlsx")
df = df.iloc[::-1].reset_index(drop=True)
df = df.loc[:,['土壤温度','空气温度','空气湿度']]

for i in range(len(df)):
    df.iloc[i,0] = float(df.iloc[i,0][:-1])
    df.iloc[i,1] = float(df.iloc[i,1][:-1])
    df.iloc[i,2] = float(df.iloc[i,2][:-1])

print(f"数据形状: {df.shape}")

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# ==================== 2. 时序数据构建 ====================
print("\n" + "="*60)
print("2. 时序数据构建")
print("="*60)

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

look_back = 10
n_future = 6

supervised_data = series_to_supervised(scaled_data, n_in=look_back, n_out=n_future)
supervised_data = supervised_data.reset_index(drop=True)

input_cols = [col for col in supervised_data.columns if '(t-' in col or col == 'var1(t)']
X = supervised_data[input_cols].values

print(f"输入特征形状 X: {X.shape}")

# ==================== 3. 数据集划分（与主脚本一致） ====================
print("\n" + "="*60)
print("3. 数据集划分")
print("="*60)

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ==================== 4. Prompt生成器 ====================
print("\n" + "="*60)
print("4. 初始化Prompt生成器")
print("="*60)

class PromptGenerator:
    """生成描述性prompt"""
    def __init__(self, feature_names, look_back=10):
        self.feature_names = feature_names
        self.look_back = look_back
    
    def generate_prompt(self, x_batch, scaler):
        batch_size = x_batch.shape[0]
        prompts = []
        
        for i in range(batch_size):
            sample_features = []
            for step in range(self.look_back):
                start_idx = step * 3
                sample_features.append(x_batch[i, start_idx:start_idx+3])
            sample = np.array(sample_features)
            
            sample_real = scaler.inverse_transform(sample)
            
            soil_temp = sample_real[:, 0]
            air_temp = sample_real[:, 1]
            humidity = sample_real[:, 2]
            
            soil_trend = np.polyfit(np.arange(len(soil_temp)), soil_temp, 1)[0]
            air_trend = np.polyfit(np.arange(len(air_temp)), air_temp, 1)[0]
            
            prompt = (
                f"<|greenhouse_prediction|>地理位置：广东省云浮市（北纬22°22′-23°19′，东经111°03′-112°31′）。"
                f"历史数据：过去{self.look_back}个时间步的环境监测数据。"
                f"土壤温度范围：{soil_temp.min():.1f}℃-{soil_temp.max():.1f}℃，平均{soil_temp.mean():.1f}℃，"
                f"变化趋势：{'升温' if soil_trend > 0 else '降温'}{abs(soil_trend):.2f}℃/步。"
                f"空气温度范围：{air_temp.min():.1f}℃-{air_temp.max():.1f}℃，平均{air_temp.mean():.1f}℃，"
                f"变化趋势：{'升温' if air_trend > 0 else '降温'}{abs(air_trend):.2f}℃/步。"
                f"空气湿度范围：{humidity.min():.1f}%-{humidity.max():.1f}%，平均{humidity.mean():.1f}%。"
                f"任务：基于以上信息，预测未来{n_future}步（120分钟）的土壤温度变化。<|endoftext|>"
            )
            prompts.append(prompt)
        
        return prompts

feature_names = ['土壤温度', '空气温度', '空气湿度']
prompt_gen = PromptGenerator(feature_names, look_back=look_back)

# ==================== 5. 加载GPT-2模型 ====================
print("\n" + "="*60)
print("5. 加载GPT-2模型")
print("="*60)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2Model.from_pretrained('gpt2').to(device)
gpt2_model.eval()

print("✓ GPT-2模型加载完成 - 测试版本")

# ==================== 6. 生成并保存Embeddings ====================
print("\n" + "="*60)
print("6. 生成Prompt Embeddings")
print("="*60)

def generate_embeddings(X_data, split_name, batch_size=32):
    """
    为数据集生成embeddings并保存为h5文件
    """
    save_dir = f"embeddings/greenhouse_soil/{split_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = len(X_data)
    print(f"\n处理 {split_name} 集: {num_samples} 样本")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size), 
                              desc=f"生成{split_name}集embeddings"):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = X_data[start_idx:end_idx]
            
            # 生成prompts
            prompts = prompt_gen.generate_prompt(batch_x, scaler)
            
            # Tokenize
            encoded = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # 获取GPT-2 embeddings
            outputs = gpt2_model(**encoded)
            embeddings = outputs.last_hidden_state  # (batch, seq_len, 768)
            
            # 使用最后一个token的embedding（pooled）
            attention_mask = encoded['attention_mask']
            lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效token的位置
            
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                last_token_idx = lengths[i].item()
                embedding = embeddings[i, last_token_idx, :]  # (768,)
                
                # 保存为h5文件
                h5_path = os.path.join(save_dir, f"{global_idx}.h5")
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset('embedding', 
                                    data=embedding.cpu().numpy().astype('float32'))
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"✓ {split_name}集embeddings已保存至 {save_dir}/")

# 生成所有数据集的embeddings (测试修改：调整batch_size)
generate_embeddings(X_train, 'train', batch_size=16)
generate_embeddings(X_val, 'val', batch_size=16)
generate_embeddings(X_test, 'test', batch_size=16)

# ==================== 7. 验证生成的embeddings ====================
print("\n" + "="*60)
print("7. 验证Embeddings")
print("="*60)

def verify_embeddings(split_name, expected_count):
    save_dir = f"embeddings/greenhouse_soil/{split_name}"
    files = [f for f in os.listdir(save_dir) if f.endswith('.h5')]
    
    print(f"{split_name}: 生成了 {len(files)} 个文件（预期 {expected_count}）")
    
    if len(files) > 0:
        # 检查第一个文件
        sample_path = os.path.join(save_dir, files[0])
        with h5py.File(sample_path, 'r') as hf:
            emb = hf['embedding'][:]
            print(f"  样本embedding形状: {emb.shape}, dtype: {emb.dtype}")

verify_embeddings('train', len(X_train))
verify_embeddings('val', len(X_val))
verify_embeddings('test', len(X_test))

print("\n" + "="*60)
print("✓ Embedding生成完成！(测试运行)")
print("="*60)
# print(f"日志文件: {log_filename}")  # 测试：注释掉未定义的变量
