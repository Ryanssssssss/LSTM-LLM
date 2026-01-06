"""
为ProLLM数据集生成离线GPT-2 embeddings
适配con_normalized数据格式
"""
import os
import sys
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model

# 添加ProLLM路径
sys.path.insert(0, 'ProLLM')
from prompt_handler import PromptHandler


def generate_embeddings_for_dataset(
    dataset_name: str,
    data_dir: str = "ProLLM/con_normalized",
    device: str = "cpu",
    batch_size: int = 16,
    representation: str = "pooled_last_token"
):
    """为指定数据集生成离线embeddings"""
    
    print(f"\n{'='*60}")
    print(f"生成数据集 {dataset_name} 的离线embeddings")
    print(f"{'='*60}")
    
    # 加载数据
    train_x_path = f"{data_dir}/{dataset_name}/{dataset_name}_train_x.npy"
    test_x_path = f"{data_dir}/{dataset_name}/{dataset_name}_test_x.npy"
    
    if not os.path.exists(train_x_path) or not os.path.exists(test_x_path):
        raise FileNotFoundError(f"数据文件不存在！\n  {train_x_path}\n  {test_x_path}")
    
    train_x = np.load(train_x_path)
    test_x = np.load(test_x_path)
    
    print(f"✓ 数据加载完成")
    print(f"  训练集: {train_x.shape}")
    print(f"  测试集: {test_x.shape}")
    
    # 初始化PromptHandler
    prompt_handler = PromptHandler(
        tokenizer_path="gpt2",
        llm_path="gpt2",
        device=torch.device(device),
        max_length=768,
        representation=representation
    )
    prompt_handler.transformer.eval()
    
    print(f"✓ GPT-2模型加载完成")
    print(f"  表示类型: {representation}")
    print(f"  设备: {device}")
    
    # 处理训练集和测试集
    for split, data_x in [("train", train_x), ("test", test_x)]:
        # 确定输出目录
        if representation == "sequence":
            output_dir = f"ProLLM/embeddings/{dataset_name}/{split}"
        else:
            output_dir = f"ProLLM/embeddings/{dataset_name}/{representation}/{split}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n处理 {split} 集...")
        num_samples = data_x.shape[0]
        tensor_x = torch.from_numpy(data_x).float()
        
        with torch.no_grad():
            for start in tqdm(range(0, num_samples, batch_size), desc=f"  生成embeddings"):
                end = min(start + batch_size, num_samples)
                batch = tensor_x[start:end]
                
                # 生成prompt
                prompts = prompt_handler.generate_prompt(batch)
                
                # 编码prompt
                if representation == "sequence":
                    embeddings = prompt_handler.embed_prompt_timemixxer(prompts)
                else:  # pooled_last_token
                    embeddings, mask = prompt_handler.embed_prompt_timemixxer(prompts, return_mask=True)
                    # 提取最后一个有效token
                    lengths = mask.sum(dim=1).clamp(min=1)
                    last_indices = (lengths - 1).long().view(-1, 1, 1)
                    gather_idx = last_indices.expand(-1, 1, embeddings.size(-1))
                    embeddings = torch.gather(embeddings, 1, gather_idx)  # [B, 1, D]
                
                # 保存每个样本
                for local_idx in range(embeddings.size(0)):
                    global_idx = start + local_idx
                    out_path = os.path.join(output_dir, f"{global_idx}.h5")
                    
                    if os.path.exists(out_path):
                        continue
                    
                    with h5py.File(out_path, "w") as hf:
                        vector = embeddings[local_idx]
                        if representation == "sequence":
                            data = vector.detach().cpu().numpy().astype("float32")
                        else:
                            data = vector.squeeze(0).detach().cpu().numpy().astype("float32")
                        hf.create_dataset("embeddings", data=data)
                
                # 清理显存
                if torch.cuda.is_available() and device == "cuda":
                    torch.cuda.empty_cache()
        
        print(f"  ✓ {split} 集完成，保存至: {output_dir}")
    
    print(f"\n{'='*60}")
    print(f"✓ 数据集 {dataset_name} 的embeddings生成完成！")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="为ProLLM数据集生成GPT-2 embeddings")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="指定单个数据集名称，不指定则处理所有con*Sensor")
    parser.add_argument("--device", type=str, default="cuda", 
                        choices=["cpu", "cuda"], help="计算设备")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="批大小")
    parser.add_argument("--representation", type=str, default="pooled_last_token",
                        choices=["sequence", "pooled_last_token"],
                        help="Embedding表示类型")
    args = parser.parse_args()
    
    # 如果指定了数据集，只处理该数据集
    if args.dataset:
        datasets = [args.dataset]
    else:
        # 处理所有con*Sensor数据集
        data_base = "ProLLM/con_normalized"
        datasets = [d for d in os.listdir(data_base) 
                   if os.path.isdir(os.path.join(data_base, d)) and 'Sensor' in d]
        datasets.sort()
        print(f"\n发现 {len(datasets)} 个数据集:")
        for ds in datasets:
            print(f"  - {ds}")
        print()
    
    for dataset_name in datasets:
        generate_embeddings_for_dataset(
            dataset_name=dataset_name,
            device=args.device,
            batch_size=args.batch_size,
            representation=args.representation
        )


if __name__ == "__main__":
    main()
