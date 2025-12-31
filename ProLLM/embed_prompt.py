"""
离线生成和存储 prompt embedding 的脚本
将每个样本分别存为单独的 h5 文件，文件名为该样本在原始数据集中的索引
目录结构：
embeddings/{dataset}/train/{idx}.h5
embeddings/{dataset}/test/{idx}.h5
"""
import os
import argparse
import torch
import numpy as np
import h5py
from tqdm import tqdm
from prompt_handler import PromptHandler
import time
import logging

def setup_logging(args):
    log_dir = 'embedding_logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_filename = os.path.join(log_dir, f"embedding_generation_{args.dataset}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def StringToLabel(y):
    labels = np.unique(y)
    new_label_list = []
    for label in y:
        for position, StringLabel in enumerate(labels):
            if label == StringLabel:
                new_label_list.append(position)
            else:
                continue
    new_label_list = np.array(new_label_list)
    return new_label_list

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save prompt embeddings")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., con1con2_fold1)')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for computation')
    parser.add_argument('--tokenizer_path', type=str, default="FacebookAI/roberta-base", help='Path to the tokenizer model')
    parser.add_argument('--llm_path', type=str, default="FacebookAI/roberta-base", help='Path to the LLM model for embeddings')
    parser.add_argument('--max_token_length', type=int, default=768, help='Maximum token length for all prompts')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for offline embedding generation')
    parser.add_argument('--representation', type=str, choices=['sequence', 'pooled_last_token'], default='sequence',
                        help='Store full token sequences or pooled last-token vectors')
    return parser.parse_args()

def save_single_sample_h5(save_dir_split: str, global_idx: int, embedding: torch.Tensor, representation: str):
    """
    将单个样本的 embedding 保存为 h5 文件
    - save_dir_split: 形如 embeddings/{dataset}/train 或 embeddings/{dataset}/test
    - global_idx: 样本在原始数据集中的索引
    - embedding: [max_token_length, embedding_dim] (已对齐长度)
    """
    file_path = os.path.join(save_dir_split, f"{global_idx}.h5")
    if representation == 'sequence':
        emb_np = embedding.detach().cpu().numpy().astype('float32')
    else:
        emb_np = embedding.squeeze(0).detach().cpu().numpy().astype('float32')
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('embeddings', data=emb_np)

def process_split(logger, prompt_handler, x_array: torch.Tensor, save_dir_split: str,
                  max_token_length: int, batch_size: int, representation: str):
    """
    对一个 split（train/test）的所有样本生成并逐样本保存 embeddings。
    - x_array: torch.Tensor, shape [N, C, T] (原始顺序)
    - save_dir_split: 保存目录（train/test）
    """
    os.makedirs(save_dir_split, exist_ok=True)

    N = x_array.shape[0]
    logger.info(f"Processing {save_dir_split}, total samples: {N}")

    # 先取一个小 batch 来确定 embedding 维度
    sample_bs = min(batch_size, N)
    with torch.no_grad():
        sample_prompts = prompt_handler.generate_prompt(x_array[:sample_bs])
        if representation == 'sequence':
            sample_embeddings = prompt_handler.embed_prompt_timemixxer(sample_prompts)
        else:
            sample_embeddings, sample_mask = prompt_handler.embed_prompt_timemixxer(sample_prompts, return_mask=True)
            lengths = sample_mask.sum(dim=1).clamp(min=1)
            last_indices = (lengths - 1).long().view(-1, 1, 1)
            gather_idx = last_indices.expand(-1, 1, sample_embeddings.size(-1))
            sample_embeddings = torch.gather(sample_embeddings, 1, gather_idx)
    embedding_dim = sample_embeddings.shape[-1]
    logger.info(f"Embedding dim = {embedding_dim}, max_token_length = {max_token_length}")

    # 按批生成，并逐样本写文件
    for i in tqdm(range(0, N, batch_size), desc=f"Generating {os.path.basename(save_dir_split)} embeddings"):
        end_idx = min(i + batch_size, N)
        batch_x = x_array[i:end_idx]  # 保持全局索引 i..end_idx-1

        with torch.no_grad():
            prompts = prompt_handler.generate_prompt(batch_x)
            if representation == 'sequence':
                batch_embeddings = prompt_handler.embed_prompt_timemixxer(prompts)
            else:
                batch_embeddings, mask = prompt_handler.embed_prompt_timemixxer(prompts, return_mask=True)
                lengths = mask.sum(dim=1).clamp(min=1)
                last_indices = (lengths - 1).long().view(-1, 1, 1)
                gather_idx = last_indices.expand(-1, 1, batch_embeddings.size(-1))
                batch_embeddings = torch.gather(batch_embeddings, 1, gather_idx)

        B = batch_embeddings.size(0)
        seq_len = batch_embeddings.size(1)
        cut_len = min(seq_len, max_token_length)

        for b in range(B):
            global_idx = i + b  # 原始索引
            if representation == 'sequence':
                padded = torch.zeros((max_token_length, embedding_dim), device=batch_embeddings.device, dtype=batch_embeddings.dtype)
                padded[:cut_len, :] = batch_embeddings[b, :cut_len, :]
                save_single_sample_h5(save_dir_split, global_idx, padded, representation)
            else:
                save_single_sample_h5(save_dir_split, global_idx, batch_embeddings[b], representation)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Finished saving embeddings to {save_dir_split}")

def main():
    args = parse_args()
    logger = setup_logging(args)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 初始化 PromptHandler
    prompt_handler = PromptHandler(
        tokenizer_path=args.tokenizer_path,
        llm_path=args.llm_path,
        device=device,
        max_length=args.max_token_length,
        representation=args.representation
    )

    # 数据路径
    dataset_path = [
        f'npydata/{args.dataset}/{args.dataset}_train_x.npy',
        f'npydata/{args.dataset}/{args.dataset}_train_y.npy',
        f'npydata/{args.dataset}/{args.dataset}_test_x.npy',
        f'npydata/{args.dataset}/{args.dataset}_test_y.npy'
    ]

    # 检查数据集文件
    for path in dataset_path:
        if not os.path.exists(path):
            logger.error(f"Dataset file not found: {path}")
            return

    # 加载数据
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        X_train = np.load(dataset_path[0])
        y_train = np.load(dataset_path[1])
        X_test  = np.load(dataset_path[2])
        y_test  = np.load(dataset_path[3])

        # 标签转数值（为一致性，虽不影响 embeddings 生成）
        y_train = StringToLabel(y_train)
        y_test  = StringToLabel(y_test)

        # 转为 torch
        x_train = torch.from_numpy(X_train).to(device)
        x_test  = torch.from_numpy(X_test).to(device)

        logger.info(f"Train data shape: {x_train.shape}")
        logger.info(f"Test  data shape: {x_test.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return

    # 目标根目录
    if args.representation == 'sequence':
        save_root = f'embeddings/{args.dataset}'
        train_dir = os.path.join(save_root, 'train')
        test_dir  = os.path.join(save_root, 'test')
    else:
        save_root = os.path.join('embeddings', args.dataset, args.representation)
        train_dir = os.path.join(save_root, 'train')
        test_dir = os.path.join(save_root, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 处理训练集（逐样本文件）
    process_split(
        logger=logger,
        prompt_handler=prompt_handler,
        x_array=x_train,
        save_dir_split=train_dir,
        max_token_length=args.max_token_length,
        batch_size=args.batch_size,
        representation=args.representation
    )

    # 处理测试集（逐样本文件）
    process_split(
        logger=logger,
        prompt_handler=prompt_handler,
        x_array=x_test,
        save_dir_split=test_dir,
        max_token_length=args.max_token_length,
        batch_size=args.batch_size,
        representation=args.representation
    )

    logger.info(f"Successfully generated and saved per-sample embeddings for dataset {args.dataset}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")