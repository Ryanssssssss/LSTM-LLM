"""预先计算并缓存 ProLLM 使用的 prompt 嵌入。

支持两种表示：
1. "sequence"：保留完整的 token 级嵌入 [max_length, d_model]（与原实现一致）。
2. "pooled_last_token"：仿照 timeCMA，将每个 prompt 压缩成单个向量（最后一个有效 token）。

根据选择的表示，嵌入将存储在不同的子目录中：
    embeddings/<dataset>/<split> 或 embeddings/<dataset>/<representation>/<split>
"""
import argparse
import os
from typing import List

import h5py
import numpy as np
import torch
from tqdm import tqdm

from prompt_handler import PromptHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute pooled prompt embeddings for ProLLM datasets")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="gpt2",
        help="Tokenizer path used by PromptHandler"
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default="gpt2",
        help="Backbone model path used by PromptHandler"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device label for embedding generation (cuda / cpu)"
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=768,
        help="Maximum token length for prompt tokenization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when processing samples"
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["sequence", "pooled_last_token"],
        default="pooled_last_token",
        help="Store full token sequences or single pooled vectors"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names under npydata to process; process all if omitted"
    )
    return parser.parse_args()


def get_all_dataset_names(base_dir: str = "npydata") -> List[str]:
    if not os.path.isdir(base_dir):
        print(f"警告: 数据目录 '{base_dir}' 不存在。")
        return []
    return [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]


def ensure_output_dir(dataset: str, split: str, representation: str) -> str:
    if representation == "sequence":
        out_dir = os.path.join("embeddings", dataset, split)
    else:
        out_dir = os.path.join("embeddings", dataset, representation, split)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def generate_for_split(
    prompt_handler: PromptHandler,
    dataset: str,
    split: str,
    data_x: np.ndarray,
    batch_size: int,
    representation: str
) -> None:
    output_dir = ensure_output_dir(dataset, split, representation)
    num_samples = data_x.shape[0]

    print(f"  -> {split}: {num_samples} samples")

    tensor_x = torch.from_numpy(data_x).float()

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc=f"    {split} batches", leave=False):
            end = min(start + batch_size, num_samples)
            batch = tensor_x[start:end]

            prompts = prompt_handler.generate_prompt(batch)
            if representation == "sequence":
                embeddings = prompt_handler.embed_prompt_timemixxer(prompts)
            else:
                embeddings, mask = prompt_handler.embed_prompt_timemixxer(prompts, return_mask=True)
                lengths = mask.sum(dim=1).clamp(min=1)
                last_indices = (lengths - 1).long().view(-1, 1, 1)
                gather_idx = last_indices.expand(-1, 1, embeddings.size(-1))
                embeddings = torch.gather(embeddings, 1, gather_idx)  # [B,1,D]

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

            if torch.cuda.is_available() and prompt_handler.device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset_names = get_all_dataset_names()
    if args.datasets:
        missing = [name for name in args.datasets if name not in dataset_names]
        if missing:
            raise ValueError(f"数据集未找到: {', '.join(missing)}")
        dataset_names = args.datasets

    if not dataset_names:
        print("在 'npydata' 下未找到任何数据集，退出。")
        return

    print(f"将处理以下数据集: {', '.join(dataset_names)}")

    prompt_handler = PromptHandler(
        tokenizer_path=args.tokenizer_path,
        llm_path=args.llm_path,
        device=device,
        max_length=args.max_token_length,
        representation=args.representation
    )
    prompt_handler.transformer.eval()

    for dataset in dataset_names:
        print(f"\n处理数据集: {dataset}")
        train_path = f"npydata/{dataset}/{dataset}_train_x.npy"
        test_path = f"npydata/{dataset}/{dataset}_test_x.npy"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"  跳过: 找不到必要数据文件 ({train_path} 或 {test_path})")
            continue

        train_x = np.load(train_path)
        test_x = np.load(test_path)

        generate_for_split(prompt_handler, dataset, "train", train_x, args.batch_size, args.representation)
        generate_for_split(prompt_handler, dataset, "test", test_x, args.batch_size, args.representation)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
