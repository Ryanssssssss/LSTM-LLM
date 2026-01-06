import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer
import os
import h5py

class PromptHandler:
    def __init__(
        self,
        tokenizer_path,
        llm_path,
        device='cuda',
        max_length=768,
        representation: str = "pooled_last_token"
    ):
        self.device = device
        self.max_length = max_length
        self.representation = representation
        # 使用 RoBERTa 或通用 Auto 模型
        if 'roberta' in tokenizer_path.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            self.transformer = RobertaModel.from_pretrained(llm_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                clean_up_tokenization_spaces=False
            )
            self.transformer = AutoModel.from_pretrained(
                llm_path,
                ignore_mismatched_sizes=True
            )

        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.transformer.config.pad_token_id = self.transformer.config.eos_token_id

        # 冻结模型参数
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.transformer = self.transformer.to(device)

    def generate_prompt(self, x_enc):
        """
        生成 prompt：
        仅包含每个通道的信号片段、slope、response_time、recovery_time。
        """
        B, N, T = x_enc.size()
        prompts = []
        for b in range(B):
            sample_data = x_enc[b]
            channel_info_list = []
            for n in range(N):
                channel_data = sample_data[n, :].detach()

                # ---- 计算趋势（slope）----
                t = torch.arange(0, len(channel_data), device=channel_data.device, dtype=torch.float32)
                t_mean = torch.mean(t)
                channel_mean = torch.mean(channel_data)
                numerator = torch.sum((t - t_mean) * (channel_data - channel_mean))
                denominator = torch.sum((t - t_mean) ** 2)
                slope = (numerator / denominator).item() if denominator != 0 else 0.0

                # ---- 响应时间 ----
                max_val = torch.max(channel_data).item()
                max_idx = torch.argmax(channel_data).item()
                threshold_90 = 0.9 * max_val
                above_threshold = (channel_data >= threshold_90).nonzero(as_tuple=True)[0]
                if len(above_threshold) > 0:
                    first_90_idx = above_threshold[0].item()
                else:
                    first_90_idx = max_idx
                response_time_value = max(max_idx - 20, 0)

                # ---- 恢复时间 ----
                if T > 140:
                    sub_data = channel_data[140:]
                    min_val_after_140 = torch.min(sub_data).item()
                    min_idx_after_140 = torch.argmin(sub_data).item() + 140
                    recovery_time_value = max(min_idx_after_140 - 140, 0)
                else:
                    recovery_time_value = 0

                # ---- 通道数值预览 ----
                channel_values = channel_data.cpu().numpy().tolist()
                if len(channel_values) > 15:
                    values_preview = (
                        ", ".join([f"{v:.3f}" for v in channel_values[:3]]) +
                        ", ..., " +
                        ", ".join([f"{v:.3f}" for v in channel_values[-3:]])
                    )
                else:
                    values_preview = ", ".join([f"{v:.3f}" for v in channel_values])

                # ---- 构建通道文本（按你指定格式）----
                channel_text = (
                    f"Sensor[{n}]: From [0s] to [360s], the values were {values_preview} every second. "
                    f"The total trend value was {slope:.4f}. "
                    f"(response_time={response_time_value:.1f}s, recovery_time={recovery_time_value:.1f}s)"
                )
                channel_info_list.append(channel_text)

            per_channel_text = " | ".join(channel_info_list)

            # ---- 构建总 prompt（按你指定格式）----
            prompt = (
                f"<|start_prompt|>This sample contains signals from a fixed-type gas sensor array "
                f"You will receive channel-specific raw signals and dynamic responses. {per_channel_text}. "
                f"Task description: The gas types include ethanol, acetone, formaldehyde, and toluene. "
                f"Select the correct gas type among them based on these response characteristics."
            )
            prompts.append(prompt)

        return prompts

    def embed_prompt_timemixxer(self, prompts, return_mask: bool = False):
        """prompt 嵌入与原代码一致，但添加了固定长度处理"""
        encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        tokens = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        with torch.no_grad():
            if hasattr(self.transformer, 'get_input_embeddings'):
                prompt_embeddings = self.transformer.get_input_embeddings()(tokens)
            else:
                outputs = self.transformer(tokens)
                prompt_embeddings = outputs.last_hidden_state

        if return_mask:
            return prompt_embeddings, attention_mask
        return prompt_embeddings

    def _resolve_embedding_dir(self, dataset_name: str, split: str) -> str:
        """根据当前表示类型解析离线 embedding 目录"""
        base_dir = os.path.join('embeddings', 'classify', dataset_name)
        if self.representation == 'sequence':
            default_dir = os.path.join(base_dir, split)
            if os.path.isdir(default_dir):
                return default_dir
            sequence_dir = os.path.join(base_dir, 'sequence', split)
            return sequence_dir
        else:
            return os.path.join(base_dir, self.representation, split)

    def load_embeddings_by_indices(self, dataset_name, indices, is_training=True):
        """
        从逐样本文件中按索引加载 embeddings。

        路径结构：
            embeddings/{dataset_name}/train/{idx}.h5
            embeddings/{dataset_name}/test/{idx}.h5

        文件格式：
            - .h5 文件：数据集名称 'embeddings'，形状 [max_token_length, d_model]
        """
        split = 'train' if is_training else 'test'
        dir_path = self._resolve_embedding_dir(dataset_name, split)

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"Embeddings directory not found: {dir_path}. "
                f"Please run embed_prompt.py to generate per-sample embeddings first."
            )

        # 统一将 indices 转为 Python 列表
        if isinstance(indices, torch.Tensor):
            idx_list = indices.detach().cpu().tolist()
        elif isinstance(indices, np.ndarray):
            idx_list = indices.tolist()
        elif isinstance(indices, (list, tuple)):
            idx_list = list(indices)
        else:
            raise TypeError(f"indices should be Tensor/ndarray/list/tuple, got {type(indices)}")

        batch_tensors = []
        for idx in idx_list:
            file_path = os.path.join(dir_path, f"{idx}.h5")
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Embedding file not found: {file_path}. "
                    f"Ensure offline prompts are generated via embed_prompt.py or offline_embedding_generator.py"
                )
            with h5py.File(file_path, 'r') as hf:
                if 'embeddings' not in hf:
                    raise KeyError(f"'embeddings' dataset not found in file: {file_path}")
                arr = hf['embeddings'][:]

            tensor = torch.from_numpy(arr).to(self.device)
            if self.representation == 'sequence':
                if tensor.dim() != 2:
                    raise ValueError(
                        f"Expected token-level embedding of shape [L, d_model], got {tensor.shape}. "
                        f"Please regenerate embeddings or switch prompt_representation to 'pooled_last_token'."
                    )
                if tensor.shape[0] != self.max_length:
                    raise ValueError(
                        f"Embedding length mismatch for {file_path}: got {tensor.shape[0]}, "
                        f"expected {self.max_length}. Regenerate embeddings with the same max_length."
                    )
            else:
                if tensor.dim() == 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.dim() != 1:
                    raise ValueError(
                        f"Expected pooled embedding with shape [d_model], got {tensor.shape}."
                    )
                tensor = tensor.unsqueeze(0)
            batch_tensors.append(tensor)

        if not batch_tensors:
            raise RuntimeError("No embeddings loaded; check indices and embedding directory.")

        # [B, max_token_length, d_model]
        stacked = torch.stack(batch_tensors, dim=0)
        if self.representation != 'sequence':
            return stacked  # [B, 1, d_model]
        return stacked
    
    def preload_all_embeddings(self, dataset_name, is_training=True):
        """
        预加载所有embeddings到内存，显著加速训练。
        
        Args:
            dataset_name: 数据集名称
            is_training: True加载训练集，False加载测试集
        
        Returns:
            torch.Tensor: (N, 1, d_model) 或 (N, max_length, d_model)
        """
        split = 'train' if is_training else 'test'
        dir_path = self._resolve_embedding_dir(dataset_name, split)
        
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"Embeddings directory not found: {dir_path}. "
                f"Please run embed_prompt.py to generate per-sample embeddings first."
            )
        
        # 获取所有文件并排序
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.h5')],
                      key=lambda x: int(x.replace('.h5', '')))
        
        if not files:
            raise FileNotFoundError(f"No embedding files found in {dir_path}")
        
        print(f"预加载 {split} embeddings: {len(files)} 个样本...")
        
        all_embeddings = []
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            with h5py.File(file_path, 'r') as hf:
                if 'embeddings' not in hf:
                    raise KeyError(f"'embeddings' dataset not found in file: {file_path}")
                arr = hf['embeddings'][:]
            
            tensor = torch.from_numpy(arr)
            
            # 处理维度
            if self.representation == 'sequence':
                if tensor.dim() != 2:
                    raise ValueError(
                        f"Expected token-level embedding of shape [L, d_model], got {tensor.shape}."
                    )
                if tensor.shape[0] != self.max_length:
                    raise ValueError(
                        f"Embedding length mismatch: got {tensor.shape[0]}, expected {self.max_length}."
                    )
            else:  # pooled_last_token
                if tensor.dim() == 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.dim() != 1:
                    raise ValueError(
                        f"Expected pooled embedding with shape [d_model], got {tensor.shape}."
                    )
                tensor = tensor.unsqueeze(0)  # (1, d_model)
            
            all_embeddings.append(tensor)
        
        # 堆叠所有embeddings
        stacked = torch.stack(all_embeddings, dim=0)  # (N, 1, d_model) or (N, L, d_model)
        
        print(f"✓ 预加载完成: {stacked.shape}, 内存占用: {stacked.element_size() * stacked.nelement() / 1024 / 1024:.2f} MB")
        
        return stacked