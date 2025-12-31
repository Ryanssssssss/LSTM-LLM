"""改进版LLMFew，使用离线存储的prompt embeddings"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from CasualCNN import CausalCNNEncoder
from Embed import PatchEmbedding
from LLMprepare import LLMprepare
from prompt_handler import PromptHandler
import time


class LLMFew(nn.Module):
    def __init__(self, configs):
        super(LLMFew, self).__init__()
        self.configs = configs
        self.num_class = configs.num_class
        self.length = configs.length
        self.dimensions = configs.dimensions
        self.llm_type = configs.llm_type
        self.lora = configs.lora
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.llm_model, self.d_model = LLMprepare(configs)
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, configs.dropout)
        self.patch_nums = int((self.length - self.patch_len) / self.stride + 2)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # 初始化 prompt 处理器 - 只用于加载离线存储的embeddings
        self.prompt_handler = PromptHandler(
            tokenizer_path=configs.tokenizer_path if hasattr(configs, 'tokenizer_path')
            else "FacebookAI/roberta-base",
            llm_path=configs.llm_path if hasattr(configs, 'llm_path')
            else "FacebookAI/roberta-base",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            max_length=getattr(configs, 'prompt_max_length', 768),
            representation=getattr(configs, 'prompt_representation', 'sequence')
        )
        self.prompt_representation = self.prompt_handler.representation

        # 使用离线embeddings的标志
        self.use_offline_embeddings = configs.use_offline_embeddings if hasattr(configs,
                                                                                'use_offline_embeddings') else True
        self.dataset_name = configs.dataset  # 用于确定加载哪个数据集的embeddings

        self.encoder = CausalCNNEncoder(
            in_channels=self.dimensions,
            channels=configs.channels,
            depth=configs.depth,
            reduced_size=configs.reduced_size,
            out_channels=self.d_model,
            kernel_size=configs.kernel_size,
            use_channel_attention=configs.use_channel_attention,
            channel_attention_type=configs.channel_attention_type,
            attention_position=configs.attention_position,
            attention_kernel_size=configs.attention_kernel_size
        )

        # 添加维度调整层
        total_cnn_feats = self.d_model * self.patch_nums
        self.dim_adjust = nn.Linear(total_cnn_feats, self.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.relu = nn.LeakyReLU()
        self.act = F.relu
        self.ln_proj = nn.LayerNorm(self.d_model)
        self.mapping = nn.Sequential(
            nn.Linear(self.d_model, self.num_class),
            nn.Dropout(configs.dropout))
        self.activation = nn.Softmax(dim=1)

        # ✨ Gated Fusion 模块
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )

    def forward(self, x, indices=None):
        # x: batch, dim(dimensions), seq(length)
        # indices: 样本在原始数据集中的索引，用于加载对应的embeddings
        x = x.contiguous()
        B, L, M = x.shape

        # 从离线存储中加载prompt embeddings
        if self.use_offline_embeddings and indices is not None:
            # 使用当前批次的样本索引加载对应的embeddings
            is_training = self.training  # 根据模型当前模式判断加载训练集还是测试集
            prompt_embeddings = self.prompt_handler.load_embeddings_by_indices(
                self.dataset_name, indices, is_training=is_training)
            # prompt_embeddings = prompt_embeddings.to(device=x.device, dtype=x.dtype)
            prompt_embeddings = prompt_embeddings.float().to(device=x.device)
        else:
            return

        # padding patch layer
        input_x = self.padding_patch_layer(x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        input_x = rearrange(input_x, 'b m n p-> (b n) m p')

        # 通过编码器
        output_x = self.encoder(input_x)
        output_x = self.dropout(output_x)

        # 重塑并调整维度
        output_x = rearrange(output_x, '(b n) o -> b (n o)', b=B)
        output_x = self.dim_adjust(output_x)
        output_x = output_x.unsqueeze(1)  # [B, 1, d_model]

        # ✨ Gated Fusion: 融合 prompt 和时间序列特征
        # 若 prompt_embeddings 在离线阶段已压缩为单 token，确保其维度为 [B, 1, d_model]
        if prompt_embeddings.dim() == 2:
            prompt_embeddings = prompt_embeddings.unsqueeze(1)

        output_x_residual = output_x.clone()

        llm_dtype = next(self.llm_model.parameters()).dtype

        # 计算门控系数
        # 拼接两个模态的特征用于计算门控权重
        concat_feats = torch.cat([prompt_embeddings, output_x], dim=-1)  # [B, 1, 2*d_model]
        gate = self.fusion_gate(concat_feats)  # [B, 1, d_model]

        # ✅ 将所有融合操作统一转换为 LLM 的 dtype（bfloat16）
        prompt_embeddings_llm = prompt_embeddings.to(dtype=llm_dtype)
        output_x_llm = output_x.to(dtype=llm_dtype)
        gate_llm = gate.to(dtype=llm_dtype)

        # 门控融合：动态加权组合两个模态（确保计算在正确的 dtype 下进行）
        fused_embeds = gate_llm * prompt_embeddings_llm + (1 - gate_llm) * output_x_llm  # [B, 1, d_model]

        # 输入 LLM
        llm_out = self.llm_model(inputs_embeds=fused_embeds.contiguous()).last_hidden_state

        # 提取输出（现在只有1个token）
        time_series_out = llm_out  # [B, 1, d_model]

        time_series_out_f32 = time_series_out.float()

        # ✅ 改进：添加数值稳定性检查和裁剪
        # 检查 LLM 输出是否包含异常值
        if torch.isnan(time_series_out_f32).any() or torch.isinf(time_series_out_f32).any():
            # 使用安全的默认值
            time_series_out_f32 = torch.nan_to_num(time_series_out_f32, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 最终输出（保留残差连接以增强梯度流）
        outputs = self.relu(time_series_out_f32.squeeze(1) + output_x_residual.squeeze(1))
        
        # ✅ 添加数值裁剪，防止 LayerNorm 输入过大
        outputs = torch.clamp(outputs, min=-50, max=50)
        
        outputs = self.ln_proj(outputs)
        outputs = self.mapping(outputs)
        
        # ✅ 最终输出裁剪，确保进入 CrossEntropyLoss 的值在安全范围内
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        # outputs = self.activation(outputs)
        return outputs 


