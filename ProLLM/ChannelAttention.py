import torch
import torch.nn as nn
import math

class MultiscaleChannelAttention(nn.Module):
    """
    多尺度通道注意力 - 捕获不同时间尺度下的通道特征
    """
    def __init__(self, in_channels, scales=[3, 5, 7]):
        super(MultiscaleChannelAttention, self).__init__()
        self.in_channels = in_channels
        
        # 创建多个不同核大小的1D卷积，捕获不同时间尺度的模式
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
            for k in scales
        ])
        
        # 融合不同尺度的注意力特征
        self.fusion = nn.Conv1d(len(scales), 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # 加入批归一化提高训练稳定性
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        # 特征归一化
        x_norm = self.bn(x)
        
        # 全局平均池化
        y = torch.mean(x_norm, dim=2, keepdim=True)  # [batch, channels, 1]
        y = y.transpose(-1, -2)  # [batch, 1, channels]
        
        # 多尺度处理
        multi_scale_feats = []
        for conv in self.conv_layers:
            multi_scale_feats.append(conv(y))
            
        # 拼接多尺度特征 [batch, num_scales, channels]
        multi_scale_feats = torch.cat(multi_scale_feats, dim=1)
        
        # 融合多尺度特征
        fused_feats = self.fusion(multi_scale_feats)  # [batch, 1, channels]
        
        # 转置回原始维度
        fused_feats = fused_feats.transpose(-1, -2)  # [batch, channels, 1]
        
        # 生成注意力权重
        attention = self.sigmoid(fused_feats)
        
        # 采用残差结构
        return x + x * attention.expand_as(x)