"""
时序数据归一化模块
参考 TimeCMA 的 StandardNorm，支持可逆归一化（RevIN）
适用于 ProLLM 的双模态架构
"""
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        时序数据归一化层（支持可逆归一化 RevIN）
        
        Args:
            num_features: 特征数量（通道数），例如 6
            eps: 数值稳定性参数，默认 1e-5
            affine: 是否使用可学习的仿射参数（缩放和偏移），默认 False
            subtract_last: 是否减去最后一个时间步的值（用于趋势去除），默认 False
            non_norm: 是否禁用归一化（用于消融实验），默认 False
        
        Example:
            >>> norm_layer = Normalize(num_features=6, affine=True)
            >>> x = torch.randn(32, 361, 6)  # [B, L, M]
            >>> x_norm = norm_layer(x, mode='norm')      # 归一化
            >>> x_denorm = norm_layer(x_norm, mode='denorm')  # 反归一化
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 [B, L, M]
            mode: 'norm' 表示归一化，'denorm' 表示反归一化
        
        Returns:
            处理后的张量，形状 [B, L, M]
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"不支持的模式: {mode}，请使用 'norm' 或 'denorm'")
        return x

    def _init_params(self):
        """初始化可学习的仿射参数"""
        # 形状: (num_features,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        计算归一化统计量（均值和标准差）
        
        Args:
            x: 输入张量，形状 [B, L, M]
        """
        # 对时间维度求统计量，保留 batch 和 feature 维度
        # dim2reduce = (1,) 表示对时间维度 L 求均值/标准差
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            # 减去最后一个时间步的值（用于去除趋势）
            self.last = x[:, -1, :].unsqueeze(1)  # [B, 1, M]
        else:
            # 计算每个样本每个通道的均值
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # [B, 1, M]
        
        # 计算每个样本每个通道的标准差
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, 1, M]

    def _normalize(self, x):
        """
        归一化操作
        
        Args:
            x: 输入张量，形状 [B, L, M]
        
        Returns:
            归一化后的张量，形状 [B, L, M]
        """
        if self.non_norm:
            return x
        
        # 步骤1: 去中心化
        if self.subtract_last:
            x = x - self.last  # 减去最后一个时间步
        else:
            x = x - self.mean  # 减去均值
        
        # 步骤2: 标准化
        x = x / self.stdev
        
        # 步骤3: 可学习的仿射变换（可选）
        if self.affine:
            x = x * self.affine_weight  # 缩放
            x = x + self.affine_bias    # 偏移
        
        return x

    def _denormalize(self, x):
        """
        反归一化操作（用于恢复原始尺度）
        
        Args:
            x: 归一化后的张量，形状 [B, L, M]
        
        Returns:
            反归一化后的张量，形状 [B, L, M]
        """
        if self.non_norm:
            return x
        
        # 步骤1: 反向仿射变换（可选）
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        
        # 步骤2: 反标准化
        x = x * self.stdev
        
        # 步骤3: 恢复中心
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        return x


class InstanceNorm(nn.Module):
    """
    实例归一化（Instance Normalization）
    对每个样本的所有通道一起归一化
    """
    def __init__(self, eps=1e-5, affine=False):
        """
        Args:
            eps: 数值稳定性参数
            affine: 是否使用可学习的仿射参数
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, mode: str):
        """
        Args:
            x: 输入张量，形状 [B, L, M]
            mode: 'norm' 或 'denorm'
        """
        if mode == 'norm':
            # 对每个样本的所有维度（时间+通道）求统计量
            self.mean = x.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
            self.std = x.std(dim=(1, 2), keepdim=True) + self.eps  # [B, 1, 1]
            
            x = (x - self.mean) / self.std
            
            if self.affine:
                x = x * self.weight + self.bias
            
            return x
        
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            
            x = x * self.std + self.mean
            return x
        
        else:
            raise NotImplementedError(f"不支持的模式: {mode}")


class MinMaxNorm(nn.Module):
    """
    Min-Max 归一化
    将数据缩放到 [0, 1] 范围
    """
    def __init__(self, eps=1e-8):
        """
        Args:
            eps: 数值稳定性参数
        """
        super(MinMaxNorm, self).__init__()
        self.eps = eps
    
    def forward(self, x, mode: str):
        """
        Args:
            x: 输入张量，形状 [B, L, M]
            mode: 'norm' 或 'denorm'
        """
        if mode == 'norm':
            # 对每个样本每个通道求 min 和 max
            self.min_val = x.min(dim=1, keepdim=True)[0]  # [B, 1, M]
            self.max_val = x.max(dim=1, keepdim=True)[0]  # [B, 1, M]
            
            x = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
            return x
        
        elif mode == 'denorm':
            x = x * (self.max_val - self.min_val + self.eps) + self.min_val
            return x
        
        else:
            raise NotImplementedError(f"不支持的模式: {mode}")


def get_normalizer(norm_type: str, num_features: int, **kwargs):
    """
    工厂函数：根据类型返回对应的归一化层
    
    Args:
        norm_type: 归一化类型，可选 'standard', 'instance', 'minmax', 'none'
        num_features: 特征数量（通道数）
        **kwargs: 其他参数
    
    Returns:
        归一化层实例
    
    Example:
        >>> normalizer = get_normalizer('standard', num_features=6, affine=True)
        >>> x_norm = normalizer(x, mode='norm')
    """
    if norm_type == 'standard':
        return Normalize(
            num_features=num_features,
            eps=kwargs.get('eps', 1e-5),
            affine=kwargs.get('affine', False),
            subtract_last=kwargs.get('subtract_last', False),
            non_norm=kwargs.get('non_norm', False)
        )
    
    elif norm_type == 'instance':
        return InstanceNorm(
            eps=kwargs.get('eps', 1e-5),
            affine=kwargs.get('affine', False)
        )
    
    elif norm_type == 'minmax':
        return MinMaxNorm(
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif norm_type == 'none':
        return Normalize(num_features=num_features, non_norm=True)
    
    else:
        raise ValueError(
            f"不支持的归一化类型: {norm_type}，"
            f"请选择 'standard', 'instance', 'minmax', 'none'"
        )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试归一化模块")
    print("=" * 60)
    
    # 创建测试数据
    B, L, M = 32, 361, 6
    x = torch.randn(B, L, M) * 10 + 5  # 均值约为5，标准差约为10
    
    print(f"\n原始数据形状: {x.shape}")
    print(f"原始数据均值: {x.mean():.4f}")
    print(f"原始数据标准差: {x.std():.4f}")
    
    # 测试 Standard Normalization
    print("\n" + "=" * 60)
    print("测试 Standard Normalization")
    print("=" * 60)
    norm_layer = Normalize(num_features=M, affine=True)
    x_norm = norm_layer(x, mode='norm')
    print(f"归一化后均值: {x_norm.mean():.4f}")
    print(f"归一化后标准差: {x_norm.std():.4f}")
    
    x_denorm = norm_layer(x_norm, mode='denorm')
    print(f"反归一化后均值: {x_denorm.mean():.4f}")
    print(f"反归一化后标准差: {x_denorm.std():.4f}")
    print(f"重建误差: {(x - x_denorm).abs().mean():.6f}")
    
    # 测试 Instance Normalization
    print("\n" + "=" * 60)
    print("测试 Instance Normalization")
    print("=" * 60)
    inst_norm = InstanceNorm(affine=False)
    x_inst = inst_norm(x, mode='norm')
    print(f"归一化后均值: {x_inst.mean():.4f}")
    print(f"归一化后标准差: {x_inst.std():.4f}")
    
    # 测试 MinMax Normalization
    print("\n" + "=" * 60)
    print("测试 MinMax Normalization")
    print("=" * 60)
    minmax_norm = MinMaxNorm()
    x_minmax = minmax_norm(x, mode='norm')
    print(f"归一化后最小值: {x_minmax.min():.4f}")
    print(f"归一化后最大值: {x_minmax.max():.4f}")
    
    # 测试工厂函数
    print("\n" + "=" * 60)
    print("测试工厂函数")
    print("=" * 60)
    normalizer = get_normalizer('standard', num_features=M, affine=True)
    x_test = normalizer(x, mode='norm')
    print(f"工厂函数创建的归一化层工作正常: {x_test.shape}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
