


import torch
import math
from ChannelAttention import MultiscaleChannelAttention


class Chomp1d(torch.nn.Module):
    """
    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 返回输入x的最后一列去掉chomp_size个元素的结果
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    @param use_channel_attention 是否使用通道注意力机制
    @param channel_attention_type 使用的通道注意力类型
    @param attention_position 注意力应用的位置 ('before', 'after', 'both')
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False, use_channel_attention=False, 
                 channel_attention_type='eca', attention_position='after',
                 attention_kernel_size=None):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # 截断使卷积成为因果关系
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # 通道注意力模块 - 增强多传感器通道交互
        self.use_channel_attention = use_channel_attention
        self.attention_position = attention_position
        
        if use_channel_attention:
            # 构建前置注意力模块 (应用于输入通道)
            if attention_position in ['before', 'both']:
                if channel_attention_type == 'eca':
                    self.pre_attention = ECAModule(in_channels, k_size=attention_kernel_size)
                elif channel_attention_type == 'sensor_specific':
                    self.pre_attention = SensorSpecificAttention(in_channels)
                elif channel_attention_type == 'adaptive_fusion':
                    self.pre_attention = AdaptiveChannelFusion(in_channels)
                elif channel_attention_type == 'multiscale':
                    self.pre_attention = MultiscaleChannelAttention(in_channels)
                    
            # 构建后置注意力模块 (应用于输出通道)
            if attention_position in ['after', 'both']:
                if channel_attention_type == 'eca':
                    self.post_attention = ECAModule(out_channels, k_size=attention_kernel_size)
                elif channel_attention_type == 'sensor_specific':
                    self.post_attention = SensorSpecificAttention(out_channels)
                elif channel_attention_type == 'adaptive_fusion':
                    self.post_attention = AdaptiveChannelFusion(out_channels)
                elif channel_attention_type == 'multiscale':
                    self.post_attention = MultiscaleChannelAttention(out_channels)

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        # 应用前置注意力
        if self.use_channel_attention and self.attention_position in ['before', 'both']:
            x = self.pre_attention(x)
            
        # 对输入x进行因果卷积操作，得到out_causal
        out_causal = self.causal(x)  # x: 10,3,1346 out_causal: 10,40,1346
        
        # 应用后置注意力
        if self.use_channel_attention and self.attention_position in ['after', 'both']:
            out_causal = self.post_attention(out_causal)
            
        # 残差：如果self.upordownsample为None，则res=x，否则res=self.upordownsample(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        
        # 如果self.relu为None，则返回out_causal+res，否则返回self.relu(out_causal+res)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param use_channel_attention 是否使用通道注意力机制
    @param channel_attention_type 使用的通道注意力类型
    """

    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, use_channel_attention=False,
                 channel_attention_type='eca', attention_position='after',
                 attention_kernel_size=None):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        # Loop through the depth of the network
        for i in range(depth):
            # Set the input channels for the first block to the input channels of the network
            in_channels_block = in_channels if i == 0 else channels
            
            # 针对不同层设置不同的注意力参数
            # 对于输入层和中间层采用不同的策略
            current_use_attention = use_channel_attention
            current_position = attention_position
            
            # 仅在输入层和输出层应用注意力以减少过拟合风险
            if i != 0 and i != depth-1 and use_channel_attention:
                # 在中间层选择性地应用注意力
                current_use_attention = (i % 2 == 0)  # 只在偶数层应用
            
            # Add a causal convolution block to the list of layers
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size,
                use_channel_attention=current_use_attention,
                channel_attention_type=channel_attention_type,
                attention_position=current_position,
                attention_kernel_size=attention_kernel_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size,
            use_channel_attention=use_channel_attention,
            channel_attention_type=channel_attention_type,
            attention_position=attention_position,
            attention_kernel_size=attention_kernel_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param use_channel_attention 是否使用通道注意力机制
    @param channel_attention_type 使用的通道注意力类型
    """

    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, use_channel_attention=False,
                 channel_attention_type='eca', attention_position='after',
                 attention_kernel_size=None):
        super(CausalCNNEncoder, self).__init__()
        # 初始化CausalCNN
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size,
            use_channel_attention=use_channel_attention,
            channel_attention_type=channel_attention_type,
            attention_position=attention_position,
            attention_kernel_size=attention_kernel_size
        )
        # 定义自适应最大池化层
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        # 定义SqueezeChannels层
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        # 定义线性层
        linear = torch.nn.Linear(reduced_size, out_channels)
        # 将CausalCNN、自适应最大池化层、SqueezeChannels层和线性层组合成网络
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)

