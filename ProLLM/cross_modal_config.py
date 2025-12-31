"""
跨模态注意力配置文件
为ProLLM添加TimeCMA风格的跨模态注意力参数
"""

def add_cross_modal_args(parser):
    """为argparse添加跨模态注意力相关参数"""
    
    # 跨模态注意力参数
    parser.add_argument('--cross_modal_heads', type=int, default=8, 
                       help='Number of attention heads in cross-modal alignment')
    parser.add_argument('--cross_modal_layers', type=int, default=2, 
                       help='Number of layers in cross-modal alignment')
    parser.add_argument('--cross_modal_dropout', type=float, default=0.1, 
                       help='Dropout rate for cross-modal attention')
    
    # Prompt编码器参数
    parser.add_argument('--prompt_encoder_heads', type=int, default=8, 
                       help='Number of attention heads in prompt encoder')
    parser.add_argument('--prompt_encoder_layers', type=int, default=1, 
                       help='Number of layers in prompt encoder')
    
    # 融合策略参数
    parser.add_argument('--fusion_strategy', type=str, default='adaptive', 
                       choices=['adaptive', 'concat', 'add', 'gate'],
                       help='Strategy for fusing time series and prompt features')
    parser.add_argument('--use_cross_modal', action='store_true', 
                       help='Whether to use cross-modal attention alignment')
    
    return parser


def get_cross_modal_config():
    """获取默认的跨模态配置"""
    config = {
        'cross_modal_heads': 8,
        'cross_modal_layers': 2,
        'cross_modal_dropout': 0.1,
        'prompt_encoder_heads': 8,
        'prompt_encoder_layers': 1,
        'fusion_strategy': 'adaptive',
        'use_cross_modal': True
    }
    return config


class CrossModalConfig:
    """跨模态配置类"""
    
    def __init__(self, **kwargs):
        # 默认配置
        self.cross_modal_heads = 8
        self.cross_modal_layers = 2
        self.cross_modal_dropout = 0.1
        self.prompt_encoder_heads = 8
        self.prompt_encoder_layers = 1
        self.fusion_strategy = 'adaptive'
        self.use_cross_modal = True
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'cross_modal_heads': self.cross_modal_heads,
            'cross_modal_layers': self.cross_modal_layers,
            'cross_modal_dropout': self.cross_modal_dropout,
            'prompt_encoder_heads': self.prompt_encoder_heads,
            'prompt_encoder_layers': self.prompt_encoder_layers,
            'fusion_strategy': self.fusion_strategy,
            'use_cross_modal': self.use_cross_modal
        }
    
    def update_from_args(self, args):
        """从argparse参数更新配置"""
        for key in self.to_dict().keys():
            if hasattr(args, key):
                setattr(self, key, getattr(args, key))