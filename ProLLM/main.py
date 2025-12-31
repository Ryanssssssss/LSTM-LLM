import argparse
import logging
import os
import random
from datetime import datetime
import wandb
import numpy as np
import torch
from pathlib import Path
from DataloaderConstructing import DataloaderConstructing
from LLMFew import LLMFew
from Trainer import Trainer

def setup_logging(args):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"{args.dataset}_{args.llm_type}_{args.depth}depth_{args.channels}channels_{args.few_shot}fewshot_lr_{args.lr}{args.batch_size}batchsize_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for training LLMFew model.")
    # Experiment setup
    parser.add_argument('--dataset', default='BasicMotions', help='Dataset to use for training and evaluation.')
    parser.add_argument('--few_shot', type=int, default=1, help='Whether to use few-shot learning scenario.')
    parser.add_argument('--ablation', type=str, default='train', help='Type of ablation study to perform.')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generators.')
    parser.add_argument('--path', default='ckpt', help='Path to save checkpoints.')
    # Wandb configuration
    parser.add_argument('--wandb_project', default='LLMFew', help='Wandb project name')
    parser.add_argument('--wandb_entity', default=None, help='Wandb entity name')
    parser.add_argument('--wandb_run_name', default=None, help='Wandb run name')
    # Model architecture
    parser.add_argument('--dimensions', type=int, default=6, help='Number of input planes.')
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=4, help='Number of classes.')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the network.')
    parser.add_argument('--channels', type=int, default=256, help='Number of channels.')
    parser.add_argument('--reduced_size', type=int, default=128, help='Reduced size for bottleneck layers.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutions.')
    parser.add_argument('--patch_len', type=int, default=16, help='Length of each patch for embeddings.')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patch embedding.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    # Channel attention parameters
    parser.add_argument('--use_channel_attention', action='store_true', help='Whether to use channel attention mechanism.')
    parser.add_argument('--channel_attention_type', type=str, default='multiscale',
                        choices=['eca', 'sensor_specific', 'adaptive_fusion', 'multiscale'],
                        help='Type of channel attention to use')
    parser.add_argument('--attention_position', type=str, default='before',
                        choices=['before', 'after', 'both'],
                        help='Where to apply attention: before conv, after conv, or both')
    parser.add_argument('--attention_kernel_size', type=int, default=None,
                        help='Kernel size for attention module, if None, will be adaptively determined')
    # Training process
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoints during training.')
    parser.add_argument('--interval', type=int, default=2, help='Interval between testing.')
    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for learning rate decay.')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma for learning rate decay.')
    # Large language model settings
    parser.add_argument('--llm_type', default='llama3', help='Type of large language model to use.')
    parser.add_argument('--lora', type=int, default=1, help='Use Lora layers or not.')
    # Prompt handling parameters
    parser.add_argument('--llm_path', type=str, default='gpt2', help='Path of pretrained LLM for prompt embedding')
    parser.add_argument('--tokenizer_path', type=str, default='gpt2', help='Path of tokenizer for prompt embedding')
    parser.add_argument('--use_offline_embeddings', action='store_true', help='Whether to use offline stored embeddings')
    parser.add_argument('--prompt_max_length', type=int, default=768, help='Maximum token length when tokenizing prompts')
    parser.add_argument('--prompt_representation', type=str, choices=['sequence', 'pooled_last_token'],
                        default='sequence', help='Representation type stored for offline prompts')
    # 生成离线 embeddings 的参数
    parser.add_argument('--generate_embeddings', action='store_true', help='Generate and save prompt embeddings before training')
    parser.add_argument('--embedding_device', type=str, default='cuda', help='Device to use for generating embeddings')
    return parser.parse_args()

def init_wandb(args):
    run_name = args.wandb_run_name or f"{args.dataset}_{args.few_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(mode="disabled")

def check_embeddings_exist(dataset_name, representation):
    """检查每样本 h5 是否存在的最小条件：train 与 test 目录存在且包含至少一个 .h5 文件"""
    if representation == 'sequence':
        train_dir = f'embeddings/{dataset_name}/train'
        test_dir = f'embeddings/{dataset_name}/test'
        # 兼容新目录组织
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            train_dir = f'embeddings/{dataset_name}/sequence/train'
            test_dir = f'embeddings/{dataset_name}/sequence/test'
    else:
        train_dir = f'embeddings/{dataset_name}/{representation}/train'
        test_dir = f'embeddings/{dataset_name}/{representation}/test'
    def has_h5(d):
        return os.path.isdir(d) and any(f.endswith('.h5') for f in os.listdir(d))
    return has_h5(train_dir) and has_h5(test_dir)

def main():
    args = parse_args()
    logger = setup_logging(args)

    # 固定随机种子
    global_seed = 42
    random.seed(global_seed)

    # 实验种子
    if args.seed != -1:
        experiment_seed = args.seed
    else:
        experiment_seed = random.randint(1, 100)
    logger.info(f"Using random seed: {experiment_seed}")
    seed_everything(experiment_seed)

    embeddings_available = check_embeddings_exist(args.dataset, args.prompt_representation)
    
    # 调试信息
    logger.info(f"检查 embeddings 存在性:")
    logger.info(f"  数据集: {args.dataset}")
    logger.info(f"  表示类型: {args.prompt_representation}")
    logger.info(f"  Embeddings 可用: {embeddings_available}")
    
    if args.generate_embeddings or not embeddings_available:
        logger.info(
            f"Generating offline prompt embeddings ({args.prompt_representation}) for dataset {args.dataset}..."
        )
        from embed_prompt import main as generate_embeddings
        import sys
        sys.argv = [
            'embed_prompt.py',
            f'--dataset={args.dataset}',
            f'--device={args.embedding_device}',
            f'--tokenizer_path={args.tokenizer_path}',
            f'--llm_path={args.llm_path}',
            f'--max_token_length={args.prompt_max_length}',
            f'--representation={args.prompt_representation}'
        ]
        try:
            generate_embeddings()
            embeddings_available = True
            logger.info(f"Offline embeddings ({args.prompt_representation}) generated for dataset {args.dataset}.")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            logger.warning("Continuing with online embedding computation...")
    else:
        logger.info(f"Using existing offline embeddings ({args.prompt_representation}) for dataset {args.dataset}.")

    args.use_offline_embeddings = embeddings_available
    if args.use_offline_embeddings:
        logger.info("Offline prompt embeddings available and will be used.")
    else:
        logger.warning("Offline prompt embeddings unavailable; falling back to online prompt encoding.")

    # Initialize wandb
    init_wandb(args)

    # Log config
    logger.info(f"Starting training with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # Data
    logger.info("Preparing dataloaders...")
    train_loader, test_loader = DataloaderConstructing(
        args.dataset,
        batch_size=args.batch_size,
        few_shot=args.few_shot,
        random_seed=experiment_seed
    )
    logger.info(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

    # Model
    logger.info("Initializing model...")
    model = LLMFew(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {total_params:,}')
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to CUDA")

    # Trainer
    trainer = Trainer(model, train_loader, test_loader, args, logger, wandb)
    logger.info(f"Training preparation finished! Starting training on the {args.dataset} dataset!")
    try:
        trainer.run()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        logger.info("Training completed. CUDA memory cleared.")

if __name__ == '__main__':
    main()