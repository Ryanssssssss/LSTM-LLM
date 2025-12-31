import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import time
from datetime import datetime
import os
import wandb
import logging
import matplotlib.pyplot as plt
import numpy as np
from thop import profile

class Trainer:
    def __init__(self, model, train_loader, test_loader, configs, logger, wandb):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.configs = configs
        self.logger = logger
        self.optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr,
                                         weight_decay=configs.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=configs.step_size, gamma=configs.gamma)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter(
            'tensorboard/' + configs.llm_type + configs.dataset +
            datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.best_test_accuracy = 0
        self.best_epoch = 0
        # 添加指标记录列表
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_f1_scores = []
        self.test_f1_scores = []
        self.epochs_list = []
        self.train_epochs = []  # 添加训练 epoch 列表

    def train(self):
        self.model.train()
        total_loss = 0
        total_accuracy = MulticlassAccuracy(num_classes=self.configs.num_class).cuda()
        total_precision = MulticlassPrecision(num_classes=self.configs.num_class).cuda()
        total_recall = MulticlassRecall(num_classes=self.configs.num_class).cuda()
        total_f1 = MulticlassF1Score(num_classes=self.configs.num_class).cuda()
        
        nan_detected = False  # 添加 NaN 检测标志
        
        for batch_idx, (data, target, indices) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                data, target = data.float().cuda(), target.long().cuda()
                indices = indices.cuda()
                
            # ✅ 修复：禁用 autocast 或使用 bfloat16，避免与 LLM 的 bfloat16 冲突
            # 修改：将索引传递给模型
            output_y = self.model(data, indices)
            
            # 检查模型输出
            if torch.isnan(output_y).any():
                self.logger.warning(f"Train Batch {batch_idx}: 模型输出包含 NaN")
                nan_detected = True
                continue
            
            # 检查模型输出的数值范围（过大会导致 softmax 溢出）
            if torch.isinf(output_y).any():
                self.logger.warning(f"Train Batch {batch_idx}: 模型输出包含 Inf")
                self.logger.warning(f"  输出统计: min={output_y.min():.4f}, max={output_y.max():.4f}")
                nan_detected = True
                continue
            
            # ✅ 关键修复：限制输出范围，防止 CrossEntropyLoss 计算溢出
            output_y = torch.clamp(output_y, min=-100, max=100)
            
            loss = self.loss_function(output_y, target)
            
            # 检查 loss
            if torch.isnan(loss):
                self.logger.warning(f"Train Batch {batch_idx}: Loss 为 NaN, 跳过此批次")
                self.logger.warning(f"  Output stats: min={output_y.min():.4f}, max={output_y.max():.4f}, mean={output_y.mean():.4f}")
                self.logger.warning(f"  Target: {target}")
                nan_detected = True
                continue
            
            # 在 autocast 外部计算指标，避免 dtype 不匹配
            _, output = torch.max(output_y.float(), 1)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_accuracy(output, target)
            total_precision(output, target)
            total_recall(output, target)
            total_f1(output, target)
            
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy.compute()
        avg_precision = total_precision.compute()
        avg_recall = total_recall.compute()
        avg_f1 = total_f1.compute()
        
        # 如果检测到 NaN，记录警告
        if nan_detected:
            self.logger.warning("训练中检测到 NaN，已跳过相应批次")
        
        # 记录训练指标
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy.item())
        self.train_f1_scores.append(avg_f1.item())
        self.train_epochs.append(self.scheduler.last_epoch)  # 记录训练 epoch
        
        # Log metrics to both TensorBoard and W&B
        metrics = {
            'train/loss': avg_loss,
            'train/accuracy': avg_accuracy,
            'train/precision': avg_precision,
            'train/recall': avg_recall,
            'train/f1_score': avg_f1,
            'train/learning_rate': self.scheduler.get_last_lr()[0],
            'epoch': self.scheduler.last_epoch
        }
        wandb.log(metrics)
        
        # TensorBoard logging
        self.writer.add_scalar('Train/Loss', avg_loss, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Accuracy', avg_accuracy, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Precision', avg_precision, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Recall', avg_recall, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/F1 Score', avg_f1, global_step=self.scheduler.last_epoch)
        
        self.logger.info(f'Epoch {self.scheduler.last_epoch} - Train Loss: {avg_loss:.4f}, '
                       f'Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, '
                       f'Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')

    def test(self):
        self.model.eval()
        total_loss = 0
        total_accuracy = MulticlassAccuracy(num_classes=self.configs.num_class).cuda()
        total_precision = MulticlassPrecision(num_classes=self.configs.num_class).cuda()
        total_recall = MulticlassRecall(num_classes=self.configs.num_class).cuda()
        total_f1 = MulticlassF1Score(num_classes=self.configs.num_class).cuda()
        
        nan_detected = False  # 添加 NaN 检测标志
        
        with torch.no_grad():
            for batch_idx, (data, target, indices) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    data, target = data.float().cuda(), target.long().cuda()
                    indices = indices.cuda()
                    
                # 检查输入数据是否包含 NaN
                if torch.isnan(data).any():
                    self.logger.warning(f"Test Batch {batch_idx}: 输入数据包含 NaN")
                    nan_detected = True
                    continue
                    
                # ✅ 修复：禁用 autocast，避免与 LLM 的 bfloat16 冲突
                # 修改：将索引传递给模型
                output_y = self.model(data, indices)
                
                # 检查模型输出是否包含 NaN 或 Inf
                if torch.isnan(output_y).any():
                    self.logger.warning(f"Test Batch {batch_idx}: 模型输出包含 NaN")
                    self.logger.warning(f"  输出统计: min={output_y.min():.4f}, max={output_y.max():.4f}")
                    nan_detected = True
                    continue
                
                if torch.isinf(output_y).any():
                    self.logger.warning(f"Test Batch {batch_idx}: 模型输出包含 Inf")
                    self.logger.warning(f"  输出统计: min={output_y.min():.4f}, max={output_y.max():.4f}")
                    nan_detected = True
                    continue
                
                # ✅ 关键修复：限制输出范围，防止 CrossEntropyLoss 计算溢出
                output_y = torch.clamp(output_y, min=-100, max=100)
                
                loss = self.loss_function(output_y, target)
                
                # 检查 loss 是否为 NaN
                if torch.isnan(loss):
                    self.logger.warning(f"Test Batch {batch_idx}: Loss 为 NaN")
                    self.logger.warning(f"  Output stats: min={output_y.min():.4f}, max={output_y.max():.4f}, mean={output_y.mean():.4f}")
                    self.logger.warning(f"  Target: {target}")
                    self.logger.warning(f"  Output sample: {output_y[0]}")
                    nan_detected = True
                    continue  # 跳过这个批次
                
                # 在 autocast 外部计算指标，避免 dtype 不匹配
                _, output = torch.max(output_y.float(), 1)
                
                total_loss += loss.item()
                total_accuracy(output, target)
                total_precision(output, target)
                total_recall(output, target)
                total_f1(output, target)
                
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_accuracy.compute()
        avg_precision = total_precision.compute()
        avg_recall = total_recall.compute()
        avg_f1 = total_f1.compute()
        
        # 如果检测到 NaN，记录警告
        if nan_detected:
            self.logger.error("=" * 60)
            self.logger.error("检测到 NaN! 可能的原因:")
            self.logger.error("1. 模型输出数值不稳定")
            self.logger.error("2. 离线 embeddings 数据损坏")
            self.logger.error("3. 数据预处理问题")
            self.logger.error("=" * 60)
        
        # 记录测试指标
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(avg_accuracy.item())
        self.test_f1_scores.append(avg_f1.item())
        self.epochs_list.append(self.scheduler.last_epoch)
        
        # Log metrics to both TensorBoard and W&B
        metrics = {
            'test/loss': avg_loss,
            'test/accuracy': avg_accuracy,
            'test/precision': avg_precision,
            'test/recall': avg_recall,
            'test/f1_score': avg_f1,
            'epoch': self.scheduler.last_epoch
        }
        wandb.log(metrics)
        
        # TensorBoard logging
        self.writer.add_scalar('Test/Loss', avg_loss, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Accuracy', avg_accuracy, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Precision', avg_precision, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Recall', avg_recall, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/F1 Score', avg_f1, global_step=self.scheduler.last_epoch)
        
        self.logger.info(f'Epoch {self.scheduler.last_epoch} - Test Loss: {avg_loss:.4f}, '
                        f'Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, '
                        f'Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')
        
        # Checkpoint the best model
        if avg_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = avg_accuracy
            self.best_epoch = self.scheduler.last_epoch
            checkpoint_path = f"{self.configs.path}/best_model_epoch_{self.best_epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            self.logger.info(f"New best model saved at {checkpoint_path} with accuracy: {avg_accuracy:.4f}")
            
            # Log best model to W&B
            wandb.run.summary["best_accuracy"] = avg_accuracy
            wandb.run.summary["best_epoch"] = self.best_epoch

    def plot_metrics(self):
        """绘制训练和测试指标的对比图"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = (
            f"plots/"
            f"{self.configs.dataset}_{self.configs.llm_type}_"
            f"{self.configs.depth}depth_{self.configs.channels}channels_"
            f"{self.configs.few_shot}fewshot_lr_{self.configs.lr}_"
            f"{self.configs.batch_size}batchsize_{timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_epochs, self.train_losses, label='Train Loss', marker='o')
        if len(self.epochs_list) > 0:  # 确保有测试数据
            plt.plot(self.epochs_list, self.test_losses, label='Test Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Testing Loss\n{self.configs.dataset} - {self.configs.llm_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/loss_curves_{self.configs.dataset}_{self.configs.llm_type}_{timestamp}.png')
        plt.close()
        
        # 绘制准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_epochs, self.train_accuracies, label='Train Accuracy', marker='o')
        if len(self.epochs_list) > 0:  # 确保有测试数据
            plt.plot(self.epochs_list, self.test_accuracies, label='Test Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Testing Accuracy\n{self.configs.dataset} - {self.configs.llm_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/accuracy_curves_{self.configs.dataset}_{self.configs.llm_type}_{timestamp}.png')
        plt.close()
        
        # 绘制 F1 分数曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_epochs, self.train_f1_scores, label='Train F1', marker='o')
        if len(self.epochs_list) > 0:  # 确保有测试数据
            plt.plot(self.epochs_list, self.test_f1_scores, label='Test F1', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title(f'Training and Testing F1 Score\n{self.configs.dataset} - {self.configs.llm_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/f1_curves_{self.configs.dataset}_{self.configs.llm_type}_{timestamp}.png')
        plt.close()

    def measure_efficiency_metrics(self):
        """测量模型效率指标：参数量、FLOPs、推理时间、显存占用"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测量模型效率指标")
        self.logger.info("=" * 60)
        
        self.model.eval()
        
        # 初始化所有默认值
        params_M = 0.0
        trainable_params_M = 0.0
        flops_G = 0.0
        avg_inference_time = 0.0
        std_inference_time = 0.0
        peak_memory = 0.0
        
        # 检测模型的数据类型（从 LLM 模型参数获取）
        model_dtype = torch.float32
        if hasattr(self.model, 'llm_model'):
            for param in self.model.llm_model.parameters():
                model_dtype = param.dtype
                break
        
        self.logger.info(f"检测到模型 LLM 数据类型: {model_dtype}")
        
        # 创建单样本输入（全局使用）
        # 注意：输入数据保持 float32，因为训练时 encoder 部分使用 float32
        # 只有 LLM 部分使用 bfloat16，模型内部会自动处理类型转换
        dummy_input = torch.randn(1, self.configs.dimensions, self.configs.length, dtype=torch.float32).cuda()
        dummy_indices = torch.tensor([0], dtype=torch.long).cuda()
        
        # 1. 参数量（固定值，测1次）
        self.logger.info(f"\n1. 参数量:")
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            params_M = total_params / 1e6
            trainable_params_M = trainable_params / 1e6
            
            self.logger.info(f"   总参数: {params_M:.2f} M ({total_params:,} 个)")
            self.logger.info(f"   可训练参数: {trainable_params_M:.2f} M ({trainable_params:,} 个)")
        except Exception as e:
            self.logger.error(f"   参数量计算失败: {e}")
            self.logger.info(f"   总参数: 0.00 M (计算失败)")
        
        # 2. FLOPs（固定值，测1次）- 单独 try-except，失败不影响后续
        self.logger.info(f"\n2. FLOPs 测量:")
        try:
            # ✅ 修复：直接使用原模型，thop 库不会修改模型状态
            # 不使用 deepcopy，因为 LoRA/weight_norm 等组件不支持深拷贝
            self.logger.info(f"   正在计算 FLOPs...")
            
            # 创建用于 FLOPs 计算的输入（使用 float32，与训练时一致）
            flops_input = dummy_input.clone()
            flops_indices = dummy_indices.clone()
            
            self.logger.info(f"   FLOPs 输入数据类型: {flops_input.dtype}")
            
            # 计算 FLOPs
            flops, params = profile(
                self.model, 
                inputs=(flops_input, flops_indices), 
                verbose=False
            )
            flops_G = flops / 1e9
            self.logger.info(f"   FLOPs: {flops_G:.2f} GFLOPs ({flops:,} 次浮点运算)")
            
            # 删除副本释放内存
            del model_copy
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.warning(f"   FLOPs 计算失败 (跳过): {str(e)}")
            # 打印详细的错误信息
            import traceback
            self.logger.warning(f"   详细错误:\n{traceback.format_exc()}")
            self.logger.info(f"   FLOPs: 0.00 GFLOPs (计算失败)")
        
        # 3. 单样本推理时间（需要多次测量取平均）
        self.logger.info(f"\n3. 推理时间测量:")
        try:
            num_warmup = 10
            num_runs = 100
            
            self.logger.info(f"   输入数据类型: {dummy_input.dtype}")
            self.logger.info(f"   输入设备: {dummy_input.device}")
            self.logger.info(f"   输入形状: {dummy_input.shape}")
            
            # 直接使用 dummy_input，它已经是正确的数据类型
            # 预热GPU
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = self.model(dummy_input, dummy_indices)
            
            # 测量推理时间
            torch.cuda.synchronize()
            times = []
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input, dummy_indices)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_inference_time = np.mean(times)
            std_inference_time = np.std(times)
            min_inference_time = np.min(times)
            max_inference_time = np.max(times)
            
            self.logger.info(f"   预热次数: {num_warmup}, 测量次数: {num_runs}")
            self.logger.info(f"   平均推理时间: {avg_inference_time*1000:.3f} ms ({avg_inference_time:.6f} s)")
            self.logger.info(f"   标准差: {std_inference_time*1000:.3f} ms")
            self.logger.info(f"   最小值: {min_inference_time*1000:.3f} ms")
            self.logger.info(f"   最大值: {max_inference_time*1000:.3f} ms")
        except Exception as e:
            self.logger.error(f"   推理时间测量失败: {e}")
            # 打印详细的错误信息
            import traceback
            self.logger.error(f"   详细错误:\n{traceback.format_exc()}")
            self.logger.info(f"   平均推理时间: 0.000 ms (测量失败)")
        
        # 4. 推理显存占用（峰值）
        self.logger.info(f"\n4. GPU显存占用:")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            self.logger.info(f"   输入数据类型: {dummy_input.dtype}")
            
            # 直接使用 dummy_input，它已经是正确的数据类型
            with torch.no_grad():
                _ = self.model(dummy_input, dummy_indices)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            
            self.logger.info(f"   峰值显存: {peak_memory:.2f} MiB")
            self.logger.info(f"   当前显存: {current_memory:.2f} MiB")
        except Exception as e:
            self.logger.error(f"   显存测量失败: {e}")
            # 打印详细的错误信息
            import traceback
            self.logger.error(f"   详细错误:\n{traceback.format_exc()}")
            self.logger.info(f"   峰值显存: 0.00 MiB (测量失败)")
        
        # 汇总指标
        efficiency_metrics = {
            'params_M': params_M,
            'trainable_params_M': trainable_params_M,
            'flops_G': flops_G,
            'inference_time_ms': avg_inference_time * 1000,
            'inference_time_std_ms': std_inference_time * 1000,
            'memory_MiB': peak_memory
        }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("效率指标汇总:")
        self.logger.info("=" * 60)
        self.logger.info(f"参数量:       {params_M:.2f} M")
        self.logger.info(f"FLOPs:        {flops_G:.2f} GFLOPs")
        self.logger.info(f"推理时间:     {avg_inference_time*1000:.3f} ± {std_inference_time*1000:.3f} ms")
        self.logger.info(f"显存占用:     {peak_memory:.2f} MiB")
        self.logger.info("=" * 60 + "\n")
        
        # 记录到 W&B
        wandb.run.summary.update({
            'efficiency/params_M': params_M,
            'efficiency/trainable_params_M': trainable_params_M,
            'efficiency/flops_G': flops_G,
            'efficiency/inference_time_ms': avg_inference_time * 1000,
            'efficiency/inference_time_std_ms': std_inference_time * 1000,
            'efficiency/memory_MiB': peak_memory
        })
        
        return efficiency_metrics

    def run(self):
        is_exist = os.path.exists(self.configs.path)
        if not is_exist:
            os.makedirs(self.configs.path)
        
        self.logger.info(f"Starting training for {self.configs.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.configs.epochs):
            epoch_start_time = time.time()
            self.logger.info(f'Starting Epoch {epoch + 1}/{self.configs.epochs}')
            
            self.train()
            
            if epoch % self.configs.interval == 0:
                self.test()
                
            self.scheduler.step()
            elapsed_time = time.time() - epoch_start_time
            self.logger.info(f'Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds.')
        
        # 训练结束后绘制图表
        self.plot_metrics()
        self.logger.info("Training curves have been plotted and saved in 'plots' directory.")
        
        # 测量模型效率指标
        self.logger.info("\n" + "=" * 60)
        self.logger.info("训练完成，开始测量模型效率指标...")
        self.logger.info("=" * 60)
        efficiency_metrics = self.measure_efficiency_metrics()
        
        self.writer.close()
        self.logger.info('Training complete.')
        self.logger.info(f'Best Test Performance at Epoch {self.best_epoch + 1}: Accuracy {self.best_test_accuracy:.4f}')
        
        # Final W&B logging
        wandb.run.summary["final_accuracy"] = self.best_test_accuracy
        wandb.run.summary["total_epochs"] = self.configs.epochs
        wandb.run.summary["training_time"] = time.time() - start_time
        
        return efficiency_metrics