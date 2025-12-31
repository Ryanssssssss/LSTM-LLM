"""优化后的数据加载器构建，添加样本索引信息"""
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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

def get_few_shot_samples(x, y, num_samples_per_class=1, random_seed=None):
    """添加随机种子控制以减少种子敏感性"""
    if random_seed is not None:
        random.seed(random_seed)
    
    class_to_samples = defaultdict(list)
    for idx in range(len(y)):
        label = y[idx].item()
        class_to_samples[label].append((x[idx], y[idx], idx))  # 保存原始索引
    
    few_shot_samples = []
    for label, samples in class_to_samples.items():
        if len(samples) >= num_samples_per_class:
            few_shot_samples.extend(random.sample(samples, num_samples_per_class))
        else:
            few_shot_samples.extend(samples)
    
    # 构建结果的 tensor，包括原始索引
    x_few_shot = torch.stack([sample[0] for sample in few_shot_samples])
    y_few_shot = torch.stack([sample[1] for sample in few_shot_samples])
    indices_few_shot = torch.tensor([sample[2] for sample in few_shot_samples], dtype=torch.long)
    
    return x_few_shot, y_few_shot, indices_few_shot

class IndexedTensorDataset(TensorDataset):
    """扩展TensorDataset，添加样本索引"""
    def __init__(self, *tensors):
        super(IndexedTensorDataset, self).__init__(*tensors[:-1])  # 去掉最后一个tensor（索引）
        self.indices = tensors[-1]  # 最后一个tensor是索引
    
    def __getitem__(self, index):
        samples = super(IndexedTensorDataset, self).__getitem__(index)
        return samples + (self.indices[index],)  # 添加索引到返回的元组

def DataloaderConstructing(dataset, batch_size, shuffle=False, pin_memory=True,
                          few_shot=0, random_seed=None):
    """优化的数据加载器构建函数，添加样本索引
    Args:
        dataset: 数据集名称
        batch_size: 批大小
        shuffle: 是否打乱数据
        pin_memory: 是否使用 pin_memory
        few_shot: few-shot 学习的样本数
        random_seed: 随机种子
    """
    # 定义数据集路径
    dataset_path = [
        f'npydata/{dataset}/{dataset}_train_x.npy',
        f'npydata/{dataset}/{dataset}_train_y.npy',
        f'npydata/{dataset}/{dataset}_test_x.npy',
        f'npydata/{dataset}/{dataset}_test_y.npy'
    ]
    
    # 加载数据集
    X_train, y_train, X_test, y_test = np.load(dataset_path[0]), \
                                       np.load(dataset_path[1]), \
                                       np.load(dataset_path[2]), \
                                       np.load(dataset_path[3])
    
    # 将字符串标签转换为数字标签
    y_train, y_test = StringToLabel(y_train), StringToLabel(y_test)
    
    # 将 numpy 数组转换为 torch 张量
    x_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train.squeeze())
    x_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test.squeeze())
    
    # 创建索引张量
    train_indices = torch.arange(len(x_train), dtype=torch.long)
    test_indices = torch.arange(len(x_test), dtype=torch.long)
    
    # 如果设置了 few_shot 参数，则获取少量样本（传入随机种子）
    if few_shot > 0:
        x_train, y_train, train_indices = get_few_shot_samples(
            x_train, y_train, num_samples_per_class=few_shot, random_seed=random_seed)
    
    # 将数据集转换为 IndexedTensorDataset
    deal_train_dataset = IndexedTensorDataset(x_train, y_train, train_indices)
    deal_test_dataset = IndexedTensorDataset(x_test, y_test, test_indices)
    
    # 创建数据加载器时使用固定的 generator
    train_generator = torch.Generator()
    test_generator = torch.Generator()
    if random_seed is not None:
        train_generator.manual_seed(random_seed)
        test_generator.manual_seed(random_seed)
    
    # 将 IndexedTensorDataset 转换为 DataLoader
    # 训练集：使用 shuffle 但配置固定的 generator
    train_loader = DataLoader(
        dataset=deal_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        generator=train_generator if shuffle else None,
        drop_last=True  # 确保每个 epoch batch 数量一致
    )
    
    # 测试集：不使用 shuffle 以确保结果可重现
    test_loader = DataLoader(
        dataset=deal_test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时设为 False
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # 返回训练集和测试集的 DataLoader
    return train_loader, test_loader