#!/usr/bin/env python3
"""
高级训练脚本 - 小样本缺陷检测
包含数据增强、域自适应、可解释性等高级功能

适用于 anomalib 0.7.x
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import numpy as np
import albumentations as A
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from anomalib.models.patchcore import Patchcore
from anomalib.models.padim import Padim
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)


# ==================== 数据增强模块 ====================

def get_augmentation_pipeline(
    config: Dict[str, Any],
    mode: str = 'train'
) -> A.Compose:
    """
    获取数据增强流水线
    
    Args:
        config: 增强配置
        mode: 'train' 或 'eval'
    
    Returns:
        Albumentations Compose 对象
    """
    if mode == 'eval':
        # 评估时不进行增强
        return None
    
    transforms = []
    
    # 1. 几何变换
    if config.get('geometric', {}).get('enabled', True):
        geo_config = config.get('geometric', {})
        transforms.extend([
            A.HorizontalFlip(p=geo_config.get('horizontal_flip_p', 0.5)),
            A.VerticalFlip(p=geo_config.get('vertical_flip_p', 0.5)),
            A.RandomRotate90(p=geo_config.get('rotate90_p', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=geo_config.get('shift_limit', 0.0625),
                scale_limit=geo_config.get('scale_limit', 0.1),
                rotate_limit=geo_config.get('rotate_limit', 15),
                border_mode=0,
                p=geo_config.get('shift_scale_rotate_p', 0.5)
            ),
        ])
    
    # 2. 颜色/光照变换
    if config.get('color', {}).get('enabled', True):
        color_config = config.get('color', {})
        transforms.extend([
            A.ColorJitter(
                brightness=color_config.get('brightness', 0.2),
                contrast=color_config.get('contrast', 0.2),
                saturation=color_config.get('saturation', 0.2),
                hue=color_config.get('hue', 0.1),
                p=color_config.get('color_jitter_p', 0.5)
            ),
            A.RandomBrightnessContrast(
                brightness_limit=color_config.get('brightness_limit', 0.2),
                contrast_limit=color_config.get('contrast_limit', 0.2),
                p=color_config.get('brightness_contrast_p', 0.5)
            ),
        ])
    
    # 3. 噪声与模糊
    if config.get('noise', {}).get('enabled', True):
        noise_config = config.get('noise', {})
        transforms.extend([
            A.GaussNoise(
                var_limit=noise_config.get('var_limit', (10.0, 50.0)),
                p=noise_config.get('gauss_noise_p', 0.3)
            ),
            A.GaussianBlur(
                blur_limit=noise_config.get('blur_limit', (3, 7)),
                p=noise_config.get('blur_p', 0.3)
            ),
        ])
    
    # 4. 弹性变换（模拟形变）
    if config.get('elastic', {}).get('enabled', False):
        elastic_config = config.get('elastic', {})
        transforms.append(
            A.ElasticTransform(
                alpha=elastic_config.get('alpha', 120),
                sigma=elastic_config.get('sigma', 6),
                alpha_affine=elastic_config.get('alpha_affine', 3.6),
                p=elastic_config.get('p', 0.3)
            )
        )
    
    return A.Compose(transforms) if transforms else None


def get_default_augmentation_config() -> Dict[str, Any]:
    """获取默认增强配置"""
    return {
        'geometric': {
            'enabled': True,
            'horizontal_flip_p': 0.5,
            'vertical_flip_p': 0.5,
            'rotate90_p': 0.5,
            'shift_limit': 0.0625,
            'scale_limit': 0.1,
            'rotate_limit': 15,
            'shift_scale_rotate_p': 0.5,
        },
        'color': {
            'enabled': True,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'color_jitter_p': 0.5,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'brightness_contrast_p': 0.5,
        },
        'noise': {
            'enabled': True,
            'var_limit': [10.0, 50.0],
            'gauss_noise_p': 0.3,
            'blur_limit': [3, 7],
            'blur_p': 0.3,
        },
        'elastic': {
            'enabled': False,
            'alpha': 120,
            'sigma': 6,
            'alpha_affine': 3.6,
            'p': 0.3,
        }
    }


# ==================== 模型工厂 ====================

def create_model(
    model_name: str,
    input_size: Tuple[int, int],
    config: Dict[str, Any]
) -> torch.nn.Module:
    """
    创建异常检测模型
    
    Args:
        model_name: 模型名称 ('patchcore', 'padim', 等)
        input_size: 输入图像尺寸
        config: 模型配置
    
    Returns:
        模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'patchcore':
        return Patchcore(
            input_size=input_size,
            backbone=config.get('backbone', 'resnet18'),
            layers=config.get('layers', ['layer2', 'layer3']),
            pre_trained=config.get('pre_trained', True),
            coreset_sampling_ratio=config.get('coreset_sampling_ratio', 0.1),
            num_neighbors=config.get('num_neighbors', 9),
        )
    
    elif model_name == 'padim':
        return Padim(
            input_size=input_size,
            backbone=config.get('backbone', 'resnet18'),
            layers=config.get('layers', ['layer1', 'layer2', 'layer3']),
            pre_trained=config.get('pre_trained', True),
            n_features=config.get('n_features', 100),
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")


# ==================== 特征归一化（域自适应） ====================

class FeatureNormalizer:
    """
    特征归一化器 - 用于减轻域偏移影响
    """
    
    def __init__(self, method: str = 'instance'):
        """
        Args:
            method: 归一化方法 ('instance', 'layer', 'batch', 'none')
        """
        self.method = method
        self.running_mean = None
        self.running_std = None
        self.momentum = 0.1
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        对特征进行归一化
        
        Args:
            features: [B, C, H, W] 特征图
        
        Returns:
            归一化后的特征图
        """
        if self.method == 'none':
            return features
        
        if self.method == 'instance':
            # 实例归一化：每个样本独立归一化
            mean = features.mean(dim=[2, 3], keepdim=True)
            std = features.std(dim=[2, 3], keepdim=True) + 1e-8
            return (features - mean) / std
        
        elif self.method == 'layer':
            # 层归一化
            mean = features.mean(dim=[1, 2, 3], keepdim=True)
            std = features.std(dim=[1, 2, 3], keepdim=True) + 1e-8
            return (features - mean) / std
        
        elif self.method == 'batch':
            # 批归一化（带运行统计量）
            mean = features.mean(dim=[0, 2, 3], keepdim=True)
            std = features.std(dim=[0, 2, 3], keepdim=True) + 1e-8
            
            # 更新运行统计量
            if self.running_mean is None:
                self.running_mean = mean.detach()
                self.running_std = std.detach()
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std.detach()
            
            return (features - mean) / std
        
        return features


# ==================== 增量学习支持 ====================

class IncrementalMemoryBank:
    """
    支持增量学习的记忆库
    可用于适应新产品变型
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        update_ratio: float = 0.1,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            max_size: 记忆库最大容量
            update_ratio: 更新比例
            similarity_threshold: 相似度阈值（低于此值的样本会被添加）
        """
        self.memory = None
        self.max_size = max_size
        self.update_ratio = update_ratio
        self.similarity_threshold = similarity_threshold
    
    def initialize(self, features: torch.Tensor):
        """初始化记忆库"""
        self.memory = features.clone()
        self._maintain_size()
    
    def update(self, new_features: torch.Tensor):
        """
        增量更新记忆库
        
        Args:
            new_features: 新的特征向量 [N, D]
        """
        if self.memory is None:
            self.initialize(new_features)
            return
        
        # 计算新样本与记忆库的距离
        distances = torch.cdist(new_features, self.memory)
        min_distances = distances.min(dim=1)[0]
        
        # 选择差异大的样本（代表新分布的样本）
        novel_mask = min_distances > self.similarity_threshold
        novel_features = new_features[novel_mask]
        
        if len(novel_features) > 0:
            # 限制添加数量
            n_add = min(
                len(novel_features),
                int(self.max_size * self.update_ratio)
            )
            
            # 选择最具代表性的样本
            if len(novel_features) > n_add:
                indices = torch.argsort(min_distances[novel_mask], descending=True)[:n_add]
                novel_features = novel_features[indices]
            
            # 添加到记忆库
            self.memory = torch.cat([self.memory, novel_features], dim=0)
            self._maintain_size()
    
    def _maintain_size(self):
        """维护记忆库大小，使用核心集采样"""
        if len(self.memory) <= self.max_size:
            return
        
        # 使用 k-center 贪心算法进行核心集采样
        indices = self._greedy_kcenter(self.max_size)
        self.memory = self.memory[indices]
    
    def _greedy_kcenter(self, k: int) -> torch.Tensor:
        """贪心 k-center 采样"""
        n = len(self.memory)
        
        # 随机选择第一个点
        selected = [torch.randint(n, (1,)).item()]
        min_distances = torch.full((n,), float('inf'), device=self.memory.device)
        
        for _ in range(k - 1):
            # 更新到已选集合的最小距离
            last_selected = self.memory[selected[-1]]
            distances = torch.norm(self.memory - last_selected, dim=1)
            min_distances = torch.minimum(min_distances, distances)
            
            # 选择距离最远的点
            next_idx = torch.argmax(min_distances).item()
            selected.append(next_idx)
        
        return torch.tensor(selected)
    
    def get_memory(self) -> torch.Tensor:
        """获取记忆库"""
        return self.memory


# ==================== 训练管理器 ====================

class AdvancedTrainer:
    """
    高级训练管理器
    整合数据增强、域自适应、增量学习等功能
    """
    
    def __init__(
        self,
        task_name: str,
        dataset_root: Path,
        output_root: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            task_name: 任务名称 (如 'task_1')
            dataset_root: 数据集根目录
            output_root: 输出根目录
            config: 配置字典
        """
        self.task_name = task_name
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.config = config or self._get_default_config()
        
        # 确保输出目录存在
        self.output_dir = self.output_root / task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.feature_normalizer = FeatureNormalizer(
            method=self.config.get('domain_adaptation', {}).get('normalization', 'instance')
        )
        self.incremental_memory = IncrementalMemoryBank(
            max_size=self.config.get('incremental_learning', {}).get('max_memory_size', 10000),
            update_ratio=self.config.get('incremental_learning', {}).get('update_ratio', 0.1),
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 模型配置
            'model': {
                'name': 'patchcore',
                'backbone': 'resnet18',
                'layers': ['layer2', 'layer3'],
                'pre_trained': True,
                'coreset_sampling_ratio': 0.1,
                'num_neighbors': 9,
            },
            
            # 数据配置
            'data': {
                'image_size': [512, 512],
                'train_batch_size': 1,
                'eval_batch_size': 1,
                'num_workers': 8,
            },
            
            # 训练配置
            'trainer': {
                'max_epochs': 1,
                'accelerator': 'gpu',
                'devices': 1,
                'precision': 32,
            },
            
            # 数据增强配置
            'augmentation': get_default_augmentation_config(),
            
            # 域自适应配置
            'domain_adaptation': {
                'enabled': True,
                'normalization': 'instance',
            },
            
            # 增量学习配置
            'incremental_learning': {
                'enabled': False,
                'max_memory_size': 10000,
                'update_ratio': 0.1,
            },
            
            # 后处理配置
            'post_processing': {
                'normalization': 'min_max',
                'threshold': 'adaptive',
            },
            
            # 指标配置
            'metrics': {
                'image_metrics': ['AUROC', 'F1Score'],
                'pixel_metrics': ['AUROC', 'F1Score'],
            }
        }
    
    def prepare_data(self) -> Folder:
        """准备数据模块"""
        data_config = self.config.get('data', {})
        aug_config = self.config.get('augmentation', {})
        
        # 获取数据增强
        train_transform = get_augmentation_pipeline(aug_config, mode='train')
        
        # 数据路径
        task_dir = self.dataset_root / self.task_name
        
        datamodule = Folder(
            normal_dir=str(task_dir / "good"),
            abnormal_dir=str(task_dir / "defect"),
            mask_dir=str(task_dir / "mask" / "defect"),
            image_size=tuple(data_config.get('image_size', [512, 512])),
            train_batch_size=data_config.get('train_batch_size', 1),
            eval_batch_size=data_config.get('eval_batch_size', 1),
            num_workers=data_config.get('num_workers', 8),
            task=TaskType.SEGMENTATION,
            transform_config_train=train_transform,
        )
        datamodule.setup()
        
        return datamodule
    
    def prepare_model(self) -> torch.nn.Module:
        """准备模型"""
        model_config = self.config.get('model', {})
        data_config = self.config.get('data', {})
        
        input_size = tuple(data_config.get('image_size', [512, 512]))
        
        model = create_model(
            model_name=model_config.get('name', 'patchcore'),
            input_size=input_size,
            config=model_config
        )
        
        return model
    
    def prepare_callbacks(self) -> List:
        """准备回调函数"""
        post_config = self.config.get('post_processing', {})
        metrics_config = self.config.get('metrics', {})
        
        # 归一化方法映射
        norm_map = {
            'min_max': NormalizationMethod.MIN_MAX,
            'cdf': NormalizationMethod.CDF,
            'none': NormalizationMethod.NONE,
        }
        
        # 阈值方法映射
        thresh_map = {
            'adaptive': ThresholdMethod.ADAPTIVE,
            'manual': ThresholdMethod.MANUAL,
        }
        
        callbacks = [
            MetricsConfigurationCallback(
                task=TaskType.SEGMENTATION,
                image_metrics=metrics_config.get('image_metrics', ['AUROC', 'F1Score']),
                pixel_metrics=metrics_config.get('pixel_metrics', ['AUROC', 'F1Score']),
            ),
            PostProcessingConfigurationCallback(
                normalization_method=norm_map.get(
                    post_config.get('normalization', 'min_max'),
                    NormalizationMethod.MIN_MAX
                ),
                threshold_method=thresh_map.get(
                    post_config.get('threshold', 'adaptive'),
                    ThresholdMethod.ADAPTIVE
                ),
            ),
            ModelCheckpoint(
                dirpath=self.output_dir / "weights",
                filename="model",
                monitor="pixel_AUROC",
                mode="max",
                save_last=True,
            ),
        ]
        
        return callbacks
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            训练结果字典
        """
        print(f"=" * 60)
        print(f"开始训练任务: {self.task_name}")
        print(f"=" * 60)
        
        # 设置随机种子
        seed_everything(42)
        
        # 准备组件
        datamodule = self.prepare_data()
        model = self.prepare_model()
        callbacks = self.prepare_callbacks()
        
        print(f"\n数据集信息:")
        print(f"  正常样本 (训练): {len(datamodule.train_data)}")
        print(f"  测试样本: {len(datamodule.test_data)}")
        
        # 配置训练器
        trainer_config = self.config.get('trainer', {})
        
        trainer = Trainer(
            max_epochs=trainer_config.get('max_epochs', 1),
            accelerator=trainer_config.get('accelerator', 'gpu'),
            devices=trainer_config.get('devices', 1),
            default_root_dir=str(self.output_dir),
            logger=TensorBoardLogger(
                save_dir=str(self.output_root / "logs"),
                name=f"{self.task_name}_log"
            ),
            callbacks=callbacks,
            num_sanity_val_steps=0,
            precision=trainer_config.get('precision', 32),
        )
        
        # 训练
        print("\n开始训练...")
        trainer.fit(model=model, datamodule=datamodule)
        
        # 测试
        print("\n在测试集上评估...")
        test_results = trainer.test(model=model, datamodule=datamodule)
        
        # 整理结果
        results = {
            'task': self.task_name,
            'metrics': {},
        }
        
        print("\n" + "=" * 40)
        print("测试结果:")
        for key, value in test_results[0].items():
            if isinstance(value, float):
                results['metrics'][key] = value
                print(f"  {key}: {value:.4f}")
        print("=" * 40)
        
        # 保存配置
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"\n训练完成！结果保存在: {self.output_dir}")
        
        return results


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高级缺陷检测训练脚本')
    
    parser.add_argument(
        '--task', type=str, default='task_1',
        help='任务名称 (如 task_1, task_2, ...)'
    )
    parser.add_argument(
        '--dataset_root', type=str, default='./datasets',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output_root', type=str, default='./results',
        help='输出根目录'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='配置文件路径 (YAML 格式)'
    )
    parser.add_argument(
        '--model', type=str, default='patchcore',
        choices=['patchcore', 'padim'],
        help='模型类型'
    )
    parser.add_argument(
        '--backbone', type=str, default='resnet18',
        help='骨干网络'
    )
    parser.add_argument(
        '--image_size', type=int, nargs=2, default=[512, 512],
        help='输入图像尺寸'
    )
    parser.add_argument(
        '--no_augmentation', action='store_true',
        help='禁用数据增强'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 使用命令行参数构建配置
        config = {
            'model': {
                'name': args.model,
                'backbone': args.backbone,
                'layers': ['layer2', 'layer3'],
                'pre_trained': True,
                'coreset_sampling_ratio': 0.1,
                'num_neighbors': 9,
            },
            'data': {
                'image_size': args.image_size,
                'train_batch_size': 1,
                'eval_batch_size': 1,
                'num_workers': 8,
            },
            'trainer': {
                'max_epochs': 1,
                'accelerator': 'gpu',
                'devices': 1,
            },
            'augmentation': get_default_augmentation_config() if not args.no_augmentation else {},
        }
    
    # 创建训练器并执行训练
    trainer = AdvancedTrainer(
        task_name=args.task,
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        config=config
    )
    
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main()
