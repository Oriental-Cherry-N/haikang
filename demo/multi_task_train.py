#!/usr/bin/env python3
"""
多任务批量训练脚本
支持同时训练多个任务，并生成汇总报告

适用于 anomalib 0.7.x
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
from tqdm import tqdm

# 导入高级训练模块
from advanced_train import AdvancedTrainer, get_default_augmentation_config


class MultiTaskTrainer:
    """
    多任务训练管理器
    """
    
    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            dataset_root: 数据集根目录
            output_root: 输出根目录
            config: 全局配置
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.config = config or self._get_default_config()
        
        # 结果存储
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model': {
                'name': 'patchcore',
                'backbone': 'resnet18',
                'layers': ['layer2', 'layer3'],
                'pre_trained': True,
                'coreset_sampling_ratio': 0.1,
                'num_neighbors': 9,
            },
            'data': {
                'image_size': [512, 512],
                'train_batch_size': 1,
                'eval_batch_size': 1,
                'num_workers': 8,
            },
            'trainer': {
                'max_epochs': 1,
                'accelerator': 'gpu',
                'devices': 1,
            },
            'augmentation': get_default_augmentation_config(),
        }
    
    def discover_tasks(self) -> List[str]:
        """
        发现所有可用任务
        
        Returns:
            任务名称列表
        """
        tasks = []
        
        for item in sorted(self.dataset_root.iterdir()):
            if item.is_dir() and item.name.startswith('task_'):
                # 检查是否有完整的数据结构
                good_dir = item / 'good'
                defect_dir = item / 'defect'
                mask_dir = item / 'mask' / 'defect'
                
                if good_dir.exists() and defect_dir.exists():
                    tasks.append(item.name)
        
        return tasks
    
    def train_single_task(
        self,
        task_name: str,
        task_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        训练单个任务
        
        Args:
            task_name: 任务名称
            task_config: 任务特定配置（可覆盖全局配置）
        
        Returns:
            训练结果
        """
        # 合并配置
        config = self.config.copy()
        if task_config:
            config = self._merge_config(config, task_config)
        
        # 创建训练器
        trainer = AdvancedTrainer(
            task_name=task_name,
            dataset_root=self.dataset_root,
            output_root=self.output_root,
            config=config
        )
        
        # 执行训练
        result = trainer.train()
        
        return result
    
    def _merge_config(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """深度合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def train_all(
        self,
        tasks: Optional[List[str]] = None,
        skip_existing: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        训练所有任务
        
        Args:
            tasks: 任务列表（None 表示所有任务）
            skip_existing: 是否跳过已训练的任务
        
        Returns:
            所有任务的结果
        """
        self.start_time = datetime.now()
        
        # 发现任务
        if tasks is None:
            tasks = self.discover_tasks()
        
        print("=" * 60)
        print(f"多任务训练")
        print(f"数据集目录: {self.dataset_root}")
        print(f"输出目录: {self.output_root}")
        print(f"任务数量: {len(tasks)}")
        print(f"任务列表: {tasks}")
        print("=" * 60)
        
        # 逐个训练
        for task in tqdm(tasks, desc="任务进度"):
            print(f"\n{'#' * 60}")
            print(f"# 任务: {task}")
            print(f"{'#' * 60}")
            
            # 检查是否跳过
            if skip_existing:
                model_path = self.output_root / task / 'weights' / 'model.ckpt'
                if model_path.exists():
                    print(f"跳过已存在的任务: {task}")
                    continue
            
            try:
                result = self.train_single_task(task)
                self.results[task] = {
                    'status': 'success',
                    'metrics': result.get('metrics', {}),
                }
            except Exception as e:
                print(f"任务 {task} 训练失败: {e}")
                self.results[task] = {
                    'status': 'failed',
                    'error': str(e),
                }
        
        self.end_time = datetime.now()
        
        # 生成汇总报告
        self._generate_summary_report()
        
        return self.results
    
    def _generate_summary_report(self):
        """生成汇总报告"""
        report_dir = self.output_root / 'summary'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算统计信息
        successful_tasks = [t for t, r in self.results.items() if r['status'] == 'success']
        failed_tasks = [t for t, r in self.results.items() if r['status'] == 'failed']
        
        # 收集所有指标
        all_metrics = {}
        for task, result in self.results.items():
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append((task, value))
        
        # 计算平均值
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = sum(v for _, v in values) / len(values)
        
        # 构建报告
        report = {
            'summary': {
                'total_tasks': len(self.results),
                'successful': len(successful_tasks),
                'failed': len(failed_tasks),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration': str(self.end_time - self.start_time) if self.start_time and self.end_time else None,
            },
            'average_metrics': avg_metrics,
            'task_results': self.results,
            'config': self.config,
        }
        
        # 保存 JSON 报告
        json_path = report_dir / 'training_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存文本报告
        txt_path = report_dir / 'training_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("多任务训练汇总报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"开始时间: {self.start_time}\n")
            f.write(f"结束时间: {self.end_time}\n")
            f.write(f"总耗时: {self.end_time - self.start_time}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("任务统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  总任务数: {len(self.results)}\n")
            f.write(f"  成功: {len(successful_tasks)}\n")
            f.write(f"  失败: {len(failed_tasks)}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("平均指标:\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in avg_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write("各任务详情:\n")
            f.write("-" * 40 + "\n")
            
            for task, result in sorted(self.results.items()):
                f.write(f"\n{task}:\n")
                f.write(f"  状态: {result['status']}\n")
                
                if result['status'] == 'success':
                    metrics = result.get('metrics', {})
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
                else:
                    f.write(f"  错误: {result.get('error', 'Unknown')}\n")
        
        # 保存 CSV 格式的结果（便于分析）
        csv_path = report_dir / 'metrics.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 获取所有指标名称
            metric_names = set()
            for result in self.results.values():
                if result['status'] == 'success':
                    metric_names.update(result.get('metrics', {}).keys())
            metric_names = sorted(metric_names)
            
            # 写入表头
            f.write('task,status,' + ','.join(metric_names) + '\n')
            
            # 写入数据
            for task, result in sorted(self.results.items()):
                row = [task, result['status']]
                if result['status'] == 'success':
                    metrics = result.get('metrics', {})
                    for name in metric_names:
                        value = metrics.get(name, '')
                        row.append(f"{value:.4f}" if isinstance(value, float) else str(value))
                else:
                    row.extend([''] * len(metric_names))
                f.write(','.join(row) + '\n')
        
        print("\n" + "=" * 60)
        print("训练完成！汇总报告已保存:")
        print(f"  - JSON: {json_path}")
        print(f"  - TXT: {txt_path}")
        print(f"  - CSV: {csv_path}")
        print("=" * 60)
        
        # 打印摘要
        print("\n平均指标:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多任务批量训练脚本')
    
    parser.add_argument(
        '--dataset_root', type=str, default='./datasets',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output_root', type=str, default='./results',
        help='输出根目录'
    )
    parser.add_argument(
        '--tasks', type=str, nargs='+', default=None,
        help='指定任务列表（如 task_1 task_2），不指定则训练所有任务'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='配置文件路径'
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
        '--skip_existing', action='store_true',
        help='跳过已存在的训练结果'
    )
    parser.add_argument(
        '--no_augmentation', action='store_true',
        help='禁用数据增强'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载或构建配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
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
            'augmentation': {} if args.no_augmentation else get_default_augmentation_config(),
        }
    
    # 创建多任务训练器
    trainer = MultiTaskTrainer(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        config=config
    )
    
    # 处理任务参数
    if args.tasks and args.tasks[0].lower() == 'all':
        tasks = None  # 自动发现所有任务
    else:
        tasks = args.tasks
    
    # 执行训练
    results = trainer.train_all(
        tasks=tasks,
        skip_existing=args.skip_existing
    )
    
    return results


if __name__ == "__main__":
    main()
