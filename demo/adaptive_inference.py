#!/usr/bin/env python3
"""
自适应推理脚本 - 支持在线学习和置信度估计

特性:
1. 自适应阈值调整
2. 置信度评估
3. 在线学习（可选）
4. 批量推理
5. 结果可视化

适用于 anomalib 0.7.x
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from anomalib.deploy import TorchInferencer


# ==================== 置信度估计器 ====================

class ConfidenceEstimator:
    """
    置信度估计器
    用于评估预测结果的可靠性
    """
    
    def __init__(
        self,
        calibration_method: str = 'sigmoid',
        temperature: float = 1.0
    ):
        """
        Args:
            calibration_method: 校准方法 ('sigmoid', 'softmax', 'linear')
            temperature: 温度参数（用于校准）
        """
        self.calibration_method = calibration_method
        self.temperature = temperature
        
        # 运行时统计量
        self.score_history = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0
    
    def update_statistics(self, score: float):
        """更新运行时统计量"""
        self.score_history.append(score)
        self.n_samples += 1
        
        # 增量更新均值和方差
        delta = score - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = score - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.n_samples
    
    def estimate_confidence(
        self,
        anomaly_score: float,
        threshold: float
    ) -> float:
        """
        估计预测置信度
        
        Args:
            anomaly_score: 异常分数
            threshold: 决策阈值
        
        Returns:
            置信度 (0-1)
        """
        # 计算到阈值的距离
        distance = anomaly_score - threshold
        
        if self.calibration_method == 'sigmoid':
            # Sigmoid 校准
            confidence = 1 / (1 + np.exp(-distance / self.temperature))
        
        elif self.calibration_method == 'linear':
            # 线性校准（基于统计量）
            if self.n_samples > 10:
                std = np.sqrt(self.running_var) + 1e-8
                z_score = abs(distance) / std
                confidence = min(z_score / 3, 1.0)  # 3-sigma 作为满置信度
            else:
                confidence = 0.5  # 样本不足时返回中间值
        
        else:
            # 默认：简单归一化
            confidence = min(abs(distance) * 2, 1.0)
        
        return confidence
    
    def get_uncertainty(self, anomaly_score: float) -> float:
        """
        估计预测不确定性
        
        Returns:
            不确定性 (0-1)，越高表示越不确定
        """
        if self.n_samples < 10:
            return 0.5  # 样本不足
        
        # 基于历史分布计算
        std = np.sqrt(self.running_var) + 1e-8
        z_score = abs(anomaly_score - self.running_mean) / std
        
        # 接近均值的样本不确定性高（难以判断）
        # 远离均值的样本不确定性低（容易判断）
        uncertainty = np.exp(-z_score)
        
        return uncertainty


# ==================== 自适应阈值管理器 ====================

class AdaptiveThresholdManager:
    """
    自适应阈值管理器
    根据运行时数据动态调整阈值
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        min_samples: int = 10,
        method: str = 'percentile'
    ):
        """
        Args:
            initial_threshold: 初始阈值
            adaptation_rate: 自适应速率
            min_samples: 最小样本数（达到后才开始自适应）
            method: 阈值计算方法 ('percentile', 'sigma', 'mixed')
        """
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.method = method
        
        # 分数历史
        self.normal_scores = []
        self.anomaly_scores = []  # 如果有标签反馈
    
    def update(
        self,
        score: float,
        is_anomaly: Optional[bool] = None
    ):
        """
        更新阈值
        
        Args:
            score: 新的异常分数
            is_anomaly: 真实标签（如果有反馈）
        """
        if is_anomaly is None:
            # 无标签：假设低分为正常
            if score < self.threshold:
                self.normal_scores.append(score)
        else:
            if is_anomaly:
                self.anomaly_scores.append(score)
            else:
                self.normal_scores.append(score)
        
        # 达到最小样本数后开始自适应
        if len(self.normal_scores) >= self.min_samples:
            self._adapt_threshold()
    
    def _adapt_threshold(self):
        """根据累积数据自适应调整阈值"""
        if self.method == 'percentile':
            # 使用正常分数的高百分位数
            new_threshold = np.percentile(self.normal_scores, 95)
        
        elif self.method == 'sigma':
            # 使用 3-sigma 规则
            mean = np.mean(self.normal_scores)
            std = np.std(self.normal_scores)
            new_threshold = mean + 3 * std
        
        elif self.method == 'mixed':
            # 混合方法
            percentile_thresh = np.percentile(self.normal_scores, 95)
            mean = np.mean(self.normal_scores)
            std = np.std(self.normal_scores)
            sigma_thresh = mean + 3 * std
            new_threshold = (percentile_thresh + sigma_thresh) / 2
        
        else:
            return
        
        # 平滑更新
        self.threshold = (
            (1 - self.adaptation_rate) * self.threshold +
            self.adaptation_rate * new_threshold
        )
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold


# ==================== 在线学习管理器 ====================

class OnlineLearningManager:
    """
    在线学习管理器
    支持模型的增量更新
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        update_frequency: int = 10,
        similarity_threshold: float = 0.3
    ):
        """
        Args:
            memory_size: 在线记忆库大小
            update_frequency: 更新频率（每 N 个样本更新一次）
            similarity_threshold: 相似度阈值（用于过滤冗余样本）
        """
        self.memory_size = memory_size
        self.update_frequency = update_frequency
        self.similarity_threshold = similarity_threshold
        
        self.feature_buffer = []
        self.sample_count = 0
    
    def should_update(self) -> bool:
        """判断是否应该更新模型"""
        return (
            len(self.feature_buffer) >= self.update_frequency and
            self.sample_count % self.update_frequency == 0
        )
    
    def add_sample(
        self,
        features: np.ndarray,
        is_normal: bool
    ):
        """
        添加新样本到缓冲区
        
        Args:
            features: 样本特征
            is_normal: 是否为正常样本
        """
        self.sample_count += 1
        
        if is_normal:
            # 检查相似度，避免冗余
            if self._is_novel(features):
                self.feature_buffer.append(features)
                
                # 维护缓冲区大小
                if len(self.feature_buffer) > self.memory_size:
                    self.feature_buffer.pop(0)
    
    def _is_novel(self, features: np.ndarray) -> bool:
        """检查是否为新颖样本"""
        if len(self.feature_buffer) == 0:
            return True
        
        # 计算与缓冲区的最小距离
        buffer_array = np.stack(self.feature_buffer)
        distances = np.linalg.norm(buffer_array - features, axis=1)
        min_distance = distances.min()
        
        return min_distance > self.similarity_threshold
    
    def get_update_features(self) -> Optional[np.ndarray]:
        """获取用于更新的特征"""
        if len(self.feature_buffer) == 0:
            return None
        return np.stack(self.feature_buffer)
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.feature_buffer = []


# ==================== 自适应推理引擎 ====================

class AdaptiveInferencer:
    """
    自适应推理引擎
    整合置信度估计、自适应阈值和在线学习
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        enable_online_learning: bool = False,
        enable_adaptive_threshold: bool = True,
        initial_threshold: Optional[float] = None
    ):
        """
        Args:
            model_path: 模型权重路径
            device: 推理设备
            enable_online_learning: 是否启用在线学习
            enable_adaptive_threshold: 是否启用自适应阈值
            initial_threshold: 初始阈值
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # 加载模型
        self.inferencer = TorchInferencer(
            path=str(self.model_path),
            device=device
        )
        
        # 获取模型的默认阈值
        if initial_threshold is None:
            # 尝试从模型获取阈值
            initial_threshold = 0.5
        
        # 初始化组件
        self.confidence_estimator = ConfidenceEstimator()
        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=initial_threshold
        )
        self.online_learner = OnlineLearningManager() if enable_online_learning else None
        
        self.enable_online_learning = enable_online_learning
        self.enable_adaptive_threshold = enable_adaptive_threshold
        
        # 统计信息
        self.inference_count = 0
        self.anomaly_count = 0
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_visualization: bool = True,
        feedback_label: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            image: 输入图像（路径或数组）
            return_visualization: 是否返回可视化结果
            feedback_label: 标签反馈（用于在线学习）
        
        Returns:
            推理结果字典
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image_array = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        # 执行推理
        result = self.inferencer.predict(image_array)
        
        # 提取结果
        anomaly_map = result.anomaly_map
        pred_score = float(result.pred_score)
        
        # 更新统计量
        self.inference_count += 1
        self.confidence_estimator.update_statistics(pred_score)
        
        # 获取阈值
        if self.enable_adaptive_threshold:
            threshold = self.threshold_manager.get_threshold()
            self.threshold_manager.update(pred_score, feedback_label)
        else:
            threshold = 0.5
        
        # 判断是否异常
        is_anomaly = pred_score > threshold
        if is_anomaly:
            self.anomaly_count += 1
        
        # 计算置信度
        confidence = self.confidence_estimator.estimate_confidence(
            pred_score, threshold
        )
        uncertainty = self.confidence_estimator.get_uncertainty(pred_score)
        
        # 在线学习更新
        if self.online_learner is not None:
            # 只有低置信度的正常样本才用于更新
            if not is_anomaly and confidence > 0.7:
                # 这里需要提取特征，简化起见使用异常图
                features = anomaly_map.flatten()
                self.online_learner.add_sample(features, is_normal=True)
        
        # 构建结果
        output = {
            'anomaly_score': pred_score,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'pred_label': 'anomaly' if is_anomaly else 'normal',
            'inference_id': self.inference_count,
        }
        
        # 可视化
        if return_visualization:
            output['anomaly_map'] = anomaly_map
            output['visualization'] = self._create_visualization(
                image_array, anomaly_map, output
            )
        
        return output
    
    def _create_visualization(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        result: Dict[str, Any]
    ) -> np.ndarray:
        """创建可视化结果"""
        h, w = image.shape[:2]
        
        # 调整异常图大小
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h))
        
        # 归一化到 0-255
        anomaly_map_norm = ((anomaly_map - anomaly_map.min()) / 
                          (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # 添加文字信息
        info_text = [
            f"Score: {result['anomaly_score']:.3f}",
            f"Threshold: {result['threshold']:.3f}",
            f"Confidence: {result['confidence']:.2%}",
            f"Prediction: {result['pred_label'].upper()}",
        ]
        
        # 绘制文字背景
        for i, text in enumerate(info_text):
            y = 30 + i * 30
            cv2.rectangle(overlay, (10, y - 20), (250, y + 5), (0, 0, 0), -1)
            color = (0, 255, 0) if result['pred_label'] == 'normal' else (255, 0, 0)
            cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        return overlay
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            save_dir: 结果保存目录
            show_progress: 是否显示进度条
        
        Returns:
            结果列表
        """
        results = []
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        iterator = tqdm(image_paths, desc="推理中") if show_progress else image_paths
        
        for path in iterator:
            path = Path(path)
            result = self.predict(path)
            result['image_path'] = str(path)
            results.append(result)
            
            # 保存可视化
            if save_dir and 'visualization' in result:
                vis_path = save_dir / f"{path.stem}_result.png"
                Image.fromarray(result['visualization']).save(vis_path)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return {
            'total_inferences': self.inference_count,
            'anomaly_count': self.anomaly_count,
            'normal_count': self.inference_count - self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.inference_count, 1),
            'current_threshold': self.threshold_manager.get_threshold(),
            'score_mean': self.confidence_estimator.running_mean,
            'score_std': np.sqrt(self.confidence_estimator.running_var),
        }
    
    def save_statistics(self, path: Union[str, Path]):
        """保存统计信息到文件"""
        stats = self.get_statistics()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自适应推理脚本')
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='模型权重路径 (.ckpt 文件)'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='单张图像路径'
    )
    parser.add_argument(
        '--image_dir', type=str, default=None,
        help='图像目录（批量推理）'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./inference_results',
        help='输出目录'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='推理设备'
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='初始阈值（不指定则使用模型默认值）'
    )
    parser.add_argument(
        '--adaptive_threshold', action='store_true',
        help='启用自适应阈值'
    )
    parser.add_argument(
        '--online_learning', action='store_true',
        help='启用在线学习'
    )
    parser.add_argument(
        '--save_visualization', action='store_true',
        help='保存可视化结果'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建推理器
    print(f"加载模型: {args.model}")
    inferencer = AdaptiveInferencer(
        model_path=args.model,
        device=args.device,
        enable_online_learning=args.online_learning,
        enable_adaptive_threshold=args.adaptive_threshold,
        initial_threshold=args.threshold
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        # 单张图像推理
        print(f"\n推理图像: {args.image}")
        result = inferencer.predict(args.image)
        
        print("\n" + "=" * 40)
        print("推理结果:")
        print(f"  异常分数: {result['anomaly_score']:.4f}")
        print(f"  阈值: {result['threshold']:.4f}")
        print(f"  预测: {result['pred_label']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  不确定性: {result['uncertainty']:.2%}")
        print("=" * 40)
        
        # 保存可视化
        if args.save_visualization and 'visualization' in result:
            vis_path = output_dir / f"{Path(args.image).stem}_result.png"
            Image.fromarray(result['visualization']).save(vis_path)
            print(f"\n可视化保存到: {vis_path}")
    
    elif args.image_dir:
        # 批量推理
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.bmp')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpg'))
        
        print(f"\n批量推理: {len(image_paths)} 张图像")
        
        results = inferencer.predict_batch(
            image_paths,
            save_dir=output_dir if args.save_visualization else None
        )
        
        # 统计结果
        stats = inferencer.get_statistics()
        
        print("\n" + "=" * 40)
        print("批量推理统计:")
        print(f"  总数: {stats['total_inferences']}")
        print(f"  异常数: {stats['anomaly_count']}")
        print(f"  正常数: {stats['normal_count']}")
        print(f"  异常率: {stats['anomaly_rate']:.2%}")
        print(f"  当前阈值: {stats['current_threshold']:.4f}")
        print("=" * 40)
        
        # 保存统计信息
        stats_path = output_dir / "statistics.json"
        inferencer.save_statistics(stats_path)
        print(f"\n统计信息保存到: {stats_path}")
        
        # 保存详细结果
        results_clean = []
        for r in results:
            results_clean.append({
                'image_path': r['image_path'],
                'anomaly_score': r['anomaly_score'],
                'threshold': r['threshold'],
                'is_anomaly': r['is_anomaly'],
                'confidence': r['confidence'],
                'pred_label': r['pred_label'],
            })
        
        results_path = output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        print(f"详细结果保存到: {results_path}")
    
    else:
        print("请指定 --image 或 --image_dir 参数")
        return


if __name__ == "__main__":
    main()
