#!/usr/bin/env python3
"""
可解释性分析脚本 - 异常检测结果可视化与解释

特性:
1. 多级别可视化（图像级、区域级、像素级）
2. 异常区域提取与分析
3. 决策解释报告生成
4. 对比分析（与正常样本）

适用于 anomalib 0.7.x
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from anomalib.deploy import TorchInferencer


# ==================== 数据类定义 ====================

@dataclass
class AnomalyRegion:
    """异常区域数据类"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    score: float
    centroid: Tuple[int, int]
    severity: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExplanationReport:
    """解释报告数据类"""
    image_path: str
    prediction: str
    anomaly_score: float
    threshold: float
    confidence: float
    regions: List[AnomalyRegion]
    summary: str
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['regions'] = [r.to_dict() if isinstance(r, AnomalyRegion) else r for r in self.regions]
        return d


# ==================== 异常区域提取器 ====================

class AnomalyRegionExtractor:
    """
    异常区域提取器
    从异常热力图中提取并分析异常区域
    """
    
    def __init__(
        self,
        threshold_ratio: float = 0.5,
        min_area: int = 100,
        max_regions: int = 10
    ):
        """
        Args:
            threshold_ratio: 阈值比例（相对于最大值）
            min_area: 最小区域面积（像素）
            max_regions: 最大返回区域数
        """
        self.threshold_ratio = threshold_ratio
        self.min_area = min_area
        self.max_regions = max_regions
    
    def extract(
        self,
        anomaly_map: np.ndarray,
        score_threshold: Optional[float] = None
    ) -> List[AnomalyRegion]:
        """
        提取异常区域
        
        Args:
            anomaly_map: 异常热力图 [H, W]
            score_threshold: 分数阈值
        
        Returns:
            异常区域列表
        """
        # 归一化
        map_min = anomaly_map.min()
        map_max = anomaly_map.max()
        
        if map_max - map_min < 1e-8:
            return []
        
        normalized = (anomaly_map - map_min) / (map_max - map_min)
        
        # 计算阈值
        if score_threshold is None:
            threshold = self.threshold_ratio
        else:
            threshold = (score_threshold - map_min) / (map_max - map_min + 1e-8)
            threshold = max(0.3, min(threshold, 0.9))
        
        # 二值化
        binary = (normalized > threshold).astype(np.uint8) * 255
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域分析
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # 边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 质心
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # 区域内的最大异常分数
            region_mask = np.zeros_like(anomaly_map, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            region_scores = anomaly_map[region_mask > 0]
            max_score = float(region_scores.max()) if len(region_scores) > 0 else 0
            
            # 严重程度分级
            if max_score > 0.7 * map_max:
                severity = 'high'
            elif max_score > 0.4 * map_max:
                severity = 'medium'
            else:
                severity = 'low'
            
            regions.append(AnomalyRegion(
                id=i + 1,
                bbox=(x, y, w, h),
                area=int(area),
                score=max_score,
                centroid=(cx, cy),
                severity=severity
            ))
        
        # 按分数排序，取前 N 个
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions[:self.max_regions]


# ==================== 可视化生成器 ====================

class VisualizationGenerator:
    """
    可视化生成器
    创建多种类型的可解释性可视化
    """
    
    def __init__(
        self,
        colormap: str = 'jet',
        alpha: float = 0.5,
        font_scale: float = 0.6
    ):
        """
        Args:
            colormap: 颜色映射
            alpha: 叠加透明度
            font_scale: 字体缩放
        """
        self.colormap = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
        self.alpha = alpha
        self.font_scale = font_scale
    
    def create_heatmap_overlay(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray
    ) -> np.ndarray:
        """
        创建热力图叠加
        
        Args:
            image: 原始图像 [H, W, 3]
            anomaly_map: 异常图 [H, W]
        
        Returns:
            叠加后的图像
        """
        h, w = image.shape[:2]
        
        # 调整大小
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h))
        
        # 归一化到 0-255
        map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        map_uint8 = (map_normalized * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(map_uint8, self.colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加
        overlay = cv2.addWeighted(image, 1 - self.alpha, heatmap, self.alpha, 0)
        
        return overlay
    
    def create_region_visualization(
        self,
        image: np.ndarray,
        regions: List[AnomalyRegion]
    ) -> np.ndarray:
        """
        创建区域可视化
        
        Args:
            image: 原始图像
            regions: 异常区域列表
        
        Returns:
            标注后的图像
        """
        output = image.copy()
        
        # 颜色定义
        colors = {
            'high': (255, 0, 0),     # 红色
            'medium': (255, 165, 0),  # 橙色
            'low': (255, 255, 0),     # 黄色
        }
        
        for region in regions:
            x, y, w, h = region.bbox
            color = colors.get(region.severity, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"#{region.id} ({region.severity})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                         self.font_scale, 2)[0]
            
            # 标签背景
            cv2.rectangle(output, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0] + 10, y),
                         color, -1)
            
            # 标签文字
            cv2.putText(output, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (255, 255, 255), 2)
            
            # 绘制质心
            cv2.circle(output, region.centroid, 5, color, -1)
        
        return output
    
    def create_comparison_view(
        self,
        original: np.ndarray,
        anomaly_map: np.ndarray,
        regions: List[AnomalyRegion],
        result: Dict[str, Any]
    ) -> np.ndarray:
        """
        创建对比视图
        
        Args:
            original: 原始图像
            anomaly_map: 异常图
            regions: 异常区域
            result: 推理结果
        
        Returns:
            对比视图图像
        """
        h, w = original.shape[:2]
        
        # 创建子图
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('原始图像', fontsize=12)
        ax1.axis('off')
        
        # 2. 异常热力图
        ax2 = fig.add_subplot(gs[0, 1])
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
        else:
            anomaly_map_resized = anomaly_map
        im = ax2.imshow(anomaly_map_resized, cmap='jet')
        ax2.set_title('异常热力图', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. 热力图叠加
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = self.create_heatmap_overlay(original, anomaly_map)
        ax3.imshow(overlay)
        ax3.set_title('叠加视图', fontsize=12)
        ax3.axis('off')
        
        # 4. 区域标注
        ax4 = fig.add_subplot(gs[1, 0])
        region_vis = self.create_region_visualization(original, regions)
        ax4.imshow(region_vis)
        ax4.set_title(f'异常区域 (共 {len(regions)} 个)', fontsize=12)
        ax4.axis('off')
        
        # 5. 分数分布
        ax5 = fig.add_subplot(gs[1, 1])
        map_flat = anomaly_map_resized.flatten()
        ax5.hist(map_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(result.get('threshold', 0.5), color='red', linestyle='--', 
                   linewidth=2, label=f"阈值: {result.get('threshold', 0.5):.3f}")
        ax5.set_xlabel('异常分数')
        ax5.set_ylabel('像素数量')
        ax5.set_title('分数分布', fontsize=12)
        ax5.legend()
        
        # 6. 结果摘要
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        summary_text = f"""
检测结果摘要
{'=' * 30}

预测结果: {result.get('pred_label', 'N/A').upper()}
异常分数: {result.get('anomaly_score', 0):.4f}
决策阈值: {result.get('threshold', 0):.4f}
置信度:   {result.get('confidence', 0):.2%}

异常区域统计:
- 总数量: {len(regions)}
- 高严重度: {sum(1 for r in regions if r.severity == 'high')}
- 中严重度: {sum(1 for r in regions if r.severity == 'medium')}
- 低严重度: {sum(1 for r in regions if r.severity == 'low')}

区域详情:
"""
        for region in regions[:5]:
            summary_text += f"\n  #{region.id}: 面积={region.area}px, 分数={region.score:.3f}"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 转换为图像数组
        fig.canvas.draw()
        comparison = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        comparison = comparison.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return comparison
    
    def create_multi_scale_view(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        scales: List[float] = [1.0, 2.0, 4.0]
    ) -> np.ndarray:
        """
        创建多尺度视图
        
        Args:
            image: 原始图像
            anomaly_map: 异常图
            scales: 放大倍数列表
        
        Returns:
            多尺度视图图像
        """
        h, w = image.shape[:2]
        
        # 找到最大异常区域的中心
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h))
        
        max_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        cy, cx = max_idx
        
        # 创建子图
        n_scales = len(scales)
        fig, axes = plt.subplots(2, n_scales, figsize=(5 * n_scales, 10))
        
        overlay = self.create_heatmap_overlay(image, anomaly_map)
        
        for i, scale in enumerate(scales):
            # 计算裁剪区域
            crop_h = int(h / scale)
            crop_w = int(w / scale)
            
            y1 = max(0, cy - crop_h // 2)
            y2 = min(h, y1 + crop_h)
            x1 = max(0, cx - crop_w // 2)
            x2 = min(w, x1 + crop_w)
            
            # 原图裁剪
            crop_image = image[y1:y2, x1:x2]
            axes[0, i].imshow(crop_image)
            axes[0, i].set_title(f'原图 (x{scale:.1f})', fontsize=10)
            axes[0, i].axis('off')
            
            # 叠加裁剪
            crop_overlay = overlay[y1:y2, x1:x2]
            axes[1, i].imshow(crop_overlay)
            axes[1, i].set_title(f'叠加 (x{scale:.1f})', fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 转换为图像数组
        fig.canvas.draw()
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return result


# ==================== 解释报告生成器 ====================

class ExplanationReportGenerator:
    """
    解释报告生成器
    生成人类可读的检测结果解释
    """
    
    def __init__(self):
        self.region_extractor = AnomalyRegionExtractor()
        self.visualization_generator = VisualizationGenerator()
    
    def generate_report(
        self,
        image_path: Union[str, Path],
        anomaly_map: np.ndarray,
        result: Dict[str, Any]
    ) -> ExplanationReport:
        """
        生成解释报告
        
        Args:
            image_path: 图像路径
            anomaly_map: 异常图
            result: 推理结果
        
        Returns:
            解释报告
        """
        # 提取异常区域
        regions = self.region_extractor.extract(
            anomaly_map,
            score_threshold=result.get('threshold')
        )
        
        # 生成摘要
        summary = self._generate_summary(result, regions)
        
        return ExplanationReport(
            image_path=str(image_path),
            prediction=result.get('pred_label', 'unknown'),
            anomaly_score=result.get('anomaly_score', 0),
            threshold=result.get('threshold', 0),
            confidence=result.get('confidence', 0),
            regions=regions,
            summary=summary
        )
    
    def _generate_summary(
        self,
        result: Dict[str, Any],
        regions: List[AnomalyRegion]
    ) -> str:
        """生成文字摘要"""
        pred = result.get('pred_label', 'unknown')
        score = result.get('anomaly_score', 0)
        threshold = result.get('threshold', 0)
        confidence = result.get('confidence', 0)
        
        if pred == 'normal':
            summary = f"检测结果为【正常】，置信度 {confidence:.1%}。"
            summary += f"异常分数 {score:.3f} 低于阈值 {threshold:.3f}。"
            if regions:
                summary += f"虽有 {len(regions)} 个疑似区域，但均未超过判定标准。"
        else:
            summary = f"检测结果为【异常】，置信度 {confidence:.1%}。"
            summary += f"异常分数 {score:.3f} 超过阈值 {threshold:.3f}。"
            
            high_count = sum(1 for r in regions if r.severity == 'high')
            medium_count = sum(1 for r in regions if r.severity == 'medium')
            low_count = sum(1 for r in regions if r.severity == 'low')
            
            summary += f"\n共发现 {len(regions)} 个异常区域"
            if high_count > 0:
                summary += f"，其中 {high_count} 个高严重度"
            if medium_count > 0:
                summary += f"，{medium_count} 个中等严重度"
            if low_count > 0:
                summary += f"，{low_count} 个低严重度"
            summary += "。"
            
            if regions:
                top_region = regions[0]
                summary += f"\n最显著异常位于 ({top_region.centroid[0]}, {top_region.centroid[1]})，"
                summary += f"面积 {top_region.area} 像素，分数 {top_region.score:.3f}。"
        
        return summary
    
    def save_report(
        self,
        report: ExplanationReport,
        output_dir: Union[str, Path],
        image: np.ndarray,
        anomaly_map: np.ndarray,
        result: Dict[str, Any],
        save_visualizations: bool = True
    ):
        """
        保存报告到文件
        
        Args:
            report: 解释报告
            output_dir: 输出目录
            image: 原始图像
            anomaly_map: 异常图
            result: 推理结果
            save_visualizations: 是否保存可视化
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(report.image_path).stem
        
        # 1. 保存 JSON 报告
        json_path = output_dir / f"{image_name}_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 2. 保存文本报告
        txt_path = output_dir / f"{image_name}_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("异常检测解释报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"图像: {report.image_path}\n")
            f.write(f"预测: {report.prediction.upper()}\n")
            f.write(f"异常分数: {report.anomaly_score:.4f}\n")
            f.write(f"阈值: {report.threshold:.4f}\n")
            f.write(f"置信度: {report.confidence:.2%}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("摘要:\n")
            f.write("-" * 40 + "\n")
            f.write(report.summary + "\n\n")
            
            if report.regions:
                f.write("-" * 40 + "\n")
                f.write("异常区域详情:\n")
                f.write("-" * 40 + "\n")
                for region in report.regions:
                    f.write(f"\n区域 #{region.id}:\n")
                    f.write(f"  - 位置: ({region.bbox[0]}, {region.bbox[1]})\n")
                    f.write(f"  - 尺寸: {region.bbox[2]} x {region.bbox[3]}\n")
                    f.write(f"  - 面积: {region.area} 像素\n")
                    f.write(f"  - 分数: {region.score:.4f}\n")
                    f.write(f"  - 严重度: {region.severity}\n")
        
        # 3. 保存可视化
        if save_visualizations:
            # 热力图叠加
            overlay = self.visualization_generator.create_heatmap_overlay(image, anomaly_map)
            overlay_path = output_dir / f"{image_name}_heatmap.png"
            Image.fromarray(overlay).save(overlay_path)
            
            # 区域标注
            region_vis = self.visualization_generator.create_region_visualization(
                image, report.regions
            )
            region_path = output_dir / f"{image_name}_regions.png"
            Image.fromarray(region_vis).save(region_path)
            
            # 对比视图
            comparison = self.visualization_generator.create_comparison_view(
                image, anomaly_map, report.regions, result
            )
            comparison_path = output_dir / f"{image_name}_comparison.png"
            Image.fromarray(comparison).save(comparison_path)


# ==================== 可解释性分析器 ====================

class ExplainabilityAnalyzer:
    """
    可解释性分析器
    主要入口类
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto'
    ):
        """
        Args:
            model_path: 模型路径
            device: 推理设备
        """
        self.model_path = Path(model_path)
        
        # 加载模型
        self.inferencer = TorchInferencer(
            path=str(self.model_path),
            device=device
        )
        
        # 初始化组件
        self.region_extractor = AnomalyRegionExtractor()
        self.visualization_generator = VisualizationGenerator()
        self.report_generator = ExplanationReportGenerator()
    
    def analyze(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_all: bool = True
    ) -> Tuple[ExplanationReport, Dict[str, np.ndarray]]:
        """
        分析单张图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            save_all: 是否保存所有结果
        
        Returns:
            (解释报告, 可视化字典)
        """
        image_path = Path(image_path)
        
        # 加载图像
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # 推理
        result = self.inferencer.predict(image)
        anomaly_map = result.anomaly_map
        
        # 构建结果字典
        result_dict = {
            'anomaly_score': float(result.pred_score),
            'threshold': 0.5,  # 默认阈值
            'pred_label': 'anomaly' if result.pred_score > 0.5 else 'normal',
            'confidence': min(abs(result.pred_score - 0.5) * 2, 1.0)
        }
        
        # 生成报告
        report = self.report_generator.generate_report(
            image_path, anomaly_map, result_dict
        )
        
        # 生成可视化
        visualizations = {
            'heatmap_overlay': self.visualization_generator.create_heatmap_overlay(
                image, anomaly_map
            ),
            'region_view': self.visualization_generator.create_region_visualization(
                image, report.regions
            ),
            'comparison_view': self.visualization_generator.create_comparison_view(
                image, anomaly_map, report.regions, result_dict
            ),
            'multi_scale_view': self.visualization_generator.create_multi_scale_view(
                image, anomaly_map
            )
        }
        
        # 保存结果
        if output_dir and save_all:
            self.report_generator.save_report(
                report, output_dir, image, anomaly_map, result_dict
            )
            
            # 保存多尺度视图
            output_dir = Path(output_dir)
            multi_scale_path = output_dir / f"{image_path.stem}_multiscale.png"
            Image.fromarray(visualizations['multi_scale_view']).save(multi_scale_path)
        
        return report, visualizations
    
    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_all: bool = True
    ) -> List[ExplanationReport]:
        """
        批量分析
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            save_all: 是否保存所有结果
        
        Returns:
            报告列表
        """
        from tqdm import tqdm
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reports = []
        for path in tqdm(image_paths, desc="分析中"):
            try:
                report, _ = self.analyze(path, output_dir, save_all)
                reports.append(report)
            except Exception as e:
                print(f"分析失败 {path}: {e}")
        
        # 保存汇总报告
        summary_path = output_dir / "summary.json"
        summary = {
            'total': len(reports),
            'anomaly_count': sum(1 for r in reports if r.prediction == 'anomaly'),
            'normal_count': sum(1 for r in reports if r.prediction == 'normal'),
            'reports': [r.to_dict() for r in reports]
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return reports


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可解释性分析脚本')
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='单张图像路径'
    )
    parser.add_argument(
        '--image_dir', type=str, default=None,
        help='图像目录（批量分析）'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./explainability_results',
        help='输出目录'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='推理设备'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建分析器
    print(f"加载模型: {args.model}")
    analyzer = ExplainabilityAnalyzer(
        model_path=args.model,
        device=args.device
    )
    
    output_dir = Path(args.output_dir)
    
    if args.image:
        # 单张图像分析
        print(f"\n分析图像: {args.image}")
        report, visualizations = analyzer.analyze(
            args.image, output_dir, save_all=True
        )
        
        print("\n" + "=" * 50)
        print("分析结果:")
        print("=" * 50)
        print(report.summary)
        print("\n区域数量:", len(report.regions))
        for region in report.regions:
            print(f"  #{region.id}: {region.severity} - 分数 {region.score:.3f}")
        print("=" * 50)
        print(f"\n结果保存到: {output_dir}")
    
    elif args.image_dir:
        # 批量分析
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.bmp')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpg'))
        
        print(f"\n批量分析: {len(image_paths)} 张图像")
        reports = analyzer.analyze_batch(image_paths, output_dir)
        
        anomaly_count = sum(1 for r in reports if r.prediction == 'anomaly')
        print("\n" + "=" * 50)
        print("批量分析结果:")
        print(f"  总数: {len(reports)}")
        print(f"  异常: {anomaly_count}")
        print(f"  正常: {len(reports) - anomaly_count}")
        print("=" * 50)
        print(f"\n结果保存到: {output_dir}")
    
    else:
        print("请指定 --image 或 --image_dir 参数")


if __name__ == "__main__":
    main()
