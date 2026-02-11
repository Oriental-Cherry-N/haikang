#!/usr/bin/env python3
"""
快速入门示例 - 小样本缺陷检测完整流程

此脚本演示从训练到推理的完整流程:
1. 数据准备与增强
2. 模型训练
3. 自适应推理
4. 可解释性分析

适用于 anomalib 0.7.x
"""

import os
import sys
from pathlib import Path

# 添加 demo 目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def example_1_basic_training():
    """
    示例1: 基础训练
    使用 PatchCore 模型训练单个任务
    """
    print("\n" + "=" * 60)
    print("示例 1: 基础训练")
    print("=" * 60)
    
    from pathlib import Path
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    
    from anomalib.data.folder import Folder
    from anomalib.data.task_type import TaskType
    from anomalib.models.patchcore import Patchcore
    from anomalib.post_processing import NormalizationMethod, ThresholdMethod
    from anomalib.utils.callbacks import (
        MetricsConfigurationCallback,
        PostProcessingConfigurationCallback,
    )
    
    # 配置
    task_name = "task_1"
    dataset_root = Path("./datasets")
    output_root = Path("./results")
    image_size = (512, 512)
    
    # 1. 初始化数据模块
    print("\n[1/4] 初始化数据模块...")
    datamodule = Folder(
        normal_dir=str(dataset_root / task_name / "good"),
        abnormal_dir=str(dataset_root / task_name / "defect"),
        mask_dir=str(dataset_root / task_name / "mask" / "defect"),
        image_size=image_size,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=8,
        task=TaskType.SEGMENTATION,
    )
    datamodule.setup()
    print(f"   训练集: {len(datamodule.train_data)} 样本")
    print(f"   测试集: {len(datamodule.test_data)} 样本")
    
    # 2. 初始化模型
    print("\n[2/4] 初始化模型...")
    model = Patchcore(
        input_size=image_size,
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,  # 小样本场景适当增大
        num_neighbors=9,
    )
    print("   模型: PatchCore")
    print("   骨干网络: ResNet18")
    
    # 3. 配置回调
    print("\n[3/4] 配置训练器...")
    output_dir = output_root / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        MetricsConfigurationCallback(
            task=TaskType.SEGMENTATION,
            image_metrics=["AUROC", "F1Score"],
            pixel_metrics=["AUROC", "F1Score"],
        ),
        PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
        ),
        ModelCheckpoint(
            dirpath=output_dir / "weights",
            filename="model",
            monitor="pixel_AUROC",
            mode="max",
            save_last=True,
        ),
    ]
    
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(output_dir),
        logger=TensorBoardLogger(save_dir=str(output_root / "logs"), name=f"{task_name}_log"),
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    
    # 4. 训练与测试
    print("\n[4/4] 开始训练...")
    trainer.fit(model=model, datamodule=datamodule)
    
    print("\n评估模型...")
    test_results = trainer.test(model=model, datamodule=datamodule)
    
    print("\n" + "-" * 40)
    print("训练完成! 测试结果:")
    for key, value in test_results[0].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("-" * 40)
    
    return output_dir / "weights" / "model.ckpt"


def example_2_advanced_training():
    """
    示例2: 高级训练（带数据增强）
    使用 AdvancedTrainer 进行训练
    """
    print("\n" + "=" * 60)
    print("示例 2: 高级训练（带数据增强）")
    print("=" * 60)
    
    from advanced_train import AdvancedTrainer, get_default_augmentation_config
    from pathlib import Path
    
    # 配置
    config = {
        'model': {
            'name': 'patchcore',
            'backbone': 'resnet18',
            'layers': ['layer2', 'layer3'],
            'coreset_sampling_ratio': 0.1,
        },
        'data': {
            'image_size': [512, 512],
            'train_batch_size': 1,
            'num_workers': 8,
        },
        'trainer': {
            'max_epochs': 1,
            'accelerator': 'gpu',
            'devices': 1,
        },
        # 启用数据增强
        'augmentation': get_default_augmentation_config(),
    }
    
    # 创建训练器
    trainer = AdvancedTrainer(
        task_name='task_1',
        dataset_root=Path('./datasets'),
        output_root=Path('./results_advanced'),
        config=config
    )
    
    # 执行训练
    results = trainer.train()
    
    return results


def example_3_adaptive_inference():
    """
    示例3: 自适应推理
    使用训练好的模型进行推理，支持置信度估计
    """
    print("\n" + "=" * 60)
    print("示例 3: 自适应推理")
    print("=" * 60)
    
    from adaptive_inference import AdaptiveInferencer
    from pathlib import Path
    from PIL import Image
    import numpy as np
    
    # 模型路径
    model_path = Path("./results/task_1/weights/model.ckpt")
    
    if not model_path.exists():
        print("模型文件不存在，请先运行示例1进行训练")
        return
    
    # 创建推理器
    print("\n[1/3] 加载模型...")
    inferencer = AdaptiveInferencer(
        model_path=model_path,
        device='auto',
        enable_adaptive_threshold=True,
        enable_online_learning=False
    )
    print("   模型加载完成")
    
    # 获取测试图像
    test_images = list(Path("./datasets/task_1/defect").glob("*.bmp"))[:5]
    
    if not test_images:
        print("没有找到测试图像")
        return
    
    # 推理
    print(f"\n[2/3] 推理 {len(test_images)} 张图像...")
    
    for img_path in test_images:
        result = inferencer.predict(img_path, return_visualization=True)
        
        print(f"\n  {img_path.name}:")
        print(f"    异常分数: {result['anomaly_score']:.4f}")
        print(f"    阈值: {result['threshold']:.4f}")
        print(f"    预测: {result['pred_label']}")
        print(f"    置信度: {result['confidence']:.2%}")
        print(f"    不确定性: {result['uncertainty']:.2%}")
    
    # 统计
    print(f"\n[3/3] 推理统计:")
    stats = inferencer.get_statistics()
    print(f"    总推理数: {stats['total_inferences']}")
    print(f"    异常数: {stats['anomaly_count']}")
    print(f"    正常数: {stats['normal_count']}")
    print(f"    异常率: {stats['anomaly_rate']:.2%}")


def example_4_explainability():
    """
    示例4: 可解释性分析
    生成检测结果的可视化解释
    """
    print("\n" + "=" * 60)
    print("示例 4: 可解释性分析")
    print("=" * 60)
    
    from explainability import ExplainabilityAnalyzer
    from pathlib import Path
    
    # 模型路径
    model_path = Path("./results/task_1/weights/model.ckpt")
    
    if not model_path.exists():
        print("模型文件不存在，请先运行示例1进行训练")
        return
    
    # 创建分析器
    print("\n[1/3] 加载分析器...")
    analyzer = ExplainabilityAnalyzer(
        model_path=model_path,
        device='auto'
    )
    
    # 获取测试图像
    test_image = list(Path("./datasets/task_1/defect").glob("*.bmp"))[0]
    
    if not test_image:
        print("没有找到测试图像")
        return
    
    # 分析
    print(f"\n[2/3] 分析图像: {test_image.name}")
    output_dir = Path("./explainability_results/example")
    
    report, visualizations = analyzer.analyze(
        test_image,
        output_dir=output_dir,
        save_all=True
    )
    
    # 显示结果
    print(f"\n[3/3] 分析结果:")
    print(f"    预测: {report.prediction}")
    print(f"    异常分数: {report.anomaly_score:.4f}")
    print(f"    置信度: {report.confidence:.2%}")
    print(f"    异常区域数: {len(report.regions)}")
    
    print(f"\n    摘要: {report.summary}")
    
    if report.regions:
        print("\n    异常区域详情:")
        for region in report.regions[:3]:
            print(f"      #{region.id}: {region.severity} - "
                  f"面积={region.area}px, 分数={region.score:.3f}")
    
    print(f"\n    结果保存到: {output_dir}")


def example_5_multi_task():
    """
    示例5: 多任务批量训练
    同时训练多个任务
    """
    print("\n" + "=" * 60)
    print("示例 5: 多任务批量训练")
    print("=" * 60)
    
    from multi_task_train import MultiTaskTrainer
    from pathlib import Path
    
    # 配置
    config = {
        'model': {
            'name': 'patchcore',
            'backbone': 'resnet18',
            'layers': ['layer2', 'layer3'],
            'coreset_sampling_ratio': 0.1,
        },
        'data': {
            'image_size': [512, 512],
            'train_batch_size': 1,
        },
        'trainer': {
            'max_epochs': 1,
            'accelerator': 'gpu',
            'devices': 1,
        },
    }
    
    # 创建多任务训练器
    trainer = MultiTaskTrainer(
        dataset_root=Path('./datasets'),
        output_root=Path('./results_multi'),
        config=config
    )
    
    # 发现所有任务
    tasks = trainer.discover_tasks()
    print(f"\n发现 {len(tasks)} 个任务: {tasks}")
    
    # 只训练前2个任务作为演示
    demo_tasks = tasks[:2] if len(tasks) > 2 else tasks
    print(f"演示训练 {len(demo_tasks)} 个任务: {demo_tasks}")
    
    # 执行训练
    results = trainer.train_all(tasks=demo_tasks)
    
    return results


def main():
    """主函数 - 运行所有示例"""
    print("=" * 60)
    print(" 小样本缺陷检测 - 快速入门示例")
    print(" 基于 anomalib 0.7.x")
    print("=" * 60)
    
    print("""
可用示例:
    1. 基础训练 - 使用 PatchCore 训练单个任务
    2. 高级训练 - 带数据增强的训练
    3. 自适应推理 - 支持置信度估计的推理
    4. 可解释性分析 - 生成检测结果的可视化解释
    5. 多任务训练 - 批量训练多个任务
    
请选择要运行的示例 (输入数字 1-5，或 'all' 运行全部):
    """)
    
    choice = input(">>> ").strip().lower()
    
    if choice == '1':
        example_1_basic_training()
    elif choice == '2':
        example_2_advanced_training()
    elif choice == '3':
        example_3_adaptive_inference()
    elif choice == '4':
        example_4_explainability()
    elif choice == '5':
        example_5_multi_task()
    elif choice == 'all':
        # 依次运行
        model_path = example_1_basic_training()
        # example_2_advanced_training()  # 可选
        example_3_adaptive_inference()
        example_4_explainability()
        # example_5_multi_task()  # 耗时较长，可选
    else:
        print("无效选择，请输入 1-5 或 'all'")


if __name__ == "__main__":
    main()
