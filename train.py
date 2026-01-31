#!/usr/bin/env python3
"""
自定义数据集完整训练脚本 (适用于 anomalib 0.7.x)
适用于包含正常数据、异常数据和异常掩码的工业缺陷检测任务

Folder 数据模块期望的目录结构：
./datasets/task_1/
    ├── good/               # 正常图像（用于训练和测试）
    │   ├── 000.png
    │   └── ...
    ├── defect/             # 异常图像（用于测试）
    │   ├── 000.png
    │   └── ...
    └── mask/
        └── defect/         # 异常掩码图像（与defect/中的图像一一对应，分割任务需要）
            ├── 000.png
            └── ...

注意：Folder模块会自动将normal_dir中的图像按比例划分为训练集和测试集
"""

from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from anomalib.models.patchcore import Patchcore  # 直接导入模型类
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)

def main():
    # ==================== 1. 配置参数 ====================
    # 数据集路径配置
    dataset_root = Path("./datasets/task_1")  # 任务1数据集
    normal_dir = dataset_root / "good"  # 正常样本目录（绝对路径）
    abnormal_dir = dataset_root / "defect"  # 异常样本目录（绝对路径）
    mask_dir = dataset_root / "mask" / "defect"  # 异常掩码目录（分割任务必需）
    
    # 训练参数配置
    batch_size = 1  # 批次大小
    num_workers = 8  # 数据加载工作线程数
    image_size = (1024, 1024)  # 输入图像尺寸
    
    # 模型保存路径
    output_dir = Path("results/task_1")  # 结果保存目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 2. 初始化数据模块 ====================
    print("初始化数据模块...")
    
    datamodule = Folder(
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task=TaskType.SEGMENTATION,
    )
    datamodule.setup()
    
    print(f"数据集信息：")
    print(f"  训练集大小：{len(datamodule.train_data)}")
    print(f"  测试集大小：{len(datamodule.test_data)}")
    
    # ==================== 3. 初始化模型 ====================
    print("初始化模型...")
    
    # anomalib 0.7.x: 直接实例化 Patchcore，所有参数必须显式指定
    model = Patchcore(
        input_size=image_size,
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=9,
    )
    
    # ==================== 4. 配置回调函数 ====================
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
    
    # ==================== 5. 初始化训练器 ====================
    print("初始化训练器...")
    
    trainer = Trainer(
        max_epochs=1,  # Patchcore 通常只需1个epoch
        accelerator="gpu",
        devices=1,
        default_root_dir=output_dir,
        logger=TensorBoardLogger(save_dir="logs/", name="task_1_log_test"),
        callbacks=callbacks,
        num_sanity_val_steps=0,  # 禁用 sanity check，Patchcore 需要先完成训练才能验证
    )
    
    # ==================== 6. 执行训练 ====================
    print("开始训练...")
    trainer.fit(model=model, datamodule=datamodule)
    
    # ==================== 7. 测试评估 ====================
    print("在测试集上评估模型性能...")
    test_results = trainer.test(model=model, datamodule=datamodule)
    
    # 打印评估指标
    print("测试结果：")
    for key, value in test_results[0].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    print(f"\n训练完成！结果保存在：{output_dir}")

if __name__ == "__main__":
    main()
