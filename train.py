#!/usr/bin/env python3
"""
自定义数据集完整训练脚本
适用于包含正常数据、异常数据和异常掩码的工业缺陷检测任务

Folder 数据模块期望的目录结构：
./datasets/my_dataset/
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
from anomalib.data import Folder  # 用于自定义数据集的数据模块[6]
from anomalib.models import Patchcore  # 选择Patchcore模型，也支持EfficientAd等[7]
from anomalib.engine import Engine  # 训练引擎[7]
from anomalib.deploy import ExportType  # 模型导出类型[13]

def main():
    # ==================== 1. 配置参数 ====================
    # 数据集路径配置
    dataset_root = Path("./datasets/my_dataset")  # 数据集根目录
    dataset_name = "my_dataset"  # 数据集名称
    normal_dir = "good"  # 正常样本目录名称（相对于root）
    abnormal_dir = "defect"  # 异常样本目录名称（相对于root）
    mask_dir = dataset_root / "mask" / "defect"  # 异常掩码目录（分割任务必需）
    
    # 训练参数配置
    batch_size = 32  # 批次大小
    num_workers = 8  # 数据加载工作线程数
    image_size = (256, 256)  # 输入图像尺寸
    
    # 模型保存路径
    output_dir = Path("results/my_dataset_training")  # 结果保存目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 2. 初始化数据模块 ====================
    print("初始化数据模块...")
    
    # 使用Folder数据模块加载自定义数据集
    # Folder模块期望的目录结构：
    #   dataset_root/
    #   ├── good/          # normal_dir - 正常图像
    #   ├── defect/        # abnormal_dir - 异常图像  
    #   └── mask/defect/   # mask_dir - 异常掩码（与abnormal_dir中图像一一对应）
    datamodule = Folder(
        name=dataset_name,  # 数据集名称
        root=dataset_root,  # 数据集根路径
        normal_dir=normal_dir,  # 正常样本目录（相对路径）
        abnormal_dir=abnormal_dir,  # 异常样本目录（相对路径）
        mask_dir=mask_dir,  # 掩码目录（分割任务必需，绝对路径）
        image_size=image_size,  # 调整图像尺寸
        train_batch_size=batch_size,  # 训练批次大小
        eval_batch_size=batch_size,  # 评估批次大小
        num_workers=num_workers,  # 数据加载线程数
        task="segmentation",  # 分割任务（需要mask_dir）
    )
    
    # 设置数据划分（自动识别训练集、测试集）
    datamodule.setup()
    
    print(f"数据集信息：")
    print(f"  训练集大小：{len(datamodule.train_data)}")
    print(f"  测试集大小：{len(datamodule.test_data)}")
    
    # ==================== 3. 初始化模型 ====================
    print("初始化模型...")
    
    # 使用Patchcore模型，这是一种无监督异常检测模型，适合工业缺陷检测
    # 也可以替换为EfficientAd：model = EfficientAd(teacher_out_channels=384)
    model = Patchcore(
        # 可自定义模型参数，例如：
        # backbone="wide_resnet50_2",  # 骨干网络
        # num_neighbors=9,  # 最近邻数量
    )
    
    # ==================== 4. 初始化训练引擎 ====================
    print("初始化训练引擎...")
    
    engine = Engine(
        max_epochs=1,  # Patchcore通常只需1个epoch
        # 对于EfficientAd等模型可能需要更多epochs，如max_epochs=200
        accelerator="auto",  # 自动检测GPU/CPU
        devices=1,  # 使用设备数量
        default_root_dir=output_dir,  # 结果保存目录
        # 可添加日志记录器，如：
        # logger=TensorBoardLogger(save_dir="logs/"),
    )
    
    # ==================== 5. 执行训练 ====================
    print("开始训练...")
    engine.fit(model=model, datamodule=datamodule)
    
    # ==================== 6. 测试评估 ====================
    print("在测试集上评估模型性能...")
    test_results = engine.test(model=model, datamodule=datamodule)
    
    # 打印评估指标
    print("测试结果：")
    for key, value in test_results[0].items():
        if "image" in key or "pixel" in key or "AUROC" in key:
            print(f"  {key}: {value:.4f}")
    
    # ==================== 7. 导出模型（可选） ====================
    print("导出模型为优化格式...")
    
    # 导出为OpenVINO格式以加速推理
    engine.export(
        model=model,
        export_root=output_dir / "exported_model",
        input_size=image_size,
        export_type=ExportType.OPENVINO,  # 导出为OpenVINO格式
    )
    
    print(f"训练完成！结果保存在：{output_dir}")
    print(f"优化模型保存在：{output_dir / 'exported_model'}")

if __name__ == "__main__":
    main()
