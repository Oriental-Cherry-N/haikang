#!/usr/bin/env python3
"""
批量训练所有任务的脚本
对 datasets 目录下的所有 task_* 数据集分别进行训练
"""

from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import time


def train_single_task(
    task_dir: Path,
    output_base_dir: Path,
    image_size: tuple = (256, 256),
    batch_size: int = 32,
    num_workers: int = 8,
):
    """
    训练单个任务
    
    Args:
        task_dir: 任务数据集路径 (如 datasets/task_1)
        output_base_dir: 结果保存基础目录
        image_size: 输入图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        test_results: 测试结果
    """
    task_name = task_dir.name
    output_dir = output_base_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"开始训练: {task_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # 初始化数据模块
        print(f"[{task_name}] 初始化数据模块...")
        datamodule = Folder(
            name=task_name,
            root=task_dir,
            normal_dir="good",
            abnormal_dir="defect",
            mask_dir=task_dir / "mask" / "defect",
            image_size=image_size,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            task="segmentation",
        )
        datamodule.setup()
        
        print(f"[{task_name}] 训练集: {len(datamodule.train_data)}, 测试集: {len(datamodule.test_data)}")
        
        # 初始化模型
        print(f"[{task_name}] 初始化 Patchcore 模型...")
        model = Patchcore()
        
        # 初始化训练引擎
        print(f"[{task_name}] 初始化训练引擎...")
        engine = Engine(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            default_root_dir=output_dir,
        )
        
        # 执行训练
        print(f"[{task_name}] 开始训练...")
        engine.fit(model=model, datamodule=datamodule)
        
        # 测试评估
        print(f"[{task_name}] 在测试集上评估...")
        test_results = engine.test(model=model, datamodule=datamodule)
        
        # 打印评估指标
        print(f"\n[{task_name}] 测试结果：")
        metrics = {}
        for key, value in test_results[0].items():
            if "image" in key or "pixel" in key or "AUROC" in key or "F1" in key:
                metrics[key] = value
                print(f"  {key}: {value:.4f}")
        
        # 导出模型
        print(f"[{task_name}] 导出模型...")
        engine.export(
            model=model,
            export_root=output_dir / "exported_model",
            input_size=image_size,
            export_type=ExportType.OPENVINO,
        )
        
        elapsed_time = time.time() - start_time
        print(f"[{task_name}] ✅ 训练完成，耗时: {elapsed_time:.1f}秒")
        
        return {
            "status": "success",
            "metrics": metrics,
            "time": elapsed_time,
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[{task_name}] ❌ 训练失败: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "time": elapsed_time,
        }


def train_all_tasks(
    datasets_dir: Path = Path("./datasets"),
    output_dir: Path = Path("./results"),
    image_size: tuple = (256, 256),
    batch_size: int = 32,
    num_workers: int = 8,
):
    """
    训练所有任务
    
    Args:
        datasets_dir: 数据集目录
        output_dir: 结果保存目录
        image_size: 输入图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
    """
    # 获取所有任务目录
    task_dirs = sorted([
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    ], key=lambda x: int(x.name.split("_")[1]))
    
    print(f"找到 {len(task_dirs)} 个任务: {[d.name for d in task_dirs]}")
    print(f"结果将保存到: {output_dir}")
    print(f"配置: image_size={image_size}, batch_size={batch_size}, num_workers={num_workers}")
    
    total_start_time = time.time()
    results = {}
    
    # 逐个训练
    for task_dir in task_dirs:
        result = train_single_task(
            task_dir=task_dir,
            output_base_dir=output_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results[task_dir.name] = result
    
    # 汇总结果
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("所有任务训练完成！汇总结果：")
    print("=" * 70)
    
    successful = [k for k, v in results.items() if v["status"] == "success"]
    failed = [k for k, v in results.items() if v["status"] == "failed"]
    
    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    
    if successful:
        print("\n成功的任务及其指标：")
        print("-" * 70)
        for task_name in successful:
            metrics = results[task_name]["metrics"]
            time_taken = results[task_name]["time"]
            
            # 提取关键指标
            image_auroc = metrics.get("image_AUROC", 0)
            pixel_auroc = metrics.get("pixel_AUROC", 0)
            
            print(f"{task_name:<12} Image_AUROC: {image_auroc:.4f}, "
                  f"Pixel_AUROC: {pixel_auroc:.4f}, 耗时: {time_taken:.1f}s")
    
    if failed:
        print("\n失败的任务：")
        print("-" * 70)
        for task_name in failed:
            error = results[task_name]["error"]
            print(f"{task_name}: {error}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # 配置参数
    datasets_dir = Path("./datasets")
    output_dir = Path("./results")
    
    # 可根据硬件配置调整这些参数
    image_size = (256, 256)
    batch_size = 32
    num_workers = 8
    
    # 执行批量训练
    results = train_all_tasks(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
