#!/usr/bin/env python3
"""
数据集整理脚本
将 original_dataset 中的数据整理成 anomalib Folder 模块需要的格式
每个任务独立整理成单独的数据集

原始结构：
original_dataset/
├── 1/
│   ├── 1_ng/           # 异常图像 + 掩码
│   │   ├── xxx.bmp     # 异常图像
│   │   ├── xxx_t.bmp   # 对应掩码
│   │   └── mark.txt
│   └── 1-ok/           # 正常图像
│       └── xxx.bmp
├── 2/
│   ├── 2_ng/
│   └── 2-ok/
...

目标结构（分成10个独立数据集）：
datasets/
├── task_1/
│   ├── good/           # 任务1的正常图像
│   ├── defect/         # 任务1的异常图像
│   └── mask/defect/    # 任务1的掩码
├── task_2/
│   ├── good/
│   ├── defect/
│   └── mask/defect/
...
└── task_10/
    ├── good/
    ├── defect/
    └── mask/defect/
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def organize_single_task(
    task_folder: Path,
    target_dir: Path,
    copy_files: bool = True,
):
    """
    整理单个任务的数据集
    
    Args:
        task_folder: 单个任务文件夹路径 (如 original_dataset/1)
        target_dir: 目标数据集路径 (如 datasets/task_1)
        copy_files: 是否复制文件（False则移动文件）
    
    Returns:
        stats: 统计信息字典
    """
    # 创建目标目录
    good_dir = target_dir / "good"
    defect_dir = target_dir / "defect"
    mask_dir = target_dir / "mask" / "defect"
    
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计计数器
    stats = {
        "good": 0,
        "defect": 0,
        "mask": 0,
    }
    
    file_op = shutil.copy2 if copy_files else shutil.move
    
    # 查找 ok 和 ng 文件夹
    ok_folder = None
    ng_folder = None
    
    for item in task_folder.iterdir():
        if item.is_dir():
            if item.name.endswith("-ok") or item.name.endswith("_ok"):
                ok_folder = item
            elif item.name.endswith("_ng") or item.name.endswith("-ng"):
                ng_folder = item
    
    # 处理正常图像 (ok 文件夹)
    if ok_folder and ok_folder.exists():
        for img_file in ok_folder.iterdir():
            if img_file.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg"]:
                target_path = good_dir / img_file.name
                file_op(str(img_file), str(target_path))
                stats["good"] += 1
    
    # 处理异常图像和掩码 (ng 文件夹)
    if ng_folder and ng_folder.exists():
        for img_file in ng_folder.iterdir():
            if img_file.suffix.lower() not in [".bmp", ".png", ".jpg", ".jpeg"]:
                continue  # 跳过非图像文件（如 mark.txt）
            
            # 判断是掩码还是异常图像
            if img_file.stem.endswith("_t"):
                # 这是掩码文件
                # 去掉 _t 后缀，使文件名与异常图像对应
                original_stem = img_file.stem[:-2]  # 移除 "_t"
                new_name = f"{original_stem}{img_file.suffix}"
                target_path = mask_dir / new_name
                file_op(str(img_file), str(target_path))
                stats["mask"] += 1
            else:
                # 这是异常图像
                target_path = defect_dir / img_file.name
                file_op(str(img_file), str(target_path))
                stats["defect"] += 1
    
    return stats


def organize_all_tasks(
    source_dir: Path,
    target_base_dir: Path,
    copy_files: bool = True,
):
    """
    整理所有任务，每个任务生成独立的数据集
    
    Args:
        source_dir: 原始数据集路径 (original_dataset)
        target_base_dir: 目标数据集基础路径 (datasets)
        copy_files: 是否复制文件（False则移动文件）
    """
    # 获取所有子文件夹（1, 2, 3, ... 10）
    subfolders = sorted(
        [d for d in source_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0
    )
    
    print(f"找到 {len(subfolders)} 个任务: {[d.name for d in subfolders]}")
    print(f"目标目录: {target_base_dir}")
    print(f"操作模式: {'复制' if copy_files else '移动'}")
    print("=" * 70)
    
    all_stats = {}
    
    for subfolder in tqdm(subfolders, desc="整理任务"):
        task_name = subfolder.name
        target_dir = target_base_dir / f"task_{task_name}"
        
        # 整理单个任务
        stats = organize_single_task(
            task_folder=subfolder,
            target_dir=target_dir,
            copy_files=copy_files,
        )
        
        all_stats[task_name] = stats
    
    # 打印总体统计信息
    print("\n" + "=" * 70)
    print("整理完成！各任务统计信息：")
    print("-" * 70)
    print(f"{'任务':<10} {'正常图像':<12} {'异常图像':<12} {'掩码图像':<12} {'状态':<10}")
    print("-" * 70)
    
    total_good = 0
    total_defect = 0
    total_mask = 0
    
    for task_name, stats in all_stats.items():
        status = "✅" if stats['defect'] == stats['mask'] else "⚠️"
        print(f"task_{task_name:<5} {stats['good']:<12} {stats['defect']:<12} {stats['mask']:<12} {status}")
        total_good += stats['good']
        total_defect += stats['defect']
        total_mask += stats['mask']
    
    print("-" * 70)
    print(f"{'总计':<10} {total_good:<12} {total_defect:<12} {total_mask:<12}")
    print("=" * 70)
    
    # 验证所有任务
    print(f"\n数据集已保存到: {target_base_dir}")
    print(f"共生成 {len(all_stats)} 个独立数据集")
    
    return all_stats

def verify_all_datasets(target_base_dir: Path):
    """
    验证所有整理后的数据集
    """
    print("\n" + "=" * 70)
    print("验证所有数据集...")
    print("-" * 70)
    
    task_dirs = sorted([d for d in target_base_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        good_dir = task_dir / "good"
        defect_dir = task_dir / "defect"
        mask_dir = task_dir / "mask" / "defect"
        
        if not all([good_dir.exists(), defect_dir.exists(), mask_dir.exists()]):
            print(f"{task_name}: ❌ 目录结构不完整")
            continue
        
        good_files = set(f.name for f in good_dir.iterdir() if f.is_file())
        defect_files = set(f.name for f in defect_dir.iterdir() if f.is_file())
        mask_files = set(f.name for f in mask_dir.iterdir() if f.is_file())
        
        # 检查异常图像和掩码是否一一对应
        missing_masks = defect_files - mask_files
        extra_masks = mask_files - defect_files
        
        status = "✅" if not missing_masks and not extra_masks else "⚠️"
        print(f"{task_name}: {status} 正常={len(good_files)}, 异常={len(defect_files)}, 掩码={len(mask_files)}")
        
        if missing_masks:
            print(f"  ⚠️ 缺少 {len(missing_masks)} 个掩码")
        if extra_masks:
            print(f"  ⚠️ 多余 {len(extra_masks)} 个掩码")
    
    print("=" * 70)


if __name__ == "__main__":
    # 配置路径
    source_dir = Path("./original_dataset")  # 原始数据集路径
    target_base_dir = Path("./datasets")  # 目标数据集基础路径
    
    # 如果目标目录已存在，询问是否覆盖
    if target_base_dir.exists():
        task_dirs = [d for d in target_base_dir.iterdir() if d.is_dir() and d.name.startswith("task_")]
        if task_dirs:
            response = input(f"目标目录 {target_base_dir} 中已存在 {len(task_dirs)} 个任务数据集，是否删除并重新整理？(y/n): ")
            if response.lower() == 'y':
                for task_dir in task_dirs:
                    shutil.rmtree(task_dir)
                    print(f"已删除 {task_dir.name}")
            else:
                print("操作已取消")
                exit(0)
    
    # 执行整理
    all_stats = organize_all_tasks(
        source_dir=source_dir,
        target_base_dir=target_base_dir,
        copy_files=True,  # 设为 False 可以移动文件而非复制（节省空间）
    )
    
    # 验证结果
    verify_all_datasets(target_base_dir)
