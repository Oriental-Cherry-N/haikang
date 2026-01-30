#!/usr/bin/env python3
"""
修复掩码图像的值范围
将值为 0/1 的掩码转换为 0/255，以便 anomalib 正确读取
"""

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def fix_mask_values(mask_base_dir: Path, dry_run: bool = False):
    """
    修复掩码图像的值范围
    
    Args:
        mask_base_dir: 掩码基础目录 (datasets 目录)
        dry_run: 如果为 True，只检查不修改
    """
    # 查找所有 task 的掩码目录
    task_dirs = sorted([d for d in mask_base_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    
    print(f"找到 {len(task_dirs)} 个任务目录")
    
    total_fixed = 0
    total_skipped = 0
    
    for task_dir in task_dirs:
        mask_dir = task_dir / "mask" / "defect"
        
        if not mask_dir.exists():
            print(f"警告: {mask_dir} 不存在，跳过")
            continue
        
        mask_files = list(mask_dir.glob("*.bmp")) + list(mask_dir.glob("*.png"))
        
        task_fixed = 0
        task_skipped = 0
        
        for mask_path in tqdm(mask_files, desc=f"处理 {task_dir.name}"):
            img = Image.open(mask_path)
            arr = np.array(img)
            
            # 检查是否需要修复
            unique_values = np.unique(arr)
            max_val = arr.max()
            
            if max_val <= 1 and max_val > 0:
                # 需要修复：值范围是 0-1，应该转换为 0-255
                if not dry_run:
                    # 将非零值转换为 255
                    arr_fixed = (arr > 0).astype(np.uint8) * 255
                    img_fixed = Image.fromarray(arr_fixed, mode='L')
                    # 先保存到临时文件，再覆盖
                    temp_path = mask_path.with_suffix('.tmp')
                    img_fixed.save(temp_path)
                    temp_path.replace(mask_path)
                
                task_fixed += 1
            elif max_val > 1:
                # 值大于1但不是255，也需要修复（如值为 2, 3 等）
                if not dry_run:
                    arr_fixed = (arr > 0).astype(np.uint8) * 255
                    img_fixed = Image.fromarray(arr_fixed, mode='L')
                    temp_path = mask_path.with_suffix('.tmp')
                    img_fixed.save(temp_path)
                    temp_path.replace(mask_path)
                
                task_fixed += 1
            elif max_val == 0:
                # 全黑图像，跳过
                task_skipped += 1
            else:
                task_skipped += 1
        
        print(f"  {task_dir.name}: 修复 {task_fixed} 个, 跳过 {task_skipped} 个")
        total_fixed += task_fixed
        total_skipped += task_skipped
    
    print(f"\n总计: 修复 {total_fixed} 个掩码文件, 跳过 {total_skipped} 个")
    
    if dry_run:
        print("\n[DRY RUN] 未实际修改文件")


def verify_mask(mask_dir: Path):
    """验证掩码值范围"""
    mask_files = list(mask_dir.glob("*.bmp"))[:5]
    
    print(f"验证 {mask_dir} 中的掩码...")
    
    for mask_path in mask_files:
        img = Image.open(mask_path)
        arr = np.array(img)
        print(f"  {mask_path.name}: min={arr.min()}, max={arr.max()}, unique={np.unique(arr)}")


if __name__ == "__main__":
    datasets_dir = Path("./datasets")
    
    # 先预览（不修改）
    print("=== 预览模式（不修改文件）===")
    fix_mask_values(datasets_dir, dry_run=True)
    
    # 确认后执行
    response = input("\n是否执行修复？(y/n): ")
    if response.lower() == 'y':
        print("\n=== 执行修复 ===")
        fix_mask_values(datasets_dir, dry_run=False)
        
        # 验证结果
        print("\n=== 验证修复结果 ===")
        verify_mask(datasets_dir / "task_1" / "mask" / "defect")
    else:
        print("操作已取消")
