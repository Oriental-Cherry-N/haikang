#!/usr/bin/env python3
"""
数据集整理脚本
将 original_dataset 中的数据整理成 anomalib Folder 模块需要的格式

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

目标结构：
datasets/my_dataset/
├── good/               # 所有正常图像
├── defect/             # 所有异常图像
└── mask/
    └── defect/         # 所有掩码图像（文件名与 defect/ 中一一对应）
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def organize_dataset(
    source_dir: Path,
    target_dir: Path,
    copy_files: bool = True,  # True: 复制文件, False: 移动文件
):
    """
    整理数据集
    
    Args:
        source_dir: 原始数据集路径 (original_dataset)
        target_dir: 目标数据集路径 (datasets/my_dataset)
        copy_files: 是否复制文件（False则移动文件）
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
        "skipped": 0,
    }
    
    # 用于处理文件名冲突（不同子文件夹可能有相同文件名）
    existing_files = {"good": set(), "defect": set(), "mask": set()}
    
    # 获取所有子文件夹（1, 2, 3, ... 10）
    subfolders = sorted(
        [d for d in source_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0
    )
    
    print(f"找到 {len(subfolders)} 个子文件夹: {[d.name for d in subfolders]}")
    print(f"目标目录: {target_dir}")
    print(f"操作模式: {'复制' if copy_files else '移动'}")
    print("-" * 50)
    
    file_op = shutil.copy2 if copy_files else shutil.move
    
    for subfolder in tqdm(subfolders, desc="处理子文件夹"):
        folder_name = subfolder.name
        
        # 查找 ok 和 ng 文件夹
        ok_folder = None
        ng_folder = None
        
        for item in subfolder.iterdir():
            if item.is_dir():
                if item.name.endswith("-ok") or item.name.endswith("_ok"):
                    ok_folder = item
                elif item.name.endswith("_ng") or item.name.endswith("-ng"):
                    ng_folder = item
        
        # 处理正常图像 (ok 文件夹)
        if ok_folder and ok_folder.exists():
            for img_file in ok_folder.iterdir():
                if img_file.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg"]:
                    # 添加前缀避免文件名冲突
                    new_name = f"{folder_name}_{img_file.name}"
                    if new_name in existing_files["good"]:
                        # 如果仍有冲突，添加序号
                        base = img_file.stem
                        ext = img_file.suffix
                        counter = 1
                        while new_name in existing_files["good"]:
                            new_name = f"{folder_name}_{base}_{counter}{ext}"
                            counter += 1
                    
                    existing_files["good"].add(new_name)
                    target_path = good_dir / new_name
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
                    new_name = f"{folder_name}_{original_stem}{img_file.suffix}"
                    
                    if new_name in existing_files["mask"]:
                        base = original_stem
                        ext = img_file.suffix
                        counter = 1
                        while new_name in existing_files["mask"]:
                            new_name = f"{folder_name}_{base}_{counter}{ext}"
                            counter += 1
                    
                    existing_files["mask"].add(new_name)
                    target_path = mask_dir / new_name
                    file_op(str(img_file), str(target_path))
                    stats["mask"] += 1
                else:
                    # 这是异常图像
                    new_name = f"{folder_name}_{img_file.name}"
                    
                    if new_name in existing_files["defect"]:
                        base = img_file.stem
                        ext = img_file.suffix
                        counter = 1
                        while new_name in existing_files["defect"]:
                            new_name = f"{folder_name}_{base}_{counter}{ext}"
                            counter += 1
                    
                    existing_files["defect"].add(new_name)
                    target_path = defect_dir / new_name
                    file_op(str(img_file), str(target_path))
                    stats["defect"] += 1
    
    # 打印统计信息
    print("-" * 50)
    print("整理完成！统计信息：")
    print(f"  正常图像 (good):     {stats['good']} 张")
    print(f"  异常图像 (defect):   {stats['defect']} 张")
    print(f"  掩码图像 (mask):     {stats['mask']} 张")
    
    # 验证异常图像和掩码数量是否匹配
    if stats['defect'] != stats['mask']:
        print(f"\n⚠️ 警告：异常图像数量 ({stats['defect']}) 与掩码数量 ({stats['mask']}) 不匹配！")
        print("   请检查原始数据集中是否有缺失的掩码文件。")
    else:
        print(f"\n✅ 异常图像与掩码数量匹配！")
    
    print(f"\n数据集已保存到: {target_dir}")
    
    return stats


def verify_dataset(target_dir: Path):
    """
    验证整理后的数据集
    """
    print("\n" + "=" * 50)
    print("验证数据集...")
    
    good_dir = target_dir / "good"
    defect_dir = target_dir / "defect"
    mask_dir = target_dir / "mask" / "defect"
    
    good_files = set(f.name for f in good_dir.iterdir() if f.is_file())
    defect_files = set(f.name for f in defect_dir.iterdir() if f.is_file())
    mask_files = set(f.name for f in mask_dir.iterdir() if f.is_file())
    
    print(f"正常图像数量: {len(good_files)}")
    print(f"异常图像数量: {len(defect_files)}")
    print(f"掩码图像数量: {len(mask_files)}")
    
    # 检查异常图像和掩码是否一一对应
    missing_masks = defect_files - mask_files
    extra_masks = mask_files - defect_files
    
    if missing_masks:
        print(f"\n⚠️ 以下异常图像缺少对应掩码 ({len(missing_masks)} 个):")
        for f in list(missing_masks)[:5]:
            print(f"   - {f}")
        if len(missing_masks) > 5:
            print(f"   ... 还有 {len(missing_masks) - 5} 个")
    
    if extra_masks:
        print(f"\n⚠️ 以下掩码没有对应的异常图像 ({len(extra_masks)} 个):")
        for f in list(extra_masks)[:5]:
            print(f"   - {f}")
        if len(extra_masks) > 5:
            print(f"   ... 还有 {len(extra_masks) - 5} 个")
    
    if not missing_masks and not extra_masks:
        print("\n✅ 所有异常图像都有对应的掩码！")
    
    print("=" * 50)


if __name__ == "__main__":
    # 配置路径
    source_dir = Path("./original_dataset")  # 原始数据集路径
    target_dir = Path("./datasets/my_dataset")  # 目标数据集路径
    
    # 如果目标目录已存在，询问是否覆盖
    if target_dir.exists():
        response = input(f"目标目录 {target_dir} 已存在，是否删除并重新整理？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(target_dir)
            print(f"已删除 {target_dir}")
        else:
            print("操作已取消")
            exit(0)
    
    # 执行整理
    organize_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        copy_files=True,  # 设为 False 可以移动文件而非复制（节省空间）
    )
    
    # 验证结果
    verify_dataset(target_dir)
