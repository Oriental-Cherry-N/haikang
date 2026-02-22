#!/usr/bin/env python3
"""
===============================================================================
自适应多任务 PatchCore 训练脚本 (适用于 anomalib 0.7.x)
===============================================================================

【脚本用途】
    对 datasets/ 目录下的任务（task_1 ~ task_10）训练 PatchCore 模型，
    支持灵活选择要训练的任务编号，自动搜索最佳参数组合，
    并实时监控 GPU 显存（24GB 上限），显存不足时自动跳过当前参数并降级重试。
    所有任务的最佳参数与最终分数统一保存至一个 JSON 文件。

【自动调参策略】
    采用分阶段贪心搜索（Staged Greedy Search），逐步锁定最优参数：
        阶段1: 搜索最佳 backbone        (固定其余参数)
        阶段2: 搜索最佳 layers          (固定阶段1结果)
        阶段3: 搜索最佳 coreset_ratio   (固定阶段1-2结果)
        阶段4: 搜索最佳 num_neighbors   (固定阶段1-3结果)
    每阶段候选项按显存消耗从低到高排列，触发 OOM 后自动跳过更重的配置。

【搜索空间】
    backbone:               resnet18 → resnet50 → wide_resnet50_2
    layers:                 [layer2] → [layer3] → [layer2,layer3] → [layer1,layer2,layer3]
    coreset_sampling_ratio: 0.01 → 0.02 → 0.05 → 0.07 → 0.1 → 0.15 → 0.2
    num_neighbors:          3 → 5 → 7 → 9 → 11 → 13 → 15

【显存监控】
    - 后台线程每 60 秒轮询 torch.cuda.mem_get_info()
    - 实际占用超过 23GB（预留 2GB 安全余量）即判定为显存不足
    - 同时捕获 CUDA OOM RuntimeError 作为兜底保护

【输出】
    results/adaptive_search_results.json   — 所有任务的最佳参数与评估指标
    results/task_X/weights/model.ckpt      — 每个任务最终模型权重
    logs/task_X_best/                       — 每个任务最终 TensorBoard 日志

【使用方法】
    # 训练全部任务（默认行为）
    python train.py
    python train.py --tasks all

    # 训练单个任务
    python train.py --tasks 5

    # 训练多个指定任务（逗号分隔）
    python train.py --tasks 1,3,5

    # 训练连续区间（闭区间）
    python train.py --tasks 3-7

    # 混合写法：区间+单个任务
    python train.py --tasks 1-3,5,8-10

===============================================================================
"""

# ==================== 导入依赖库 ====================
import argparse      # 命令行参数解析，用于灵活选择训练任务
import datetime      # 日期时间，用于日志文件名和时间戳
import sys           # 标准流重定向，用于日志双写
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gc            # 垃圾回收，用于训练间隙主动释放 CPU 内存
import json          # JSON 序列化，保存搜索结果
import threading     # 多线程，用于后台显存监控
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from anomalib.models.patchcore import Patchcore
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)


# ==================== 全局配置 ====================
# 以下参数可根据硬件环境和需求进行调整

# 显存限制（字节），24GB 卡预留 2GB 安全余量
# 如果你的显卡不是 24GB，请按 "显存总量 - 2GB" 修改此值
VRAM_LIMIT_GB = 23.0
VRAM_LIMIT_BYTES = int(VRAM_LIMIT_GB * (1024 ** 3))

# 显存监控轮询间隔（秒），值越大监控频率越低，对性能影响越小
VRAM_CHECK_INTERVAL_SEC = 60.0

# 图像统一输入尺寸（高, 宽），PatchCore 要求所有输入尺寸一致
# 较大的尺寸能保留更多细节但消耗更多显存，1024x1024 是工业场景常用值
IMAGE_SIZE = (1024, 1024)

# 数据加载线程数（DataLoader 的 num_workers 参数）
# Windows 下如遇多进程 pickling / BrokenPipe 问题可改为 0
NUM_WORKERS = 8

# ==================== 参数搜索空间（按显存消耗从低到高排列） ====================
# 四阶段贪心搜索会从左到右依次尝试每个候选值，
# 触发 OOM 后跳过该阶段后续更重的候选项（因为它们显存只会更大）。

# backbone：特征提取网络，越大特征越丰富但显存消耗越高
BACKBONES = ["resnet18", "resnet50", "wide_resnet50_2"]

# layers：从 backbone 中抽取哪些层的特征做 PatchCore
# 层数越多、层级越浅（如 layer1），特征维度越大，显存消耗越高
LAYER_CONFIGS = [
    ["layer2"],                          # 单层，最轻量
    ["layer3"],                          # 单层，比 layer2 特征维度更高
    ["layer2", "layer3"],                # 双层，经典搭配
    ["layer1", "layer2", "layer3"],      # 三层，最耗显存
]

# coreset_sampling_ratio：核心集采样比例
# 值越大保留的特征越多（Memory Bank 越大），推理时精度更高但速度更慢
CORESET_RATIOS = [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]

# num_neighbors：KNN 搜索的邻居数
# 对显存影响极小，但值太大可能导致异常检测灵敏度下降
NUM_NEIGHBORS_LIST = [3, 5, 7, 9, 11, 13, 15]


def enable_cuda_acceleration() -> None:
    """
    启用 CUDA 相关加速选项，可显著提升训练速度。

    包含三项优化：
        1. cuDNN benchmark: 让 cuDNN 自动搜索最快的卷积算法
           （因为我们输入尺寸固定，所以 benchmark 模式效果很好）
        2. TF32:  在 Ampere 及更新架构(RTX 30xx/40xx)上，用 TensorFloat-32
           精度替代 FP32 做矩阵乘法，速度快 ~2x 但精度损失可忽略
        3. matmul precision: PyTorch 2.x 的全局矩阵乘法精度设置
    """
    if not torch.cuda.is_available():
        return

    # cuDNN benchmark: 对固定尺寸输入自动选择最快卷积实现
    torch.backends.cudnn.benchmark = True

    # TF32 加速: 仅 Ampere+ GPU 生效（RTX 3090/4090 等）
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    # PyTorch 2.x 矩阵乘法精度: "high" 允许 TF32，"highest" 为纯 FP32
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass  # PyTorch < 2.0 没有此 API，静默跳过


# ==================== 日志双写流 ====================

class TeeStream:
    """
    同时将输出写入终端和日志文件的流包装器。

    用法：
        将 sys.stdout 和 sys.stderr 替换为 TeeStream 实例，
        即可让所有 print() 和异常信息同时输出到终端和日志文件。
    """

    def __init__(self, log_file_path: str, stream):
        self.stream = stream             # 原始流（sys.stdout 或 sys.stderr）
        self.log_file = open(log_file_path, "a", encoding="utf-8")

    def write(self, message: str):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()            # 实时刷新，防止崩溃时丢失日志

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        """关闭日志文件（不关闭原始流）。"""
        self.log_file.close()


def format_duration(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时长字符串。

    示例：
        61.5   -> "1分01秒"
        3661.0 -> "1时01分01秒"
        45.3   -> "45.30秒"
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}时{minutes:02d}分{secs:02d}秒"
    return f"{minutes}分{secs:02d}秒"


class NonOOMTrainingError(RuntimeError):
    """
    自定义异常：标记非 OOM（非显存不足）的训练错误。

    当训练过程中出现 CUDA device-side assert、数据维度不匹配等
    无法通过降低参数配置解决的错误时，抛出此异常以终止当前任务
    （或整个脚本），避免后续训练在损坏的 CUDA 上下文中继续运行。
    """


class FixBatchPatchcore(Patchcore):
    """
    继承 anomalib 的 Patchcore，修复 validation_step / test_step 方法签名兼容问题。

    背景：
        anomalib 0.7.x 的 Patchcore 基类在某些版本中，
        validation_step / test_step 的 batch_idx 参数处理存在不一致，
        当 PyTorch Lightning Trainer 以不同方式调用时可能报错。
        本子类通过重写这两个方法来确保签名兼容。

    同时提供三个静态工具方法用于维度修正，确保
    图像/掩码/标签张量始终包含正确的 batch 维度。
    """

    @staticmethod
    def _ensure_image_batch_dims(tensor: torch.Tensor) -> torch.Tensor:
        """
        确保图像或热力图张量包含 batch 维。

        维度变换规则：
            [H, W]       -> [1, 1, H, W]   (灰度图，无 batch 无 channel)
            [C, H, W]    -> [1, C, H, W]   (单张彩色图，无 batch)
            [B, C, H, W] -> 不变
        """
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _ensure_mask_dims(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        确保掩码维度为 [B, 1, H, W]。

        维度变换规则：
            [H, W]       -> [1, 1, H, W]
            [B, H, W]    -> [B, 1, H, W]   (当 dim0 == batch_size 时)
            [1, H, W]    -> [1, 1, H, W]   (当 dim0 != batch_size 时)
            [B, 1, H, W] -> 不变
        """
        if mask.ndim == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.ndim == 3:
            if mask.shape[0] == batch_size:
                return mask.unsqueeze(1)  # 缺少 channel 维
            return mask.unsqueeze(0)      # 缺少 batch 维
        return mask

    @staticmethod
    def _ensure_label_dims(label: torch.Tensor) -> torch.Tensor:
        """
        确保图像级标签包含 batch 维。

        当 batch_size=1 时，label 可能退化为标量张量 (ndim=0)，
        此方法将其扩展为 [1] 以满足后续计算要求。
        """
        if label.ndim == 0:
            return label.unsqueeze(0)
        return label

    def _fix_and_evaluate(self, batch, batch_idx=None, *args, **kwargs):
        """丢弃 batch_idx 参数后调用基类 validation_step。"""
        del batch_idx  # 基类不需要此参数
        return super().validation_step(batch, *args, **kwargs)

    def validation_step(self, batch, batch_idx=None, *args, **kwargs):
        """重写验证步骤：兼容 Trainer 传入或不传入 batch_idx 的情况。"""
        return self._fix_and_evaluate(batch, batch_idx, *args, **kwargs)

    def test_step(self, batch, batch_idx=None, *args, **kwargs):
        """重写测试步骤：batch_idx 缺省时填充默认值 0。"""
        if batch_idx is None:
            batch_idx = 0
        return super().test_step(batch, batch_idx, *args, **kwargs)


# ==================== 显存监控器 ====================

class VRAMMonitor:
    """
    后台线程实时监控 GPU 显存使用情况。

    工作原理：
        - 每隔 check_interval 秒调用 torch.cuda.mem_get_info() 获取空闲/总显存
        - 计算已用显存 = 总显存 - 空闲显存
        - 若已用显存超过 limit_bytes，则标记 exceeded = True 并停止监控
    """

    def __init__(self, limit_bytes: int, check_interval: float = VRAM_CHECK_INTERVAL_SEC):
        self.limit_bytes = limit_bytes
        self.check_interval = check_interval
        self.exceeded = False          # 是否超过显存限制
        self.max_usage_bytes = 0       # 记录峰值显存占用
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        """监控循环：在后台线程中运行。"""
        while not self._stop_event.is_set():
            if torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info(0)
                    used = total - free
                    self.max_usage_bytes = max(self.max_usage_bytes, used)
                    used_gb = used / (1024 ** 3)
                    total_gb = total / (1024 ** 3)
                    free_gb = free / (1024 ** 3)
                    print("\n" + "=" * 70)
                    print(
                        f"[VRAM监控] 已用: {used_gb:.2f}GB | 空闲: {free_gb:.2f}GB | "
                        f"总计: {total_gb:.2f}GB | 上限: {VRAM_LIMIT_GB:.2f}GB"
                    )
                    print("=" * 70)
                    if used > self.limit_bytes:
                        self.exceeded = True
                        print("\n" + "!" * 70)
                        print(f"[VRAM监控] 显存超限: {used_gb:.2f}GB > {VRAM_LIMIT_GB:.2f}GB")
                        print("!" * 70)
                        break
                except Exception:
                    pass
            self._stop_event.wait(self.check_interval)

    def start(self):
        """启动显存监控。"""
        self.exceeded = False
        self.max_usage_bytes = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止显存监控。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def max_usage_gb(self) -> float:
        """峰值显存占用（GB）。"""
        return self.max_usage_bytes / (1024 ** 3)


# ==================== 工具函数 ====================

def get_train_transform() -> A.Compose:
    """
    构建训练数据增强管线。

    策略："微抖动 + 强质感"，适用于工业印刷品缺陷检测。
    - 禁止翻转、大角度旋转（印刷品有方向性）
    - 几何变换仅模拟流水线机械误差
    - 光照/噪声变换增强成像鲁棒性
    - 首步强制 Resize 到统一尺寸，避免 PatchCore 特征索引越界
    """
    return A.Compose([
        # 第1步: 强制统一尺寸，PatchCore 要求所有输入尺寸一致
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], p=1.0),
        # 平移/缩放/旋转: 模拟流水线上产品的微小位移偏差
        A.ShiftScaleRotate(
            shift_limit=0.05,   # 最大平移 5%
            scale_limit=0.02,   # 最大缩放 2%
            rotate_limit=5,     # 最大旋转 5°（印刷品有方向性，不宜大角度）
            border_mode=0,      # 边界填充用黑色
            p=0.5,              # 50% 概率应用
        ),
        # 透视变换: 模拟相机与产品平面的微小角度偏差
        A.Perspective(scale=(0.02, 0.05), pad_mode=0, p=0.3),
        # 亮度/对比度抠动: 模拟工业光源波动
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        # 高斯噪声: 模拟传感器噪声
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # 运动模糊: 模拟流水线拍摄时的微小拖影
        A.MotionBlur(blur_limit=(3, 5), p=0.2),
        # 高斯模糊: 模拟微小失焦
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # ImageNet 归一化: 与预训练 backbone 的输入分布对齐
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        # 转换为 PyTorch 张量: [H,W,C] -> [C,H,W]
        ToTensorV2(),
    ])


def get_eval_transform() -> A.Compose:
    """
    构建评估/测试时的预处理管线。

    与训练增强不同，评估时仅做确定性变换：
        1. Resize 到统一尺寸
        2. ImageNet Normalize 对齐预训练分布
        3. 转为 PyTorch 张量
    不加任何随机增强，保证评估结果可复现。
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),   # ImageNet 均值
            std=(0.229, 0.224, 0.225),    # ImageNet 标准差
            max_pixel_value=255.0,        # 输入像素范围 [0, 255]
        ),
        ToTensorV2(),
    ])


def validate_task_dataset(dataset_root: Path) -> None:
    """
    对单任务数据集执行基础完整性检查。

    检查项：
        1) defect 与 mask 文件名一一对应
        2) 每个 defect 与同名 mask 尺寸一致
    若检查失败，抛出 ValueError 终止当前任务。
    """
    defect_dir = dataset_root / "defect"
    mask_dir = dataset_root / "mask" / "defect"

    defect_files = sorted([p for p in defect_dir.iterdir() if p.is_file()])
    mask_files = {p.name for p in mask_dir.iterdir() if p.is_file()}

    missing_masks = [p.name for p in defect_files if p.name not in mask_files]
    extra_masks = sorted(list(mask_files - {p.name for p in defect_files}))

    if missing_masks or extra_masks:
        raise ValueError(
            f"{dataset_root.name} 数据集异常：缺失掩码 {len(missing_masks)} 个，"
            f"多余掩码 {len(extra_masks)} 个。"
        )

    size_mismatches = []
    for defect_path in defect_files:
        mask_path = mask_dir / defect_path.name
        with Image.open(defect_path) as defect_img:
            defect_size = defect_img.size
        with Image.open(mask_path) as mask_img:
            mask_size = mask_img.size
        if defect_size != mask_size:
            size_mismatches.append((defect_path.name, defect_size, mask_size))

    if size_mismatches:
        sample = size_mismatches[:3]
        raise ValueError(
            f"{dataset_root.name} 数据集异常：发现 {len(size_mismatches)} 对 defect/mask 尺寸不一致，"
            f"示例: {sample}"
        )


def clear_gpu_memory():
    """
    强制清理 GPU 显存缓存。

    在每次训练尝试前后调用，确保上一轮训练的临时张量、
    模型参数、中间特征图等被及时释放，为下一轮训练腾出显存。

    清理流程：
        1. gc.collect()  — 触发 Python GC，释放无引用对象
        2. torch.cuda.empty_cache()  — 释放 PyTorch 缓存的 CUDA 内存块
        3. torch.cuda.reset_peak_memory_stats()  — 重置峰值统计（便于下轮监控）
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            message = str(e).lower()
            if "device-side assert triggered" in message:
                # CUDA device-side assert 会永久损坏当前进程的 CUDA 上下文，
                # 后续任何 CUDA 操作都会失败，只能重启 Python 进程
                print("[警告] CUDA device-side assert 已触发，当前进程 CUDA 上下文可能已损坏。")
                print("       请先修复数据/标签问题后重启 Python 进程再继续训练。")
            else:
                print(f"[警告] 清理 GPU 显存时发生异常（已忽略）: {e}")


def is_oom_error(error: Exception) -> bool:
    """判断异常是否属于显存不足（OOM）。"""
    message = str(error).lower()
    oom_keywords = [
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cuda out of memory",
    ]
    return any(keyword in message for keyword in oom_keywords)


def is_cuda_device_assert_error(error: Exception) -> bool:
    """判断异常是否属于 CUDA device-side assert。"""
    return "device-side assert triggered" in str(error).lower()


def build_datamodule(dataset_root: Path) -> Folder:
    """
    根据数据集根目录构建 anomalib Folder 数据模块。

    anomalib 的 Folder DataModule 会自动完成以下工作：
        - 将 normal_dir 中的图像标记为正常样本（label=0）
        - 将 abnormal_dir 中的图像标记为异常样本（label=1）
        - 将 mask_dir 中同名文件作为像素级标注
        - 自动将数据拆分为训练集（仅正常图像）和测试集（正常 + 异常）
          PatchCore 只用正常图像训练，异常图像仅在测试时评估

    目录结构要求：
        dataset_root/
            good/        — 正常图像（用于训练 + 测试）
            defect/      — 异常图像（仅用于测试评估）
            mask/defect/ — 异常掩码（像素级标注，与 defect/ 中文件名一一对应）

    Args:
        dataset_root: 单个任务的数据集根目录（如 ./datasets/task_1）

    Returns:
        已调用 setup() 的 Folder 数据模块，可直接传入 Trainer
    """
    datamodule = Folder(
        normal_dir=dataset_root / "good",         # 正常样本目录
        abnormal_dir=dataset_root / "defect",     # 异常样本目录
        mask_dir=dataset_root / "mask" / "defect", # 异常掩码目录
        image_size=IMAGE_SIZE,                     # 统一输入尺寸
        train_batch_size=1,                        # 训练 batch=1（PatchCore 单图提特征）
        eval_batch_size=1,                         # 评估 batch=1（避免显存溢出）
        num_workers=NUM_WORKERS,                   # DataLoader 并行加载线程数
        task=TaskType.SEGMENTATION,                # 分割任务：同时输出图像级+像素级预测
        transform_config_train=get_train_transform(),  # 训练时数据增强
        transform_config_eval=get_eval_transform(),    # 评估时仅做 resize+normalize
    )
    datamodule.setup()  # 执行数据拆分、路径索引等初始化
    return datamodule


def compute_combined_score(results: dict) -> float:
    """
    计算综合得分 = 0.5 * image_AUROC + 0.5 * pixel_AUROC。
    同时兼顾图像级分类能力和像素级定位精度。
    """
    img_auroc = results.get("image_AUROC", 0.0)
    pix_auroc = results.get("pixel_AUROC", 0.0)
    return img_auroc * 0.5 + pix_auroc * 0.5


# ==================== 单次训练与评估 ====================

def train_and_evaluate(
    dataset_root: Path,
    params: dict,
    vram_monitor: VRAMMonitor,
) -> Tuple[Optional[dict], bool]:
    """
    使用指定参数训练 PatchCore 并在测试集上评估。

    此函数是参数搜索的核心单元：每次调用用一组候选参数完成
    “建数据 → 建模型 → 训练 → 测试”的完整流程，并返回测试结果。

    异常处理策略：
        - OOM（显存不足）: 返回 (None, True)，让调用方跳过更重的配置
        - CUDA device-side assert: 抛出 NonOOMTrainingError 终止脚本
        - 其他异常: 抛出 NonOOMTrainingError 终止脚本

    Args:
        dataset_root:  数据集根目录（如 ./datasets/task_1）
        params:        参数字典 {backbone, layers, coreset_sampling_ratio, num_neighbors}
        vram_monitor:  显存监控器实例

    Returns:
        (测试结果字典, 是否为 OOM/显存超限)
        成功时返回 (results_dict, False)
        OOM/显存超限时返回 (None, True)
    """
    param_str = (
        f"backbone={params['backbone']}, layers={params['layers']}, "
        f"coreset={params['coreset_sampling_ratio']}, neighbors={params['num_neighbors']}"
    )
    print(f"    尝试: {param_str}")

    clear_gpu_memory()

    try:
        # ---- 数据: 构建 DataModule，自动完成训练/测试集拆分 ----
        datamodule = build_datamodule(dataset_root)

        # ---- 模型: 创建 PatchCore 实例 ----
        # input_size:               与数据的 IMAGE_SIZE 保持一致
        # backbone:                 CNN 特征提取网络
        # layers:                   从 backbone 中抽取特征的层名列表
        # pre_trained=True:         使用 ImageNet 预训练权重（小样本场景必须）
        # coreset_sampling_ratio:   Memory Bank 采样比例
        # num_neighbors:            KNN 推理时的 K 值
        model = FixBatchPatchcore(
            input_size=IMAGE_SIZE,
            backbone=params["backbone"],
            layers=params["layers"],
            pre_trained=True,
            coreset_sampling_ratio=params["coreset_sampling_ratio"],
            num_neighbors=params["num_neighbors"],
        )

        # ---- 回调: 搜索阶段精简版（无模型保存，仅计算指标） ----
        # MetricsConfigurationCallback: 配置要计算的评估指标
        #   - image_metrics: 图像级 AUROC 和 F1Score
        #   - pixel_metrics: 像素级 AUROC 和 F1Score
        # PostProcessingConfigurationCallback: 配置异常分数后处理
        #   - MIN_MAX 归一化: 将异常分数映射到 [0,1]
        #   - ADAPTIVE 阈值: 自动确定正常/异常的分界阈值
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
        ]

        # ---- 训练器: 搜索阶段禁用日志和进度条以加速搜索 ----
        # max_epochs=1:             PatchCore 只需一个 epoch（无反向传播训练）
        # logger=False:             搜索阶段不记录日志（节省 I/O）
        # num_sanity_val_steps=0:   跳过训练前的校验步骤（加速）
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            precision="32",
            default_root_dir="results/_search_tmp",
            logger=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # ---- 启动显存监控 & 训练 ----
        # PatchCore 的 fit() 流程：
        #   1. 用正常图像遍历训练集，提取 backbone 特征
        #   2. 将所有特征存入 Memory Bank
        #   3. 对 Memory Bank 做 Coreset 采样（压缩大小）
        #   4. 建立 KNN 索引用于推理时快速搜索
        vram_monitor.start()
        trainer.fit(model=model, datamodule=datamodule)

        # 检查训练期间显存监控器是否检测到超限
        if vram_monitor.exceeded:
            vram_monitor.stop()
            print(f"      !! 显存超限 ({vram_monitor.max_usage_gb:.1f}GB > {VRAM_LIMIT_GB}GB)，跳过")
            clear_gpu_memory()
            return None, True

        # ---- 测试评估 ----
        test_results = trainer.test(model=model, datamodule=datamodule, verbose=False)
        vram_monitor.stop()

        if not test_results:
            return None, False

        # ---- 输出评测结果（按字母序排列，跳过内部字段） ----
        results = test_results[0]
        vram_gb = vram_monitor.max_usage_gb
        score = compute_combined_score(results)
        print(f"      -> 本次训练评测指标（显存峰值 {vram_gb:.1f}GB）")
        for key in sorted(results.keys()):
            value = results[key]
            if isinstance(value, float) and not key.startswith("_"):
                print(f"         {key}: {value:.4f}")
        print(f"         combined_score: {score:.4f}")
        results["_vram_usage_gb"] = round(vram_gb, 2)  # 记录显存峰值（带下划线前缀表示内部字段）
        return results, False

    except RuntimeError as e:
        # ---- 运行时异常分类处理 ----
        vram_monitor.stop()
        if is_oom_error(e):
            print(f"      !! CUDA 显存不足 (OOM)，跳过: {e}")
            clear_gpu_memory()
            return None, True
        elif is_cuda_device_assert_error(e):
            print("      !! CUDA device-side assert：通常由标签/掩码取值或维度异常引发。")
            print("      !! 该错误会破坏当前进程 CUDA 上下文，需修复数据后重启进程。")
            raise NonOOMTrainingError(
                f"任务 {dataset_root.name} 参数 {param_str} 触发 CUDA device-side assert。"
                "请检查 mask/label 是否为二值（0/1），以及张量维度是否匹配。"
            ) from e
        else:
            print(f"      !! 非 OOM 运行时错误（将终止脚本）: {e}")
        clear_gpu_memory()
        raise NonOOMTrainingError(
            f"任务 {dataset_root.name} 参数 {param_str} 出现非 OOM RuntimeError: {e}"
        ) from e

    except Exception as e:
        vram_monitor.stop()
        print(f"      !! 非 OOM 异常（将终止脚本）: {e}")
        clear_gpu_memory()
        raise NonOOMTrainingError(
            f"任务 {dataset_root.name} 参数 {param_str} 出现非 OOM 异常: {e}"
        ) from e


# ==================== 分阶段参数搜索 ====================

def staged_search(
    task_name: str,
    dataset_root: Path,
    vram_monitor: VRAMMonitor,
) -> Tuple[dict, Optional[dict], float]:
    """
    对单个任务执行四阶段贪心参数搜索。

    搜索策略说明：
        “贪心”意味着每个阶段只搜索一个维度，其余维度固定；
        不同于网格搜索的 O(n^4) 组合爆炸，此方法复杂度为 O(n1+n2+n3+n4)，
        大幅减少尝试次数，适合显存紧张、任务多的场景。

        候选项按显存消耗升序排列，一旦触发 OOM/显存超限，
        该阶段后续更重的候选项会被直接跳过（因为它们显存只会更大）。

    Args:
        task_name:     任务名称（如 "task_1"），仅用于日志输出
        dataset_root:  数据集根目录
        vram_monitor:  显存监控器实例

    Returns:
        (best_params, best_results, best_score)
        - best_params:  搜索到的最优参数字典
        - best_results: 最优参数对应的测试结果，全部失败时为 None
        - best_score:   最优综合得分，全部失败时为 -1.0
    """
    # 初始默认参数（最轻量配置，确保能在绝大多数 GPU 上运行）
    best_params = {
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.05,
        "num_neighbors": 9,
    }
    best_score = -1.0
    best_results = None

    # ======================== 阶段 1: backbone ========================
    print(f"\n  [阶段 1/4] 搜索最佳 backbone ...")
    oom_hit = False
    for backbone in BACKBONES:
        if oom_hit:
            print(f"    跳过 backbone={backbone}（更重配置，前序已 OOM）")
            continue
        params = {**best_params, "backbone": backbone}
        results, is_oom = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
            if is_oom:
                oom_hit = True  # 后续更重的 backbone 也会 OOM，直接跳过
            continue
        score = compute_combined_score(results)
        if score > best_score:
            best_score = score
            best_params["backbone"] = backbone
            best_results = results
    print(f"  => 最佳 backbone: {best_params['backbone']}  (综合={best_score:.4f})")

    # ======================== 阶段 2: layers ========================
    print(f"\n  [阶段 2/4] 搜索最佳 layers ...")
    oom_hit = False
    for layers in LAYER_CONFIGS:
        if oom_hit:
            print(f"    跳过 layers={layers}（更重配置，前序已 OOM）")
            continue
        params = {**best_params, "layers": layers}
        results, is_oom = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
            if is_oom:
                oom_hit = True
            continue
        score = compute_combined_score(results)
        if score > best_score:
            best_score = score
            best_params["layers"] = layers
            best_results = results
    print(f"  => 最佳 layers: {best_params['layers']}  (综合={best_score:.4f})")

    # ======================== 阶段 3: coreset_sampling_ratio ========================
    print(f"\n  [阶段 3/4] 搜索最佳 coreset_sampling_ratio ...")
    oom_hit = False
    for ratio in CORESET_RATIOS:
        if oom_hit:
            print(f"    跳过 coreset={ratio}（更重配置，前序已 OOM）")
            continue
        params = {**best_params, "coreset_sampling_ratio": ratio}
        results, is_oom = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
            if is_oom:
                oom_hit = True
            continue
        score = compute_combined_score(results)
        if score > best_score:
            best_score = score
            best_params["coreset_sampling_ratio"] = ratio
            best_results = results
    print(f"  => 最佳 coreset_sampling_ratio: {best_params['coreset_sampling_ratio']}  (综合={best_score:.4f})")

    # ======================== 阶段 4: num_neighbors ========================
    print(f"\n  [阶段 4/4] 搜索最佳 num_neighbors ...")
    for neighbors in NUM_NEIGHBORS_LIST:
        params = {**best_params, "num_neighbors": neighbors}
        results, _ = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
            continue  # num_neighbors 对显存影响极小，不触发 OOM 跳过
        score = compute_combined_score(results)
        if score > best_score:
            best_score = score
            best_params["num_neighbors"] = neighbors
            best_results = results
    print(f"  => 最佳 num_neighbors: {best_params['num_neighbors']}  (综合={best_score:.4f})")

    return best_params, best_results, best_score


# ==================== 使用最佳参数训练最终模型 ====================

def final_train(
    task_name: str,
    dataset_root: Path,
    params: dict,
    output_dir: Path,
) -> Optional[dict]:
    """
    使用搜索到的最佳参数训练最终模型，并保存权重和 TensorBoard 日志。

    与搜索阶段的 train_and_evaluate() 不同，此函数：
        - 启用 TensorBoardLogger 记录训练曲线
        - 启用 ModelCheckpoint 保存 pixel_AUROC 最高的权重
        - 启用进度条和模型摘要
        - 不使用显存监控（因为参数已经验证过不会 OOM）

    Args:
        task_name:    任务名称（如 "task_1"），用于命名日志目录
        dataset_root: 数据集根目录
        params:       最佳参数字典
        output_dir:   模型权重和日志的输出目录

    Returns:
        最终测试结果字典，失败时返回 None
    """
    print(f"\n  使用最佳参数训练最终模型并保存 ...")
    clear_gpu_memory()

    try:
        datamodule = build_datamodule(dataset_root)

        model = FixBatchPatchcore(
            input_size=IMAGE_SIZE,
            backbone=params["backbone"],
            layers=params["layers"],
            pre_trained=True,
            coreset_sampling_ratio=params["coreset_sampling_ratio"],
            num_neighbors=params["num_neighbors"],
        )

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
                dirpath=str(output_dir / "weights"),
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
            precision="32",
            default_root_dir=str(output_dir),
            logger=TensorBoardLogger(save_dir="logs/", name=f"{task_name}_best"),
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )

        trainer.fit(model=model, datamodule=datamodule)
        test_results = trainer.test(model=model, datamodule=datamodule)

        if test_results:
            print(f"    最终模型测试结果:")
            for key, value in sorted(test_results[0].items()):
                if isinstance(value, float) and not key.startswith("_"):
                    print(f"      {key}: {value:.4f}")
            print(f"    模型权重已保存至: {output_dir / 'weights'}")
            return test_results[0]

        return None

    except Exception as e:
        print(f"    最终训练失败: {e}")
        clear_gpu_memory()
        return None


# ==================== 主函数 ====================

# ==================== 命令行参数解析 ====================

def parse_task_selection(task_str: str, available_ids: List[int]) -> List[int]:
    """
    将用户输入的任务选择字符串解析为排好序的任务编号列表。

    支持的格式：
        "all"       — 全部可用任务
        "5"         — 单个任务
        "1,3,5"     — 逗号分隔的多个任务
        "3-7"       — 连续区间（闭区间，即 3,4,5,6,7）
        "1-3,5,8-10" — 混合写法，区间与单个任务自由组合

    Args:
        task_str:      用户从命令行传入的原始字符串
        available_ids: datasets/ 目录下实际存在的任务编号列表

    Returns:
        去重并排序后的任务编号列表

    Raises:
        ValueError: 格式不合法或编号超出可用范围时抛出
    """
    # "all" 返回全部可用编号
    if task_str.strip().lower() == "all":
        return sorted(available_ids)

    selected = set()  # 用 set 自动去重

    # 按逗号拆分多个片段
    for part in task_str.split(","):
        part = part.strip()
        if not part:
            continue  # 忽略空片段（如尾部多余逗号 "1,2,"）

        if "-" in part:
            # ---------- 区间格式："3-7" ----------
            bounds = part.split("-", maxsplit=1)  # 最多拆分一次，避免负号干扰
            if len(bounds) != 2:
                raise ValueError(f"区间格式错误: '{part}'，正确示例: '3-7'")
            try:
                start, end = int(bounds[0].strip()), int(bounds[1].strip())
            except ValueError:
                raise ValueError(f"区间包含非数字: '{part}'")
            if start > end:
                raise ValueError(f"区间起点 {start} 大于终点 {end}: '{part}'")
            for i in range(start, end + 1):
                selected.add(i)
        else:
            # ---------- 单个编号："5" ----------
            try:
                selected.add(int(part))
            except ValueError:
                raise ValueError(f"无法识别的任务编号: '{part}'")

    # 检查选中的编号是否都在可用范围内
    invalid = sorted(selected - set(available_ids))
    if invalid:
        raise ValueError(
            f"以下任务编号在 datasets/ 中不存在: {invalid}\n"
            f"可用编号: {sorted(available_ids)}"
        )

    return sorted(selected)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        包含 tasks 属性的 Namespace 对象
    """
    parser = argparse.ArgumentParser(
        description="自适应多任务 PatchCore 训练脚本 — 支持灵活选择训练任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train.py                    # 训练全部任务（默认）
  python train.py --tasks all        # 同上
  python train.py --tasks 5          # 仅训练 task_5
  python train.py --tasks 1,3,5      # 训练 task_1, task_3, task_5
  python train.py --tasks 3-7        # 训练 task_3 到 task_7
  python train.py --tasks 1-3,5,8-10 # 混合写法
        """,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            '要训练的任务编号。默认 "all" 训练全部。'
            '支持: 单个(5), 逗号分隔(1,3,5), 区间(3-7), 混合(1-3,5,8-10)'
        ),
    )
    return parser.parse_args()


# ==================== 主函数 ====================

def main():
    """
    主函数：根据命令行参数选择任务，对每个任务执行自适应参数搜索 + 最终训练。

    执行流程：
        1. 解析命令行参数，确定要训练的任务编号列表
        2. 扫描 datasets/ 下的所有 task_* 目录，按编号筛选
        3. 对每个选中的任务执行四阶段贪心参数搜索
        4. 用最佳参数重新训练并保存模型权重
        5. 将所有结果汇总到 adaptive_search_results.json
           (增量写入：已有的旧任务结果不会被覆盖)
    """
    # ---- 记录脚本总计时起点 ----
    script_start_time = time.time()

    # ---- 解析命令行参数 ----
    args = parse_args()

    datasets_root = Path("./datasets")
    output_root = Path("results")
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = output_root / "adaptive_search_results.json"

    # ---- 设置日志双写：所有 print 输出同步写入日志文件 ----
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = logs_dir / f"train_log_{log_timestamp}.txt"
    tee_stdout = TeeStream(str(log_file_path), sys.stdout)
    tee_stderr = TeeStream(str(log_file_path), sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    print(f"运行日志将保存至: {log_file_path.resolve()}")
    print(f"脚本启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 启用 cuDNN / TF32 / AMP 相关加速
    enable_cuda_acceleration()

    # 初始化后台显存监控器（整个脚本生命周期复用同一实例）
    vram_monitor = VRAMMonitor(VRAM_LIMIT_BYTES)

    # ---- 扫描 datasets/ 获取全部可用任务编号 ----
    all_task_dirs = sorted(
        [d for d in datasets_root.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not all_task_dirs:
        print("错误：datasets/ 目录下未找到 task_* 数据集。")
        return

    # 提取所有可用的任务编号（如 [1, 2, ..., 10]）
    available_ids = [int(d.name.split("_")[1]) for d in all_task_dirs]

    # ---- 根据 --tasks 参数筛选 ----
    try:
        selected_ids = parse_task_selection(args.tasks, available_ids)
    except ValueError as e:
        print(f"错误：任务选择参数无效 —— {e}")
        return

    # 只保留用户选中的任务目录
    task_dirs = [
        d for d in all_task_dirs
        if int(d.name.split("_")[1]) in selected_ids
    ]

    if not task_dirs:
        print(f"错误：未匹配到任何任务。可用任务编号: {available_ids}")
        return

    # ---- 打印运行环境信息 ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.mem_get_info(0)[1] / (1024 ** 3)
        print(f"GPU: {gpu_name}  总显存: {total_gb:.1f}GB  监控上限: {VRAM_LIMIT_GB}GB")
    else:
        print("警告：未检测到 GPU，将使用 CPU（速度极慢）")

    selected_names = [d.name for d in task_dirs]
    print(f"选中 {len(task_dirs)} 个任务: {', '.join(selected_names)}")
    print(f"图像尺寸: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"搜索空间: {len(BACKBONES)} backbones x {len(LAYER_CONFIGS)} layers x "
          f"{len(CORESET_RATIOS)} coreset x {len(NUM_NEIGHBORS_LIST)} neighbors")
    print(f"搜索策略: 四阶段贪心搜索（每任务最多 "
          f"{len(BACKBONES) + len(LAYER_CONFIGS) + len(CORESET_RATIOS) + len(NUM_NEIGHBORS_LIST)} 次训练）")
    print("=" * 70)

    # ---- 加载已有结果（增量模式：不覆盖之前训练好的任务） ----
    all_results = {}
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"已加载历史结果: {results_file}（含 {len(all_results)} 个任务）")
        except (json.JSONDecodeError, IOError):
            print(f"警告：无法解析 {results_file}，将创建新文件")
            all_results = {}

    # ---- 各任务计时记录 ----
    task_timings = {}  # {task_name: duration_seconds}

    try:
        for task_idx, task_dir in enumerate(task_dirs, 1):
            task_name = task_dir.name
            task_start_time = time.time()
            print(f"\n{'=' * 70}")
            print(f"[{task_idx}/{len(task_dirs)}] {task_name}")
            print(f"{'=' * 70}")

            validate_task_dataset(task_dir)

            task_output_dir = output_root / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)

            # ---------- 阶段搜索 ----------
            best_params, best_results, best_score = staged_search(
                task_name, task_dir, vram_monitor,
            )

            # ---------- 最终训练 ----------
            final_results = None
            if best_results is not None:
                final_results = final_train(task_name, task_dir, best_params, task_output_dir)

            # ---------- 记录任务耗时 ----------
            task_elapsed = time.time() - task_start_time
            task_timings[task_name] = task_elapsed

            # ---------- 汇总结果 ----------
            final_metrics = {}
            if final_results is not None:
                final_metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in final_results.items()
                    if not k.startswith("_")
                }
            elif best_results is not None:
                final_metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in best_results.items()
                    if not k.startswith("_")
                }

            all_results[task_name] = {
                "best_params": {
                    "backbone": best_params["backbone"],
                    "layers": best_params["layers"],
                    "coreset_sampling_ratio": best_params["coreset_sampling_ratio"],
                    "num_neighbors": best_params["num_neighbors"],
                },
                "combined_score": round(best_score, 4),
                "metrics": final_metrics,
                "training_time": format_duration(task_elapsed),
            }

            # 打印本任务总结
            print(f"\n  {'*' * 50}")
            print(f"  {task_name} 最佳参数:")
            print(f"    backbone:               {best_params['backbone']}")
            print(f"    layers:                 {best_params['layers']}")
            print(f"    coreset_sampling_ratio: {best_params['coreset_sampling_ratio']}")
            print(f"    num_neighbors:          {best_params['num_neighbors']}")
            print(f"    综合得分:               {best_score:.4f}")
            if final_metrics:
                print(f"    image_AUROC:            {final_metrics.get('image_AUROC', 'N/A')}")
                print(f"    pixel_AUROC:            {final_metrics.get('pixel_AUROC', 'N/A')}")
                print(f"    image_F1Score:          {final_metrics.get('image_F1Score', 'N/A')}")
                print(f"    pixel_F1Score:          {final_metrics.get('pixel_F1Score', 'N/A')}")
            print(f"    训练耗时:               {format_duration(task_elapsed)}")
            print(f"  {'*' * 50}")

            # 每完成一个任务就保存中间结果（防止中途断电丢失）
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  (中间结果已保存至 {results_file})")

            clear_gpu_memory()
    except NonOOMTrainingError as e:
        print("\n" + "#" * 70)
        print("[致命错误] 发生非 OOM 异常，已按要求终止脚本")
        print(f"错误详情: {e}")
        print("#" * 70)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        raise SystemExit(1) from e

    # ==================== 最终汇总 ====================
    print(f"\n\n{'=' * 70}")
    print("全部任务训练完成！汇总结果")
    print(f"{'=' * 70}")

    # 表格输出
    header = f"{'任务':<10} {'backbone':<20} {'layers':<30} {'coreset':>8} {'neighbors':>10} {'综合':>6} {'img_AUROC':>10} {'pix_AUROC':>10}"
    print(header)
    print("-" * len(header))

    for task_name, result in all_results.items():
        p = result["best_params"]
        m = result.get("metrics", {})
        layers_str = ",".join(p["layers"])
        print(
            f"{task_name:<10} {p['backbone']:<20} {layers_str:<30} "
            f"{p['coreset_sampling_ratio']:>8.2f} {p['num_neighbors']:>10d} "
            f"{result['combined_score']:>6.4f} "
            f"{m.get('image_AUROC', 'N/A'):>10} "
            f"{m.get('pixel_AUROC', 'N/A'):>10}"
        )

    # ==================== 时间统计 ====================
    script_elapsed = time.time() - script_start_time
    print(f"\n{'=' * 70}")
    print("运行时间统计")
    print(f"{'=' * 70}")
    for tname, tdur in task_timings.items():
        print(f"  {tname:<12} {format_duration(tdur)}")
    print(f"  {'─' * 30}")
    print(f"  {'总计':<12} {format_duration(script_elapsed)}")
    print(f"脚本结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n结果文件: {results_file.resolve()}")
    print(f"运行日志: {log_file_path.resolve()}")
    print(f"使用 'tensorboard --logdir=logs/' 查看训练曲线")
    print("完成！")

    # ---- 恢复标准流并关闭日志文件 ----
    sys.stdout = tee_stdout.stream
    sys.stderr = tee_stderr.stream
    tee_stdout.close()
    tee_stderr.close()


if __name__ == "__main__":
    main()

