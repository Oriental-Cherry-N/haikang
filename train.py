#!/usr/bin/env python3
"""
===============================================================================
自适应多任务 PatchCore 训练脚本 (适用于 anomalib 0.7.x)
===============================================================================

【脚本用途】
    对 datasets/ 目录下的 10 个任务（task_1 ~ task_10）逐一训练 PatchCore 模型，
    自动搜索最佳参数组合，并实时监控 GPU 显存（24GB 上限），显存不足时自动跳过
    当前参数并降级重试。所有任务的最佳参数与最终分数统一保存至一个 JSON 文件。

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
    coreset_sampling_ratio: 0.01 → 0.05 → 0.1 → 0.25
    num_neighbors:          3 → 5 → 9 → 15

【显存监控】
    - 后台线程每 0.5 秒轮询 torch.cuda.mem_get_info()
    - 实际占用超过 22GB（预留 2GB 安全余量）即判定为显存不足
    - 同时捕获 CUDA OOM RuntimeError 作为兜底保护

【输出】
    results/adaptive_search_results.json   — 所有任务的最佳参数与评估指标
    results/task_X/weights/model.ckpt      — 每个任务最终模型权重
    logs/task_X_best/                       — 每个任务最终 TensorBoard 日志

【使用方法】
    python train.py

===============================================================================
"""

# ==================== 导入依赖库 ====================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import json
import threading
import time
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A

from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from anomalib.models.patchcore import Patchcore
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)


# ==================== 全局配置 ====================

# 显存限制（字节），24GB 卡预留 2GB 安全余量
VRAM_LIMIT_GB = 22.0
VRAM_LIMIT_BYTES = int(VRAM_LIMIT_GB * (1024 ** 3))

# 图像尺寸（高, 宽），保持与原脚本一致
IMAGE_SIZE = (1024, 1024)

# 数据加载线程数，Windows 下如遇多进程问题可改为 0
NUM_WORKERS = 8

# ==================== 参数搜索空间（按显存消耗从低到高排列） ====================

BACKBONES = ["resnet18", "resnet50", "wide_resnet50_2"]

LAYER_CONFIGS = [
    ["layer2"],                          # 单层，最轻量
    ["layer3"],                          # 单层，比 layer2 特征维度更高
    ["layer2", "layer3"],                # 双层，经典搭配
    ["layer1", "layer2", "layer3"],      # 三层，最耗显存
]

CORESET_RATIOS = [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]

NUM_NEIGHBORS_LIST = [3, 5, 7, 9, 11, 13, 15]


# ==================== 显存监控器 ====================

class VRAMMonitor:
    """
    后台线程实时监控 GPU 显存使用情况。

    工作原理：
        - 每隔 check_interval 秒调用 torch.cuda.mem_get_info() 获取空闲/总显存
        - 计算已用显存 = 总显存 - 空闲显存
        - 若已用显存超过 limit_bytes，则标记 exceeded = True 并停止监控
    """

    def __init__(self, limit_bytes: int, check_interval: float = 0.5):
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
                    if used > self.limit_bytes:
                        self.exceeded = True
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
    """
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.02, rotate_limit=5,
            border_mode=0, p=0.5,
        ),
        A.Perspective(scale=(0.02, 0.05), pad_mode=0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=(3, 5), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def clear_gpu_memory():
    """强制清理 GPU 显存缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_datamodule(dataset_root: Path) -> Folder:
    """
    根据数据集根目录构建 Folder 数据模块。

    目录结构要求：
        dataset_root/
            good/       — 正常图像
            defect/     — 异常图像
            mask/defect/ — 异常掩码
    """
    datamodule = Folder(
        normal_dir=dataset_root / "good",
        abnormal_dir=dataset_root / "defect",
        mask_dir=dataset_root / "mask" / "defect",
        image_size=IMAGE_SIZE,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=NUM_WORKERS,
        task=TaskType.SEGMENTATION,
        transform_config_train=get_train_transform(),
    )
    datamodule.setup()
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
) -> dict | None:
    """
    使用指定参数训练 PatchCore 并在测试集上评估。

    Args:
        dataset_root:  数据集根目录（如 ./datasets/task_1）
        params:        参数字典 {backbone, layers, coreset_sampling_ratio, num_neighbors}
        vram_monitor:  显存监控器实例

    Returns:
        测试结果字典（含 image_AUROC, pixel_AUROC 等），失败返回 None。
    """
    param_str = (
        f"backbone={params['backbone']}, layers={params['layers']}, "
        f"coreset={params['coreset_sampling_ratio']}, neighbors={params['num_neighbors']}"
    )
    print(f"    尝试: {param_str}")

    clear_gpu_memory()

    try:
        # ---- 数据 ----
        datamodule = build_datamodule(dataset_root)

        # ---- 模型 ----
        model = Patchcore(
            input_size=IMAGE_SIZE,
            backbone=params["backbone"],
            layers=params["layers"],
            pre_trained=True,
            coreset_sampling_ratio=params["coreset_sampling_ratio"],
            num_neighbors=params["num_neighbors"],
        )

        # ---- 回调（搜索阶段精简版） ----
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

        # ---- 训练器（禁用日志和进度条以加速搜索） ----
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            default_root_dir="results/_search_tmp",
            logger=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # ---- 启动显存监控 & 训练 ----
        vram_monitor.start()
        trainer.fit(model=model, datamodule=datamodule)

        # 检查训练期间是否超过显存限制
        if vram_monitor.exceeded:
            vram_monitor.stop()
            print(f"      !! 显存超限 ({vram_monitor.max_usage_gb:.1f}GB > {VRAM_LIMIT_GB}GB)，跳过")
            clear_gpu_memory()
            return None

        # ---- 测试评估 ----
        test_results = trainer.test(model=model, datamodule=datamodule, verbose=False)
        vram_monitor.stop()

        if not test_results:
            return None

        results = test_results[0]
        vram_gb = vram_monitor.max_usage_gb
        score = compute_combined_score(results)
        print(
            f"      -> image_AUROC={results.get('image_AUROC', 0):.4f}  "
            f"pixel_AUROC={results.get('pixel_AUROC', 0):.4f}  "
            f"综合={score:.4f}  显存={vram_gb:.1f}GB"
        )
        results["_vram_usage_gb"] = round(vram_gb, 2)
        return results

    except RuntimeError as e:
        vram_monitor.stop()
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"      !! CUDA 显存不足 (OOM)，跳过")
        else:
            print(f"      !! 运行时错误: {e}")
        clear_gpu_memory()
        return None

    except Exception as e:
        vram_monitor.stop()
        print(f"      !! 异常: {e}")
        clear_gpu_memory()
        return None


# ==================== 分阶段参数搜索 ====================

def staged_search(
    task_name: str,
    dataset_root: Path,
    vram_monitor: VRAMMonitor,
) -> tuple[dict, dict | None, float]:
    """
    对单个任务执行四阶段贪心参数搜索。

    每阶段锁定前序最优参数，仅搜索当前维度。候选项按显存消耗升序排列，
    一旦触发 OOM/超限将跳过该阶段后续更重的候选项。

    Returns:
        (best_params, best_results, best_score)
    """
    # 初始默认参数（最轻量配置）
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
        results = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
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
        results = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
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
        results = train_and_evaluate(dataset_root, params, vram_monitor)
        if results is None:
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
        results = train_and_evaluate(dataset_root, params, vram_monitor)
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
) -> dict | None:
    """
    使用搜索到的最佳参数训练最终模型，保存权重和 TensorBoard 日志。

    Returns:
        最终测试结果字典，失败返回 None。
    """
    print(f"\n  使用最佳参数训练最终模型并保存 ...")
    clear_gpu_memory()

    try:
        datamodule = build_datamodule(dataset_root)

        model = Patchcore(
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
            default_root_dir=str(output_dir),
            logger=TensorBoardLogger(save_dir="logs/", name=f"{task_name}_best"),
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )

        trainer.fit(model=model, datamodule=datamodule)
        test_results = trainer.test(model=model, datamodule=datamodule)

        if test_results:
            print(f"    最终模型测试结果:")
            for key, value in test_results[0].items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.4f}")
            print(f"    模型权重已保存至: {output_dir / 'weights'}")
            return test_results[0]

        return None

    except Exception as e:
        print(f"    最终训练失败: {e}")
        clear_gpu_memory()
        return None


# ==================== 主函数 ====================

def main():
    """
    主函数：遍历 10 个任务数据集，对每个任务执行自适应参数搜索 + 最终训练。

    执行流程：
        1. 扫描 datasets/ 下的所有 task_* 目录
        2. 对每个任务执行四阶段参数搜索
        3. 用最佳参数重新训练并保存模型权重
        4. 将所有结果汇总到 adaptive_search_results.json
    """
    datasets_root = Path("./datasets")
    output_root = Path("results")
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = output_root / "adaptive_search_results.json"

    # 初始化显存监控器
    vram_monitor = VRAMMonitor(VRAM_LIMIT_BYTES)

    # 扫描所有任务目录，按编号排序
    task_dirs = sorted(
        [d for d in datasets_root.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not task_dirs:
        print("错误：datasets/ 目录下未找到 task_* 数据集。")
        return

    # GPU 信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.mem_get_info(0)[1] / (1024 ** 3)
        print(f"GPU: {gpu_name}  总显存: {total_gb:.1f}GB  监控上限: {VRAM_LIMIT_GB}GB")
    else:
        print("警告：未检测到 GPU，将使用 CPU（速度极慢）")

    print(f"发现 {len(task_dirs)} 个任务数据集")
    print(f"图像尺寸: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"搜索空间: {len(BACKBONES)} backbones x {len(LAYER_CONFIGS)} layers x "
          f"{len(CORESET_RATIOS)} coreset x {len(NUM_NEIGHBORS_LIST)} neighbors")
    print(f"搜索策略: 四阶段贪心搜索（每任务最多 "
          f"{len(BACKBONES) + len(LAYER_CONFIGS) + len(CORESET_RATIOS) + len(NUM_NEIGHBORS_LIST)} 次训练）")
    print("=" * 70)

    all_results = {}

    for task_idx, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        print(f"\n{'=' * 70}")
        print(f"[{task_idx}/{len(task_dirs)}] {task_name}")
        print(f"{'=' * 70}")

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
        print(f"  {'*' * 50}")

        # 每完成一个任务就保存中间结果（防止中途断电丢失）
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  (中间结果已保存至 {results_file})")

        clear_gpu_memory()

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

    print(f"\n结果文件: {results_file.resolve()}")
    print(f"使用 'tensorboard --logdir=logs/' 查看训练曲线")
    print("完成！")


if __name__ == "__main__":
    main()
