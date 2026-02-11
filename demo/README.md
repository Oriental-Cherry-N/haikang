# 小样本缺陷检测 Demo

基于 **anomalib 0.7.0** 的小样本工业缺陷检测系统。

## 目录结构

```
demo/
├── quick_start.py          # 快速入门示例（推荐首先运行）
├── advanced_train.py       # 高级训练脚本（含数据增强、域自适应）
├── adaptive_inference.py   # 自适应推理脚本
├── explainability.py       # 可解释性分析脚本
├── multi_task_train.py     # 多任务批量训练
├── config/
│   └── default_config.yaml # 默认配置文件
├── requirements.txt        # Python 依赖
└── README.md              # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd demo
pip install -r requirements.txt
```

### 2. 运行快速入门示例

```bash
python quick_start.py
```

按照提示选择要运行的示例。

### 3. 单任务训练

```bash
# 基础训练
python advanced_train.py --task task_1

# 指定模型和参数
python advanced_train.py --task task_1 --model patchcore --backbone resnet18 --image_size 512 512

# 使用配置文件
python advanced_train.py --task task_1 --config config/default_config.yaml

# 禁用数据增强
python advanced_train.py --task task_1 --no_augmentation
```

### 4. 多任务批量训练

```bash
# 训练所有任务
python multi_task_train.py --tasks all

# 训练指定任务
python multi_task_train.py --tasks task_1 task_2 task_3

# 跳过已完成的任务
python multi_task_train.py --tasks all --skip_existing
```

### 5. 自适应推理

```bash
# 单张图像推理
python adaptive_inference.py --model results/task_1/weights/model.ckpt --image test.png

# 批量推理
python adaptive_inference.py --model results/task_1/weights/model.ckpt --image_dir datasets/task_1/defect

# 启用自适应阈值
python adaptive_inference.py --model results/task_1/weights/model.ckpt --image_dir datasets/task_1/defect --adaptive_threshold

# 保存可视化结果
python adaptive_inference.py --model results/task_1/weights/model.ckpt --image_dir datasets/task_1/defect --save_visualization
```

### 6. 可解释性分析

```bash
# 单张图像分析
python explainability.py --model results/task_1/weights/model.ckpt --image test.png

# 批量分析
python explainability.py --model results/task_1/weights/model.ckpt --image_dir datasets/task_1/defect
```

## 配置说明

详细配置参见 `config/default_config.yaml`，主要配置项：

### 模型配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model.name` | 模型类型 | `patchcore`（首选） |
| `model.backbone` | 骨干网络 | `resnet18`（小样本） |
| `model.coreset_sampling_ratio` | 核心集采样比例 | `0.1`（小样本适当增大） |

### 数据配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `data.image_size` | 输入图像尺寸 | `[512, 512]` |
| `data.train_batch_size` | 训练批次大小 | `1` |

### 数据增强

启用数据增强可有效缓解小样本问题：

- 几何变换：翻转、旋转、平移、缩放
- 颜色变换：亮度、对比度、饱和度
- 噪声模糊：高斯噪声、高斯模糊

## 技术文档

详细技术文档请参阅：`../docs/小样本缺陷检测技术文档.md`

包含：
- 小样本学习策略
- 领域自适应方法
- 可解释性机制
- 算法选型指南

## 常见问题

### Q: 训练样本太少效果差怎么办？

1. 启用数据增强
2. 增大 `coreset_sampling_ratio`
3. 使用 PaDiM 代替 PatchCore

### Q: 不同批次产品检测效果差？

1. 启用域自适应（特征归一化）
2. 使用增量学习更新模型
3. 增加训练数据多样性

### Q: 推理速度慢？

1. 减小输入图像尺寸
2. 使用更轻量的骨干网络
3. 减小 `coreset_sampling_ratio`
4. 导出为 ONNX 格式

## 许可证

MIT License
