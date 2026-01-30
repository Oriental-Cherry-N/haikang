import torch

print(f"PyTorch 是否检测到 GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"CUDA 版本: {torch.version.cuda}")