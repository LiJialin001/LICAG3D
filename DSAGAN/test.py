import torch

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 检查PyTorch编译时使用的CUDA版本
print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")

# 检查可用的GPU数量和名称
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")