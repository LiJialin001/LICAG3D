import torch

# 加载模型查看其结构
checkpoint = torch.load("/home/k611/data3/ljl/DSAPretrainedModels/iteration_1325000.pt", map_location='cpu')

# 打印键
print("模型文件包含的键：", checkpoint.keys())

# 如果有state_dict，查看其结构
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    # 打印编码器相关的键
    encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
    print("编码器键：", encoder_keys[:10])  # 只打印前10个