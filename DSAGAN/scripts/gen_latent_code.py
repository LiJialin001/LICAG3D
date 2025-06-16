import os
import torch
import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(".")  # 如果在DSAGAN目录下运行
sys.path.append("..")  # 如果在DSAGAN/scripts目录下运行

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from datasets.dataset import ImageFolderDataset
from models.encoders import psp_encoders

# 配置路径
checkpoint_path = "/home/k611/data3/ljl/DSAPretrainedModels/iteration_1325000.pt"
data_path = "/home/k611/data3/ljl/DSADatasets/hangyi_converted_cropped"
output_dir = "/home/k611/data3/lys/encoderresult/codes"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
print(f"加载模型：{checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 创建编码器
encoder = psp_encoders.GradualStyleEncoder(50, mode='ir_se')

# 提取编码器参数
encoder_state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if key.startswith('psp_encoder.'):
        new_key = key.replace('psp_encoder.', '')
        encoder_state_dict[new_key] = value

# 加载参数到编码器
encoder.load_state_dict(encoder_state_dict)
encoder = encoder.eval().cuda()
print("成功加载编码器权重")

# 加载数据
dataset = ImageFolderDataset(path=data_path)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
print(f"数据集大小：{len(dataset)}")

# 提取并保存编码
for i, batch in enumerate(dataloader):
    img, camera_params, name = batch
    img = img.cuda()

    with torch.no_grad():
        # 获取编码
        img = img.float()  # 将双精度转换为单精度
        codes = encoder(img)
        if codes.ndim == 2:
            codes = codes.reshape(codes.shape[0], 1, codes.shape[-1])

    # 保存编码和相机参数
    save_dict = {
        'latent_codes': codes.cpu(),
        'camera_params': camera_params,
        'image_name': name[0] if isinstance(name, list) else name
    }

    # 使用图像名称作为文件名，避免冲突
    filename = os.path.basename(name[0]) if isinstance(name, list) else f"{i:05d}"
    save_path = os.path.join(output_dir, f"{filename}_codes.pt")

    torch.save(save_dict, save_path)
    print(f"[{i + 1}/{len(dataset)}] 已保存编码到 {save_path}")