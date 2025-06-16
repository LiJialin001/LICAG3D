import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from models.dsanet import DSANet
from models.eg3d.camera_utils import LookAtPoseSampler
from utils.common import tensor2im


def generate_image_from_latent(latent_path, model_path, output_dir, view_index=None, custom_angle=None, device='cuda'):
    """从latent code生成图像

    Args:
        latent_path: latent code文件路径
        model_path: 模型检查点路径
        output_dir: 输出目录
        view_index: 预定义视角索引（0-8）
        custom_angle: 自定义角度 [primary, secondary]
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载latent code
    print(f"加载latent code: {latent_path}")
    try:
        data = np.load(latent_path, allow_pickle=True).item()
        latent_code = torch.tensor(data["latents_batch"]).unsqueeze(0).to(device)
        original_camera = torch.tensor(data["camera_param"]).unsqueeze(0).to(device)
    except Exception as e:
        print(f"加载latent code失败: {e}")
        return

    # 加载模型
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts = Namespace(**opts)

    opts.checkpoint_path = model_path

    net = DSANet(opts)
    net.eval()
    net.to(device)

    # 预定义视角
    aspect_classes = [
        [-30, 20],  # 0
        [0, 30],  # 1
        [30, 20],  # 2
        [-30, 0],  # 3
        [0, 0],  # 4
        [45, 0],  # 5
        [-30, -20],  # 6
        [0, -30],  # 7
        [40, -30]  # 8
    ]

    # 确定要使用的角度
    if custom_angle is not None:
        primary, secondary = custom_angle
    elif view_index is not None and 0 <= view_index < len(aspect_classes):
        primary, secondary = aspect_classes[view_index]
    else:
        # 默认使用正面视角
        primary, secondary = aspect_classes[4]  # [0, 0]

    # 生成新相机视角
    print(f"生成视角 [Primary: {primary}, Secondary: {secondary}]")
    cam2world_pose = LookAtPoseSampler.sampleDSA(primary, secondary, 829.4015277, device=device)
    novel_view_camera = original_camera.clone()
    novel_view_camera[:, :16] = cam2world_pose.view(novel_view_camera.size(0), -1)

    # 生成图像部分的代码改为：
    with torch.no_grad():
        # 直接使用decoder进行合成
        outputs = net.decoder.synthesis(latent_code, novel_view_camera, noise_mode='const')

        # 打印输出字典的键和形状以便调试
        print("输出字典键:", outputs.keys())
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"键: {key}, 形状: {value.shape}")

        # 正确处理图像输出
        if 'image' in outputs and len(outputs['image'].shape) == 4:
            # 假设image形状为[batch, channel, height, width]
            image = outputs['image']

            # 将[-1,1]范围转换为[0,1]
            image_np = (image + 1) / 2.0
            image_np = image_np.clamp(0, 1)

            # 转换为适合PIL的格式
            image_np = image_np.cpu().squeeze(0).permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            result_img = Image.fromarray(image_np)
        else:
            # 如果找不到有效的图像输出，尝试其他可能的键
            for key in ['final_image', 'rgb', 'img', 'output']:
                if key in outputs and torch.is_tensor(outputs[key]) and len(outputs[key].shape) == 4:
                    image = outputs[key]
                    image_np = (image + 1) / 2.0
                    image_np = image_np.clamp(0, 1)
                    image_np = image_np.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    result_img = Image.fromarray(image_np)
                    print(f"使用'{key}'作为图像源")
                    break
            else:
                raise ValueError("无法找到有效的图像输出，请检查网络输出结构")


    file_name = os.path.basename(latent_path).replace('_latent.npy', '')
    output_path = os.path.join(output_dir, f'{file_name}_view_{primary}_{secondary}.png')
    result_img.save(output_path)

    print(f"已保存生成的图像到: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="从latent code生成特定视角的图像")
    parser.add_argument("--latent", type=str, required=True, help="latent code文件路径(.npy)")
    parser.add_argument("--model", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output", type=str, default="rendered_images", help="输出目录")
    parser.add_argument("--view", type=int, default=None, help="预定义视角索引(0-8)")
    parser.add_argument("--primary", type=float, default=None, help="自定义主角度")
    parser.add_argument("--secondary", type=float, default=None, help="自定义次角度")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 检查是否指定了自定义角度
    custom_angle = None
    if args.primary is not None and args.secondary is not None:
        custom_angle = [args.primary, args.secondary]

    generate_image_from_latent(
        latent_path=args.latent,
        model_path=args.model,
        output_dir=args.output,
        view_index=args.view,
        custom_angle=custom_angle,
        device=args.device
    )