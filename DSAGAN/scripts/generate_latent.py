import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from datasets.dataset import ImageFolderDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.dsanet import DSANet


def run():
    test_opts = TestOptions().parse()


    # 硬编码数据集路径
    test_opts.data_path = "/home/k611/data3/ljl/DSAGAN/testimage"
    test_opts.checkpoint_path = '/home/k611/data3/ljl/DSAPretrainedModels/iteration_1325000.pt'
    test_opts.exp_dir = '/home/k611/data3/lys/encoderresult/'
    # 创建专门用于保存latent codes的目录
    out_path_latents = os.path.join(test_opts.exp_dir, 'latent_codes')
    os.makedirs(out_path_latents, exist_ok=True)

    # 更新测试选项
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    # 初始化模型
    print("加载模型...")
    net = DSANet(opts)
    net.eval()
    net.cuda()

    # 加载数据集
    print("加载数据集...")
    dataset = ImageFolderDataset(path=opts.data_path, resolution=None, use_labels=True)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    # 处理数据部分修改
    for x, camera_param, fname in tqdm(dataloader):
        with torch.no_grad():
            x = x.cuda().float()

            # 检查相机参数是否有效
            print(f"原始相机参数形状: {camera_param.shape}")

            # 如果相机参数为空或无效，创建默认相机参数
            if camera_param.shape[1] == 0 or camera_param.numel() == 0:
                print("创建默认相机参数...")
                # 创建25维的默认相机参数（16个用于变换矩阵，9个用于内参）
                camera_param = torch.zeros((x.size(0), 25), dtype=torch.float32, device=x.device)

                camera_param = torch.zeros((x.size(0), 25), dtype=torch.float32, device=x.device)

                # 设置相机外参（变换矩阵）- 使用示例中的实际值
                camera_param[:, 0] = -0.622501144664564  # 第1行第1列
                camera_param[:, 1] = -0.21702722305305094  # 第1行第2列
                camera_param[:, 2] = 0.7519122009001636  # 第1行第3列
                camera_param[:, 3] = -4.032395354246247  # 第1行第4列

                camera_param[:, 4] = -0.782586814501385  # 第2行第1列
                camera_param[:, 5] = 0.1726322144335806  # 第2行第2列
                camera_param[:, 6] = -0.5980979707767461  # 第2行第3列
                camera_param[:, 7] = 3.207512094971956  # 第2行第4列

                camera_param[:, 8] = 0.0  # 第3行第1列
                camera_param[:, 9] = -0.9607746569857168  # 第3行第2列
                camera_param[:, 10] = -0.2773141361996311  # 第3行第3列
                camera_param[:, 11] = 1.4871952245747364  # 第3行第4列

                camera_param[:, 12] = 0.0  # 第4行第1列
                camera_param[:, 13] = 0.0  # 第4行第2列
                camera_param[:, 14] = 0.0  # 第4行第3列
                camera_param[:, 15] = 1.0  # 第4行第4列

                # 设置相机内参
                camera_param[:, 16] = 6.988129058441559  # fx
                camera_param[:, 17] = 0.0  # 0
                camera_param[:, 18] = 0.5  # cx
                camera_param[:, 19] = 0.0  # 0
                camera_param[:, 20] = 6.988129058441559  # fy
                camera_param[:, 21] = 0.5  # cy
                camera_param[:, 22] = 0.0  # 0
                camera_param[:, 23] = 0.0  # 0
                camera_param[:, 24] = 1.0  # 1

            # 直接使用psp_encoder获取latent codes
            latents_batch = net.psp_encoder(x)
            latents_batch = latents_batch + net.latent_avg

            # 为每个图像保存单独的latent code
            for i in range(x.size(0)):
                img_name = fname[i].strip("(),")

                # 创建数据字典，使用有效的相机参数
                single_img_data = {
                    "latents_batch": latents_batch[i].cpu().numpy(),
                    "camera_param": camera_param[i].cpu().numpy()  # 现在这是有效的相机参数
                }

                # 保存latent code
                latent_path = os.path.join(out_path_latents, f'{img_name}_latent.npy')
                np.save(latent_path, single_img_data)
                print(f"保存latent code到: {latent_path}")

                # 保存原始图像作为参考
                input_im = log_input_image(x[i], opts)
                input_im.save(os.path.join(out_path_latents, f'{img_name}_original.png'))

    print(f"完成！所有latent codes已保存到 {out_path_latents}")


if __name__ == '__main__':
    run()