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

import mrcfile

from datasets.dataset import ImageFolderDataset
from utils.common import tensor2im, log_input_image
from metrics.metrics import Metrics
from options.test_options import TestOptions
from models.dsanet import DSANet

from models.eg3d.camera_utils import LookAtPoseSampler
from models.eg3d.shape_utils import extract_shape
from models.eg3d.shape_utils import convert_sdf_samples_to_ply


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    out_path_shapes = os.path.join(test_opts.exp_dir, 'inference_shapes')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_shapes, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = DSANet(opts)
    net.decoder.rendering_kwargs['depth_resolution'] = int(net.decoder.rendering_kwargs['depth_resolution'] * 2)
    net.decoder.rendering_kwargs['depth_resolution_importance'] = int(
        net.decoder.rendering_kwargs['depth_resolution_importance'] * 2)
    net.eval()
    net.cuda()

    if opts.calculate_metrics:
        metrics = Metrics()
        scores = {}
        scores = {
            'mse': [],
            'lpips': [],
            'ms-ssim': [],
            'id-sim_same-view': []

        }
        for angle_y in opts.novel_view_angles:
            scores[f'id-sim_{angle_y}'] = []

    print("Loading Dataset...")
    dataset = ImageFolderDataset(path=opts.data_path,
                                 resolution=None, use_labels=True)

    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    global_time = []
    global_i = 0
    data_dict = dict()
    for x, camera_param, fname in tqdm(dataloader):
        with torch.no_grad():
            x, camera_param = x.cuda().float(), camera_param.cuda().float()
            tic = time.time()
            results_batch, latents_batch, sigmas = run_on_batch(x, camera_param, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

            # 为当前批次中的每个图像单独保存latent code
            for i in range(len(fname)):
                img_name = fname[i].strip("(),")
                # 创建该图像的数据字典
                single_img_data = {
                    "latents_batch": latents_batch[i].cpu().numpy() if torch.is_tensor(latents_batch) else
                    latents_batch[i],
                    "camera_param": camera_param[i].cpu().numpy() if torch.is_tensor(camera_param) else camera_param[i]
                }
                # 保存到单独的文件
                np.save(os.path.join(out_path_results, f'{img_name}_latent.npy'), single_img_data)

        for i in range(opts.test_batch_size):
            res = None
            for j in range(len(results_batch)):
                result = tensor2im(results_batch[j][i])

                if opts.couple_outputs:
                    input_im = log_input_image(x[i], opts)
                    resize_amount = (256, 256)
                    if res is None:
                        res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                              np.array(result.resize(resize_amount))], axis=1)
                    else:
                        res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                if j == 0:
                    im_save_path = os.path.join(out_path_results, f'{fname[i]}_same-view.png')
                    if opts.calculate_metrics:
                        scores['mse'].append(metrics.mse(x[i], results_batch[j][i]))
                        scores['lpips'].append(metrics.lpips(x[i], results_batch[j][i]))
                        scores['ms-ssim'].append(metrics.ms_ssim(x[i], results_batch[j][i]))
                        id_sim = metrics.id_similarity(x[i], results_batch[j][i])
                        if id_sim is not None:
                            scores['id-sim_same-view'].append(id_sim)
                else:
                    im_save_path = os.path.join(out_path_results, f'{fname[i]}_{opts.novel_view_angles[j - 1]}.png')
                    if opts.calculate_metrics:
                        id_sim = metrics.id_similarity(x[i], results_batch[j][i])
                        if id_sim is not None:
                            scores[f'id-sim_{opts.novel_view_angles[j - 1]}'].append(id_sim)
                Image.fromarray(np.array(result)).save(im_save_path)
            input_im = (log_input_image(x[i], opts)).save(os.path.join(out_path_results, f'{fname[i]}_original.png'))
            # if opts.shapes:
            # convert_sdf_samples_to_ply(np.transpose(sigmas[i], (2, 1, 0)), [0, 0, 0], 1, os.path.join(out_path_shapes, f'{fname[i]}.ply'), level=10)
            # mrc_sigmas = torch.tensor(sigmas[i].copy())
            # with mrcfile.new_mmap(os.path.join(out_path_shapes, f'{fname[i]}.mrc'), overwrite=True, shape=mrc_sigmas.shape, mrc_mode=2) as mrc:
            #     mrc.data[:] = mrc_sigmas
            # global_i += 1
    print(data_dict)
    # TODO:
    # np.save(out_path_results + "/data_w_c.npy", data_dict)
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    if opts.calculate_metrics:
        scores = {key: float(np.nanmean(np.array(value))) for key, value in scores.items()}
        result_str += '\n' + f'{str(scores)}'
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(x, camera_param, net, opts):
    aspect_classes = [
        [-30, 20],
        [-0, 30],
        [30, 20],
        [-30, 0],
        [0, 0],
        [45, 0],
        [-30, -20],
        [0, -30],
        [40, -30]]
    outputs_batch = net(x, camera_param, resize=True, return_latents=True, return_triplaneoffsets=False, CTTR=opts.CTTR)
    results_batch, latents_batch = [outputs_batch[0]], [outputs_batch[1]]
    for angle_y in opts.novel_view_angles:
        # angle_p = 0
        # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p,
        #                                        torch.tensor([0, 0, 0.2]).cuda(),
        #                                        radius=2.7, batch_size=opts.test_batch_size, device="cuda")
        primary, secondary = aspect_classes[int(angle_y)]
        [x_delta, y_delta, z_delta] = [0, 0, 0]
        cam2world_pose = LookAtPoseSampler.sampleDSA(primary, secondary, 829.4015277, device="cuda")
        novel_view_camera_params = camera_param.clone()
        # print(novel_view_camera_params.shape)
        # print('--------------------------------')
        # print(cam2world_pose.view(cam2world_pose.size(0), -1).shape)
        # print(cam2world_pose.view(novel_view_camera_params.size(0), -1).shape)
        novel_view_camera_params[:, :16] = cam2world_pose.view(novel_view_camera_params.size(0), -1)
        results_out = net(x, camera_param, novel_view_camera_params=novel_view_camera_params, resize=True,
                          return_latents=True, CTTR=opts.CTTR)
        results_batch += [results_out[0]]
        latents_batch += [results_out[1]]
        # results_batch += [net(x, camera_param, novel_view_camera_params=novel_view_camera_params, resize=True, CTTR=opts.CTTR, return_latents=True)]
    sigmas = []
    if opts.shapes:
        for i in range(latents_batch.shape[0]):
            sigmas += [extract_shape(net.decoder, latents_batch[i:i + 1, :])]

    return results_batch, latents_batch, sigmas


if __name__ == '__main__':
    run()
