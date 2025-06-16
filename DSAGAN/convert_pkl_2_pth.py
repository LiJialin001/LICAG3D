# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from torch_utils import misc
from models.eg3d.triplane import TriPlaneGenerator

# ----------------------------------------------------------------------------


@click.command()
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--network_pth', help='Network pth ckpt filename', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def convert(
        network_pkl: str,
        network_pth: str,
        shuffle_seed: Optional[int],
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    save_dict = {
        'G_ema': G.state_dict(),
        'rendering_kwargs':G.rendering_kwargs
    }
    print(G.rendering_kwargs)
    print('save pth to',network_pth)
    torch.save(save_dict, network_pth)

    # ======validate=============

    init_args = ()
    init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                   'channel_max': 512, 'fused_modconv_default': 'inference_only',
                   'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25,
                                        'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                                        'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                                        'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                        'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                        'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                                        'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                                        'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                        'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                   'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'},
                   'conv_clamp': None, 'c_dim': 25, 'img_resolution': 512, 'img_channels': 3}
    rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25, 'ray_end': 3.3,
                        'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
                        'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                        'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                        'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
                        'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                        'sr_antialias': True}

    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)

    ckpt = torch.load(network_pth)
    G_new.load_state_dict(ckpt['G_ema'], strict=False)
    G_new.neural_rendering_resolution = 128

    G_new.rendering_kwargs = rendering_kwargs

    G_new.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G_new.rendering_kwargs['depth_resolution_importance'] = int(
        G_new.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    convert()

# ----------------------------------------------------------------------------