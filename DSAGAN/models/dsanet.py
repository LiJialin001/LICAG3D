import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.encoders import triplane_encoders
from models.eg3d.triplane import TriPlaneGenerator
from configs.paths_config import model_paths
import pickle
from torch_utils import misc

class DSANet(nn.Module):
    def __init__(self, opts):
        super(DSANet, self).__init__()
        self.opts = opts
        #TODO: 更改卡
        self.opts.device = "cuda:0"
        self.set_psp_encoder()
        self.set_eg3d_generator()

        self.load_weights()

    def forward(self, x, camera_params, novel_view_camera_params=None, resize=True, return_latents=False, return_yhat_psp=False, return_triplaneoffsets=False, CTTR=False):
        if novel_view_camera_params is None:
            novel_view_camera_params = camera_params
        x_clone = x.clone().detach()
        y_hat_psp, codes = self.__get_initial_inversion(x, camera_params)
        
        new_codes = codes.clone().detach()
        y_hat_psp_clone = y_hat_psp.clone().detach()
        x_input = torch.cat([y_hat_psp_clone, x_clone - y_hat_psp_clone], dim=1)
        #print(new_codes.shape)
        # print("camera_params")
        # print(camera_params)
        if CTTR:
            images = self.decoder.synthesis(new_codes, camera_params, noise_mode='const')
        else:
            images = self.decoder.synthesis(new_codes, novel_view_camera_params, noise_mode='const')
        y_hat = images['image']
        if resize or CTTR:
           y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))

        if CTTR:
            x_input = torch.cat([y_hat, x_clone - y_hat], dim=1)
        
            images = self.decoder.synthesis(new_codes, novel_view_camera_params, triplane_offsets=triplane_offsets, noise_mode='const')
            y_hat = images['image']
            if resize:
                y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))
        
        if not return_latents and not return_yhat_psp:
            return y_hat
        outputs = [y_hat]
        if return_latents:
            outputs += [codes]
        if return_yhat_psp:
            outputs += [y_hat_psp]
            
        return outputs
    

    def __get_initial_inversion(self, x, camera_params, resize=True):
        # TODO: 
        # codes = self.psp_encoder(x, camera_params)
        codes = self.psp_encoder(x)
        # add to average latent code
        # print("codes")
        # print(codes.shape)
        codes = codes + self.latent_avg
        # codes = codes + self.latent_avg[:,12,:]
        # codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        y_hat = self.decoder.synthesis(codes, camera_params, noise_mode='const')
        y_hat = F.adaptive_avg_pool2d(y_hat['image'], (256, 256))
        return y_hat, codes

    def set_psp_encoder(self):
        self.psp_encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        # self.psp_encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        # self.psp_encoder = triplane_encoders.TriPlane_Encoder(50, 'ir_se', self.opts)

    def set_eg3d_generator(self):
        ckpt = torch.load(model_paths["eg3d_dsa"], map_location='cuda:0')

        # print("ckpt")
        # print(ckpt['init_kwargs'])
        latent_avg1 = torch.load('pretrained_models/myself/latent_avg.pth', map_location='cuda:0')
        latent_avg1 = latent_avg1.squeeze()
        latent_avg1 = torch.unsqueeze(latent_avg1, 0)
        # print("latent_avg1")
        print(latent_avg1.shape)
        self.latent_avg = latent_avg1.to(self.opts.device)
        # self.latent_avg = latent_avg1.to(self.opts.device).repeat(self.opts.n_styles, 1)
        init_args = ()
        # init_kwargs = ckpt['init_kwargs']
        # print("init_kwargs")
        # print(init_kwargs)
        init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                'channel_max': 512, 'fused_modconv_default': 'inference_only',
                'rendering_kwargs': {'depth_resolution': 32, 'depth_resolution_importance': 32, 'ray_start': 2.25, 'ray_end': 3.3,
                        'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
                        'image_resolution': 128, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                        'superresolution_module': 'models.eg3d.superresolution.SuperresolutionHybrid8XDC',
                        'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
                        'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                        'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'},
                'conv_clamp': None, 'c_dim': 25, 'img_resolution': 512, 'img_channels': 3}
        rendering_kwargs = {'image_resolution': 128,
                             'disparity_space_sampling': False, 
                             'clamp_mode': 'softplus', 
                             'superresolution_module': 'models.eg3d.superresolution.SuperresolutionHybrid8XDC',
                               'c_gen_conditioning_zero': False,
                                 'gpc_reg_prob': 0.8, 'c_scale': 1.0,
                                   'superresolution_noise_mode': 'none',
                                     'density_reg': 0.25, 'density_reg_p_dist': 0.004,
                                       'reg_type': 'l1',
                                         'decoder_lr_mul': 1.0,
                                         'sr_antialias': True, 
                                         'depth_resolution': 56, 
                                         'depth_resolution_importance': 56, 
                                         'ray_start': 'auto',
                                           'ray_end': 'auto',
                                             'box_warp': 1,
                                               'white_back': True,
                                                 'avg_camera_radius': 7, 
                                                 'avg_camera_pivot': [0, 0, 0.2]}
        self.decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(self.opts.device)
        self.decoder.neural_rendering_resolution = 128
        self.decoder.rendering_kwargs = rendering_kwargs

        self.decoder.load_state_dict(ckpt['G_ema'], strict=False)
        self.decoder.requires_grad_(False)

    def load_weights(self):

        if self.opts.checkpoint_path is None:
            print('Loading encoders weights from irse100!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            #encoder_ckpt100 = torch.load(model_paths['ir_se100'])
            self.psp_encoder.load_state_dict(encoder_ckpt, strict=False)

            # alter cuz triplane encoder works with concatenated inputs
            shape = encoder_ckpt['input_layer.0.weight'].shape
            altered_input_layer = torch.randn(shape[0], 6, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
            encoder_ckpt['input_layer.0.weight'] = altered_input_layer
        else:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location='cpu')['state_dict']
            self.load_state_dict(checkpoint, strict=True)
            print(f"Loading encoders weights from {self.opts.checkpoint_path}")

