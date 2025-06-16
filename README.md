# LICAG3D
LICAG3D: A Low-Information Coronary Angiography 3D Reconstruction method using Generative AI

## About

### What's included in this Repo
- Data preprocessing scripts (keyframe localization, image crop, style unification)
- Training code for X-ray rendering based EG3D
-	Dual-view inversion via pSp encoder with tuple loss
-	Multi-stage GAN inversion & camera translation correction
-	Clinical simulation test pipeline
-	Sample reconstruction & style conversion tools
-	Pretrained model links (coming soon)

### Focus of the current work
Three-dimensional(3D) visualization of coronary artery structure plays a crucial role in guiding the optimal projection views and condition observation during percutaneous coronary intervention(PCI). We propose a low-information CAG 3D reconstruction method(LICAG3D) based on Generative Adversarial Networks(GAN). This method utilizes the relationship between CAG images and the 3D generative latent space to match frames and register multi-view, then learns the X-Ray penetration of the patient's coronary arteries through GAN inversion, reconstructing the 3D structure.

<img width="783" alt="image" src="https://github.com/user-attachments/assets/f302af12-a7d0-45a3-8e7c-41d79f7b6c1f" />

### Key Contributions
- A novel X-ray rendering-based EG3D generator adapted for grayscale coronary angiography.
- A double-tuple loss pSp encoder that maintains 3D consistency across multi-view CAG images.
- multi-stage GAN inversion framework incorporating camera translation correction using an evolutionary algorithm.
- tyle unification via CycleGAN to generalize across multi-center clinical datasets.
- Quantitative and qualitative evaluations, including reader studies with interventional cardiologists.

## Requirements

```bash
Python >= 3.8  
PyTorch >= 1.13  
CUDA Toolkit + GPU with >= 24GB VRAM  
```
Python packages:

- `torch`, `lpips`, `numpy`, `pillow`, `scipy`, `tqdm`, `imageio`, `matplotlib`, `pydicom`, `kornia`

External:

- [EG3D](https://github.com/NVlabs/eg3d)  
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)  
- [pSp Encoder](https://github.com/eladrich/pixel2style2pixel)  
- [ITK-SNAP](http://www.itksnap.org/) (for CTA annotation)  

## Dataset & Preprocessing

LICAG3D is trained on a multicenter Chinese CAG dataset:

| Hospital | Location  | Device          | Data Use               |
| -------- | --------- | --------------- | ---------------------- |
| HY       | Tianjin   | Siemens         | Train: 2873 / Test: 12 |
| BD       | Hebei     | GE              | Test: 4                |
| DG       | Guangdong | Siemens         | Test: 4                |
| TL       | Liaoning  | Philips/Neusoft | Test: 4                |

**Preprocessing steps:**

- Keyframe extraction via cosine similarity (pSp)  
- 4% image cropping to remove border artifacts  
- Camera calibration in world coordinates  
- CycleGAN style harmonization for multi-center data  

##  Pipeline Steps

### 1. Train X-ray EG3D

```bash
python DSAGAN/eg3d/train.py --outdir=OUTDIR --cfg=dsanet --data=CAGDATASET --gpus=2 --batch=2 --neural_rendering_resolution_initial=128 --gamma=5 --aug=ada --neural_rendering_resolution_final=128 --gen_pose_cond=True --gpc_reg_prob=0.8 --snap=1 --metrics=none
```

### 2. Generate images, videos, and 3D models

```bash
python DSAGAN/scripts/gen_videos.py --outdir=RESULT_PATH --trunc=0.7 --seeds=134 --grid=2x2 --network=EG3D_MODEL_PATH --num-keyframes=2 --v_num=3 --v_range=32
```

### 3. Train pSp Encoder with Tuple Loss

```bash
python DSAGAN/scripts/train_psp.py --exp_dir=RESULT_PATH --device=cuda:0 --n_styles=14 --batch_size=1 --test_batch_size=1 --workers=8 --test_workers=8 --val_interval=2500 --save_interval=5000 --checkpoint_path=COUNTINE_TRAIN_MODEL_PATH
```

#### Encoder inference

```bash
python scripts/inference_psp.py --exp_dir=RESULT_PATH --checkpoint_path=ENCODER_MODEL_PATH --data_path=DATA_PATH --test_batch_size=1 --test_workers=4
```

### 3. GAN Inversion (Multi-stage Optimization)

```bash
python invert.py --target_img=img1.png --ref_img=img2.png --ctc=True
```

### 4. Clinical Evaluation

Compare 3D outputs with CTA annotations using:

- SSIM  
- Reader scoring  
- Occlusion diagnosis and correction  

## Performance

| Metric                | Value                                 |
| --------------------- | ------------------------------------- |
| FID (↓)               | 8.49                                  |
| SSIM (avg)            | 0.76 (internal), 0.73–0.78 (external) |
| Precision / Recall    | 0.14 / 0.41                           |
| Reader Satisfaction ↑ | +20.17% after 3D adjustment           |

---

## 🔍 Use Cases

- **PCI Guidance**: Recommend optimal projection views  
- **Stenosis Assessment**: Reconstruct pre/post intervention  
- **Medical Training**: Visualize anatomical variations in 3D  

## 📝 Acknowledgements

- [EG3D](https://github.com/NVlabs/eg3d)  
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)  
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
- [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)  

## 📁 Directory Structure

```
LICAG3D/
├── DSAGAN/                    # Main GAN-based reconstruction code
│   ├── configs/              # Model configs and conversion scripts
│   ├── criteria/             # Loss functions
│   ├── dataset_preprocessing/ # Data preprocessing scripts
│   ├── datasets/             # Dataset loader modules
│   ├── dnnlib/               # Deep learning utilities
│   ├── latentcode/           # Latent code manipulation
│   ├── metrics/              # Evaluation metrics (e.g., SSIM)
│   ├── models/               # Network architectures
│   ├── options/              # Runtime argument parsing
│   ├── pytorch_ssim/         # SSIM metric implementation
│   ├── scripts/              # Entry point scripts (e.g., test.py)
│   ├── torch_utils/          # Torch-level tools
│   ├── training/             # Training pipeline
│   ├── utils/                # General utility functions
│   └── viz/                  # Visualization tools
├── LICENSE                   # License file
└── README.md                 # Project description
```

