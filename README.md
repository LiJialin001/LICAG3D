# LICAG3D
LICAG3D: A Low-Information Coronary Angiography 3D Reconstruction method using Generative AI

## About

### What's included in this Repo
This repository contains preprocessing codes for coronary angiography generation, inversion, and temporal registration, as well as links to relevant pre trained models.

### Focus of the current work
Three-dimensional(3D) visualization of coronary artery structure plays a crucial role in guiding the optimal projection views and condition observation during percutaneous coronary intervention(PCI). We propose a low-information CAG 3D reconstruction method(LICAG3D) based on Generative Adversarial Networks(GAN). This method utilizes the relationship between CAG images and the 3D generative latent space to match frames and register multi-view, then learns the X-Ray penetration of the patient's coronary arteries through GAN inversion, reconstructing the 3D structure.

<img width="783" alt="image" src="https://github.com/user-attachments/assets/f302af12-a7d0-45a3-8e7c-41d79f7b6c1f" />

## Requirements


## How to achieve three-dimensional reconstruction of coronary angiography

### Preparing the training data and labels


### Training a eg3d model on radiological data


### Projecting test radiographs to latent space


### Synthetic three-dimensional contrast imaging


## Locate the location of narrow lesions


## Assess the severity of the lesion




[eg3d](https://github.com/NVlabs/eg3d)
This repository is an official repository of eg3d.

## Requirements
- **eg3d** (https://github.com/NVlabs/eg3d)
