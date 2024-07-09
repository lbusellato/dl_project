# Learning Disentangled Representations via Mutual Information Estimation

This repository contains the code for the final project for the Deep Learning course @ UniVR. The code is based on [Learning-Disentangled-Representations-via-Mutual-Information-Estimation](https://github.com/MehdiZouitine/Learning-Disentangled-Representations-via-Mutual-Information-Estimation) an on the [paper of the same name](https://arxiv.org/abs/1912.03915) by Sanchez et. al. The code was adapted for using the [Shapes3D](https://github.com/google-deepmind/3d-shapes) dataset.

## Setup

The project was run on a commercial-grade laptop with an Intel i7-7500@2.70GHz CPU and an NVIDIA 940MX GPU with CUDA version 11.4. The project uses PyTorch v2.3.1+cu118.

The Shapes3D dataset is automatically downloaded and placed in a "cache" folder at the root of the repo. During training, a new dataset is created and saved in the cache folder.

## Usage

The method is split into two training stages, one that deals with learning the common attributes of image pairs (i.e. object scale and shape and scene orientation) and one that deals with learning the image-specific attributes (i.e. object, wall and floor hues).

### Shared Representation Learning

The training is carried out by launching:

```
python sdim_training.py
```

The parameters for the shared representation learning phase can be set in the [configuration file](conf/share_conf.yaml).


### Exclusive Representation Learning

Before launching, the path to the trained shared encoders should be set in [edim_training.py:41](edim_training.py). The path can be found under mlruns/.

The training is then carried out by launching:

```
python edim_training.py
```

The parameters for the exclusive representation learning phase can be set in the [configuration file](conf/exclusive_conf.yaml).

### Results visualization

A script for plotting the learning curves (wrt accuracy) as well as computing the accuracy at convergence is provided. It can be launched as:

```
python analyze_results.py
```