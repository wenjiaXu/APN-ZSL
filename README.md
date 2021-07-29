# Attribute Prototype Network for Zero-Shot Learning

The current project page provides [pytorch](http://pytorch.org/) code that implements the following paper:   
**Title:**      "Attribute Prototype Network for Zero-Shot Learning"    
**Authors:**     Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata    
**Project Page:**  https://wenjiaxu.github.io/APN-ZSL/          


**Abstract:**  
From the beginning of zero-shot learning research, visual attributes have been shown to play an important role. In order to better transfer attribute-based knowledge from known to unknown classes, we argue that an image representation with integrated attribute localization ability would be beneficial for zero-shot learning.
To this end, we propose a novel zero-shot representation learning framework that jointly learns discriminative global and local features using only class-level attributes. While a visual-semantic embedding layer learns global features, local features are learned through an attribute prototype network that simultaneously regresses and decorrelates attributes from intermediate features. We show that our locality augmented image representations achieve a new state-of-the-art on three zero-shot learning benchmarks. As an additional benefit, our model points to the visual evidence of the attributes in an image, e.g. for the CUB dataset, confirming the improved attribute localization ability of our image representation. 

## Requirements
Python 3.7.7

PyTorch = 1.8.1

All experiments are performed with one Quadro RTX 8000 GPU.

## Prerequisites


- Dataset: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html), and change the opt.image_root to the dataset root path on your machine
  
- Data split and APN image features: please download the [data](https://drive.google.com/file/d/12ZsOxlkKU0IfXEfhB8NHRvHzfGFdwlhB/view?usp=sharing) folder and place it in *./data/*.

- Pre-trained models: please download the [pre-trained models](https://drive.google.com/file/d/1c5scuU0kZS5a9Rz3kf5T0UweCvOpGsh2/view?usp=sharing) and place it in *./pretrained_models/*.

## Code Structures
There are four parts in the code.
 - `model`: It contains the main files of the APN network.
 - `data`: The dataset split, as well as the APN feature extracted from our APN model.
 - `ABP`: The code from [ZSL_ABP](https://github.com/EthanZhu90/ZSL_ABP), we can reproduce the results of applying our APN feature on ABP model reported in the paper.
 - `pretrained_models`: The pretrained models.
 - `script`: The training scripts for APN, e.g., *./script/SUN_ZSL.sh*, etc. The training scripts for APN+ABP, i.e., *./script/SUN_APN_ABP.sh*, etc.

If you use any content of this repo for your work, please cite the following bib entry:

    @inproceedings{xu2020attribute,
      author    = {Xu, Wenjia and Xian, Yongqin and Wang, Jiuniu and Schiele, Bernt and Akata, Zeynep},
      title     = {Attribute prototype network for zero-shot learning},
      booktitle = {NeurIPS},
      year      = {2020}
    }

The code is under construction. If you have problems, feel free to reach me at xuwenjia16@mails.ucas.ac.cn

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ZSL_ABP](https://github.com/EthanZhu90/ZSL_ABP)

- [CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning)

- [GEM-ZSL](https://github.com/osierboy/GEM-ZSL)

- [FEAT](https://github.com/Sha-Lab/FEAT)
