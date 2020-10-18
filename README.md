# Attribute Prototype Network for Zero-Shot Learning

The current project page provides [pytorch](http://pytorch.org/) code that implements the following paper:   
**Title:**      "Attribute Prototype Network for Zero-Shot Learning"    
**Authors:**     Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata    
**Project Page:**  https://wenjiaxu.github.io/APN-ZSL/          

**Abstract:**  
From the beginning of zero-shot learning research, visual attributes have been shown to play an important role. In order to better transfer attribute-based knowledge from known to unknown classes, we argue that an image representation with integrated attribute localization ability would be beneficial for zero-shot learning.
To this end, we propose a novel zero-shot representation learning framework that jointly learns discriminative global and local features using only class-level attributes. While a visual-semantic embedding layer learns global features, local features are learned through an attribute prototype network that simultaneously regresses and decorrelates attributes from intermediate features. We show that our locality augmented image representations achieve a new state-of-the-art on three zero-shot learning benchmarks. As an additional benefit, our model points to the visual evidence of the attributes in an image, e.g. for the CUB dataset, confirming the improved attribute localization ability of our image representation. The code will be publicaly available at https://wenjiaxu.github.io/APN-ZSL/.

## Requirements
Python = 3.6

PyTorch = 1.0

CUDA

Numpy

...

## APN image features for CUB, AWA2, SUN that can boost performance of other generative models.


## Code for ZSL(GZSL) on three bench mark datasets CUB, AwA2, SUN (will come up soon).


