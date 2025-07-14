# PFS2C2
JSTARS paper "Spectral-Spatial Collaborative Pretraining Framework with Multi-Constraint Cooperation for Hyperspectral-Multispectral Image Fusion"

Abstract
---------------------
Abstract— Fusing low-resolution hyperspectral images (HSI) with high-resolution multispectral images (MSI) has become a promising technique for generating high-resolution hyperspectral images (HHSI), effectively addressing low spatial resolution limitations of hyperspectral data. Deep learning-based fusion methods have emerged as the dominant approach in recent research. However, almost unsupervised deep fusion methods rely on degradation processes that may disrupt and lose the dominant spectral-spatial information in the original images. Moreover, they only construct spectral inverse mapping while lacking effective spectral-spatial interactions, resulting in insufficient details preservation, thereby affecting fusion performance. In this study, we propose a deep unsupervised pretraining fusion framework of spectral-spatial collaborative constraint (PFS2C2). Specifically, in the first stage, both spatial and spectral inverse modules are adaptively pretrained from the original HSI and MSI, providing initialize parameters for further optimization; in the second stage, the model learns spatial and spectral degradation modules to support the construction of inverse modules for subsequent stage; in the third stage, guided by pretrained parameters, we further optimize the inverse mapping and effectively extract spectral-spatial information at different resolution, thereby enhancing the fusion performance and applicability. Experimental results on simulated and real satellite data verify the superiority of our proposed method in recovering spatial details and preserving spectral fidelity.

Paper Linking
---------------------
https://ieeexplore.ieee.org/abstract/document/10969559

Citation
---------------------

**Please kindly cite the papers if this is something useful and helpful for your research.**

@ARTICLE{10969559,
  author={Jia, Jia and Yu, Haoyang and Wang, Chengjun and Zheng, Ke and Li, Jiaxin and Hu, Jiaochan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Spectral-Spatial Collaborative Pretraining Framework with Multi-Constraint Cooperation for Hyperspectral-Multispectral Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Spatial resolution;Feature extraction;Degradation;Tensors;Hyperspectral imaging;Collaboration;Satellites;Image reconstruction;Optimization;Matrix decomposition;Hyperspectral Remote Sensing;Data Fusion;Unsupervised Model;Pretraining Framework;Spectral-spatial Collaborative Constraint},
  doi={10.1109/JSTARS.2025.3562278}}
