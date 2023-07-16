<h1 align="center"> One model is enough: Toward multiclass weakly supervised remote sensing image semantic segmentation </h1> 

<h5 align="center"><em>Zhenshi Li, Xueliang Zhang, and Pengfeng Xiao</em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Usage">Usage</a>|
  <a href="#Visual results">Usage</a>|
  <a href="#Acknowledgement">Acknowledgement</a> |
  <a href="#Statement">Statement</a>
</p >
<p align="center">
<a href="https://ieeexplore.ieee.org/document/10105625"><img src="https://img.shields.io/badge/Paper-IEEE%20TGRS-red"></a>
</p>



## Introduction

This is the official repository for the paper [“One model is enough: Toward multiclass weakly supervised remote sensing image semantic segmentation”](https://ieeexplore.ieee.org/abstract/document/10167684), based on [SEAM](https://github.com/YudeWang/SEAM) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

**Abstract:** Semantic segmentation of remote sensing images is effective for large-scale land cover mapping, which heavily relies on a large amount of training data with laborious pixellevel labeling. Weakly supervised semantic segmentation (WSSS) based on image-level labels has attracted intensive attention due to its easy availability. However, existing image-level WSSS methods for remote sensing images mainly focus on binary segmentation, which are difficult to apply to multiclass scenarios. This study proposes a comprehensive framework for image-level multiclass WSSS of remote sensing images, consisting of appropriate image-level label generation, high-quality pixel-level pseudo mask generation, and segmentation network iterative training. Specifically, a training sample filtering method, as well as a dataset cooccurrence evaluation metric, is proposed to demonstrate proper image-level training samples. Leveraging multiclass class activation maps, an uncertainty-driven pixel-level weighted mask is proposed to relieve the overfitting of labeling noise in pseudo masks when training the segmentation network. Extensive experiments demonstrate that the proposed framework can achieve high-quality multiclass WSSS performance with image-level labels, which can attain 94.23% and 90.77% of the IoUs from pixel-level labels for the ISPRS Potsdam and Vaihingen datasets, respectively. Beyond that, for the DeepGlobe dataset with more complex landscapes, the WSSS framework can achieve an accuracy close to 99% of the fully supervised case. Additionally, we further demonstrate that compared to adopting multiple binary WSSS models, directly training a multiclass WSSS model can achieve better results, which can provide new thoughts to achieve WSSS of remote sensing images for multiclass application scenarios. Our code is public at https://github.com/NJU-LHRS/OME.

<figure>
<div align="center">
<img src=Figure/Fig1.png width="90%">
</div>
</figure>

## Usage

### Classification network
Several classification networks are adopted in our OME inplement. We show our SEAM-based network here.
1. co-occurrence matrix generation: Myutils/calculate_cooccur_matrix.py
2. SEAM training: train_SEAM.py
3. SEAM inference: infer_SEAM.py
4. uncertainty generation from CAMs: generate_uncertainty_from_cam.py

### Segmentation network
Please refer the segmentation phase to [URN](https://github.com/xmed-lab/URN) based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), with a improved reweighting operation as in our paper.

## Visual results
Results on the ISPRS Potsdam dataset
<figure>
<div align="center">
<img src=Figure/Fig4.png width="90%">
</div>
</figure>

Results on the ISPRS Vaihingen dataset
<figure>
<div align="center">
<img src=Figure/Fig5.png width="90%">
</div>
</figure>

Results on the Deepglobe dataset
<figure>
<div align="center">
<img src=Figure/Fig6.png width="90%">
</div>
</figure>

## Acknowledgement

+ Many thanks to the following repos: [SEAM](https://github.com/YudeWang/SEAM), [URN](https://github.com/xmed-lab/URN), and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Statement

+ Please cite our paper if our work is useful to your research. 

    ~~~
	@article{li2023one,
	  title={One model is enough: Toward multiclass weakly supervised remote sensing image semantic segmentation},
	  author={Li, Zhenshi and Zhang, Xueliang and Xiao, Pengfeng},
	  journal={IEEE Transactions on Geoscience and Remote Sensing},
	  year={2023},
	  publisher={IEEE}
	}
    ~~~

+ Any questions please contact [LZhenShi](https://github.com/LZhenShi) (Lzhenshi@outlook.com).
