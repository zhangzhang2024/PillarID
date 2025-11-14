<p align="center">

  <h1 align="center">Height3D: A Roadside Visual Framework Based on Height Prediction in Real 3-D Space</h1>
  
  </p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-TITS-00629b.svg)](https://ieeexplore.ieee.org/document/11005676)

<p align="center">
<img src="docs/assets/height3d_fig3.png" width="800" alt="" class="img-responsive">
</p>
<p align="center">
<img src="docs/assets/height3d_fig8.png" width="800" alt="" class="img-responsive">
</p>

# Overview

Height3D is a roadside visual perception framework based on height prediction in real 3D space. Height Prediction Block (HPB) with explicit height supervision is proposed to predict the height distribution of targets for roadside visual view transform. Also, Spatial Aware Block (SAB) is used to further extract spatial context information in BEV space and enhance fine-grained BEV features. The proposed method achieves the state-of-the-art in two large-scale real-world roadside visual perception benchmarks, DAIR-V2X-I and Rope3D. 


## Getting Started
### Installation

#### a. Clone this repository
```shell
https://github.com/zhangzhang2024/PillarID && cd PillarID
```
#### b. Install the environment

Following the install documents for [OpenPCDet](docs/INSTALL.md).

#### c. Prepare the datasets. 

Please follow the [Height3D](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to convert the DAIRV2X-I dataset to Kitti format and follow the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to convert the Kitti format to PCDet format, then rename it to dair-v2x-kitti-pcdet.

### Training

```shell
# single-gpu
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml  \
    --work_dir /mnt/tmpdata/pillar/pillarid/

# multi-gpus
CUDA_VISIBLE_DEVICES=0,1 bash tools/scripts/dist_train.sh 2 \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --work_dir /mnt/tmpdata/pillar/pillarid/
```

### Evaluation

```shell
# infer
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --ckpt /mnt/tmpdata/pillar/pillarid/default/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 --infer_time

# vis
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    --cfg_file tools/cfgs/dair_models/pillarid.yaml \
    --ckpt /mnt/tmpdata/pillar/pillarid/default/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 --vis_path ./vis_pillarid
```

<!-- ## Citation 
If you find this project useful in your research, please consider citing:

```
@inproceedings{chen2023voxenext,
  title={VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking},
  author={Yukang Chen and Jianhui Liu and Xiangyu Zhang and Xiaojuan Qi and Jiaya Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

``` -->

## Acknowledgement
-  This work is built upon the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt) and [Height3D](https://github.com/zhangzhang2024/Height3D). 



## License

This project is released under the [Apache 2.0 license](LICENSE).
