# AGC-Drive: A Large-Scale Dataset for Real-World Aerial-Ground Collaboration in Driving Scenarios

**AGC-Drive** is a large-scale, real-world dataset developed to advance autonomous driving research with aerial-ground collaboration. It enables multi-agent information sharing to overcome challenges such as occlusion and limited perception range, improving perception accuracy in complex driving environments.

While existing datasets often focus on vehicle-to-vehicle (V2V) or vehicle-to-infrastructure (V2I) collaboration, **AGC-Drive innovatively incorporates aerial views from unmanned aerial vehicles (UAVs)**. This integration provides dynamic, top-down perspectives that effectively reduce occlusion issues and allow monitoring of large-scale interactive scenarios.


---

## 📦 Dataset Overview

The dataset was collected using a collaborative sensing platform consisting of:

- **Two vehicles**, each equipped with **5 cameras and 1 LiDAR sensor**  
- **One UAV**, equipped with a **forward-facing camera and a LiDAR sensor**

It includes:

- **~80K LiDAR frames**  
- **~360K images**  
- **14 diverse real-world driving scenarios** (e.g., urban roundabouts, highway tunnels, on/off ramps)  
- **350 scenes**, each with approximately **100 frames**  
- Fully annotated **3D bounding boxes for 13 object categories**  
- **17% of frames** featuring dynamic interaction events: cut-ins, cut-outs, frequent lane changes

An open-source toolkit is also provided, featuring:

- 🗺️ Spatiotemporal alignment verification tools  
- 📊 Multi-agent collaborative visualization systems  
- 📝 Collaborative 3D annotation utilities  

---

## 📥 Download Dataset

We provide two download options:

- lidar_only:  https://pan.baidu.com/s/13r7msTs196CpG9huTyoRYQ?pwd=yen6
- png: Coming soon.
- radar: Processing

---

## 📝 Data Collection Method

Data was gathered across various urban and highway driving scenarios with hardware-level time synchronization and precise sensor calibration. It includes multi-agent LiDAR, multi-view RGB images, GPS/IMU data, and annotated 3D bounding boxes for collaborative perception applications.

---

## 📊 Benchmark Methods

We evaluate AGC-Drive with the following baseline models:

| Method             | Type                     | Description | configuration file| Model weights|
|:------------------|:-------------------------|:-----------------------------------------------------------|:-------------|:-------------|
| **Upper-bound**     | Early Fusion              | Shares raw point cloud data before feature extraction.      |[`early_fusion`](./OpenCOOD/opencood/hypes_yaml/pixor_early_fusion.yaml)|
| **Lower-bound**     | Late Fusion               | Independently detects and shares detection results.         |[`late_fusion`](./OpenCOOD/opencood/hypes_yaml/pixor_late_fusion.yaml)|
| **V2VNet**          | Intermediate Fusion       | Multi-agent detection via intermediate feature fusion.      |[`point_pillar_v2vnet`](./OpenCOOD/opencood/hypes_yaml/point_pillar_v2vnet.yaml)|
| **CoBEVT**          | Intermediate Fusion (BEV) | Sparse Transformer BEV fusion with FAX module.              |[`point_pillar_cobevt`](./OpenCOOD/opencood/hypes_yaml/point_pillar_cobevt.yaml)|
| **Where2comm**      | Communication-efficient   | Shares sparse, critical features guided by confidence maps. |[`point_pillar_where2comm`](./OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml)|
| **V2X-ViT**         | Transformer-based Fusion  | BEV feature fusion via attention mechanisms.                |[`point_pillar_v2xvit`](./OpenCOOD/opencood/hypes_yaml/point_pillar_v2xvit.yaml)|

---

## 🐍 Environment Setup
Our benchmark is built on the OpenCOOD framework. You can follow the [OpenCOOD installation guide](https://opencood.readthedocs.io/en/latest/md_files/installation.html) for setup.
Additionally, we provide a Conda environment file [`environment.yaml`](./OpenCOOD/environment.yml) exported from our development environment.  
You can create the environment by running the following command:

Recommended: **Python 3.7+**, **CUDA 11.7+**

### Install via Conda:
```bash
cd OpenCOOD
conda env create -f environment.yml
conda activate agcdrive
```

### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set with multi-gpu training togetger.
- You can enable UAV collaboration by setting the `uav_flag` key under `fusion/args` to `true` in the corresponding `config file`:
```yaml
fusion:
  args:
    uav_flag: true
```
To train on **multiple gpus**, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```


### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.
- `global_sort_detections`: whether to globally sort detections by confidence score. If set to True, it is the mainstream AP computing method, but would increase the tolerance for FP (False Positives). **OPV2V paper does not perform the global sort.** Please choose the consistent AP calculation method in your paper for fair comparison.

The evaluation results  will be dumped in the model directory. 

## 📆 TODO List
- [x] Paper released on arXiv.
- [x] Provide pretrained checkpoint.
- [x] Provide the lidar-only AGC-Drive dataset.
- [ ] Provide the complete AGC-Drive dataset.
- [ ] Support more of the latest methods.

## ☕ Citation
If you find our projects helpful to your research, please consider citing our paper:
```
@article{hou2025agc,
  title={AGC-Drive: A Large-Scale Dataset for Real-World Aerial-Ground Collaboration in Driving Scenarios},
  author={Hou, Yunhao and Zou, Bochao and Zhang, Min and Chen, Ran and Yang, Shangdong and Zhang, Yanmei and Zhuo, Junbao and Chen, Siheng and Chen, Jiansheng and Ma, Huimin*},
  journal={arXiv preprint arXiv:2506.16371},
  year={2025}
}
```
For any issues or further discussions, feel free to contact M202410661@xs.ustb.edu.com
## 📚 Supported Projects

The following key projects and papers are referenced and used as baselines in our benchmarks:

- **V2VNet**  
  Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, and Jiaqi Ma. Opv2v: An open benchmark dataset
  and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on
  Robotics and Automation (ICRA), page 2583–2589. IEEE Press, 2022.  
  [Paper](https://arxiv.org/abs/2008.07519)

- **CoBEVT**  
  Hao Xiang Wei Shao Bolei Zhou Jiaqi Ma Runsheng Xu, Zhengzhong Tu. Cobevt: Cooperative bird’s eye
view semantic segmentation with sparse transformers. In Conference on Robot Learning (CoRL), 2022.  
  [Paper](https://openreview.net/forum?id=PAFEQQtDf8s)

- **Where2comm**  
  Yue Hu, Shaoheng Fang, Zixing Lei, Yiqi Zhong, and Siheng Chen. Where2comm: Communication-
efficient collaborative perception via spatial confidence maps. Advances in neural information processing
systems, 35:4874–4886, 2022.  
  [Paper](https://openreview.net/forum?id=dLL4KXzKUpS)

- **V2X-ViT**  
  Runsheng Xu et al. V2x-vit: Vehicle-to-everything cooperative perception with vision transformer. In ECCV Proceedings, 2022.  
  [Paper](https://arxiv.org/abs/2203.10638)
