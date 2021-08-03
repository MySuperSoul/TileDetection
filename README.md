## 1. 团队介绍

队伍名称：米奇妙妙屋

队伍成员：黄亦非（浙江大学）、康天楠（浙江大学）、郑途（浙江大学）

预赛成绩：Rank 1
决赛成绩：Rank 4（极客奖）

## 2. 环境配置

Operating System: Ubuntu 16.04

Python version: 3.7

Pytorch version: 1.6.0

CUDA version: 10.1

我们项目使用的是conda虚拟环境，基于mmdetection检测框架实现我们的算法。在项目当中也提供了环境复现的`environment.yaml`文件。主要是安装torch环境，也可以`conda install pytorch==1.6.0 cudatoolkit==10.1.243 torchivision`命令进行安装，建议使用提供的yaml文件复现环境。

环境复现步骤：
```bash
# install miniconda3
bash Miniconda3-latest-Linux-x86_64.sh

# install the project environment from yaml
conda env create -f environment.yaml

# activate the environment and install some pip packages
conda activate mmdet
cd code

# install mmcv using prebuilt (recommend)
pip install https://download.openmmlab.com/mmcv/dist/1.2.5/torch1.6.0/cu101/mmcv_full-1.2.5%2Btorch1.6.0%2Bcu101-cp37-cp37m-manylinux1_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple
# or build from source
pip install mmcv-full==1.2.5

# install mmdetection
pip install -r requirements/build.txt -i https://mirrors.aliyun.com/pypi/simple
pip install -v -e . -i https://mirrors.aliyun.com/pypi/simple

# install some python packages
pip install ai-hub==0.1.9 flask ensemble_boxes antialiased_cnns albumentations -i https://mirrors.aliyun.com/pypi/simple
```

## 3. 解决方案

我们的解决方案在论坛的分享当中大部分都有提到了，可以参考[赛题攻略分享](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.6.d1a658efxcMRe9&postId=196094)。

我们队伍是决赛唯一一只没有使用模板的队伍，所以对于其他相关的没有提供模板的检测比赛也有很大的借鉴意义的。

- 数据预处理的代码都放在`TileDetection/utils`文件夹下，包括图片对齐，处理训练数据，对数据进行EDA等代码，不过由于线上训练的原因，统一默认数据挂载在`/tcdata`当中，可以自行修改对应的路径。
- 实现了MixUP、Mosaic、OnlineFusion等数据增强方式和SWA training，数据增强部分可以在`mmdet/datasets/pipelines/transforms.py`中查看代码。
- 实现了多种使用模板的方法，可以在`mmdet/models/detectors/two_stage.py`中查看，因为我们默认用的是CascadeRCNN，所以在TwoStageDetector里面改了（虽然我们用模板并没看到显著的提升，还是给大家做一个参考）。
- `run.sh`当中包括了完整的线上数据处理、模型训练和线上测试，可以自行查看代码～ 希望对更多其他检测任务能够有所帮助。
