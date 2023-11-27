
# 云图外推实验

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## 任务列表
- [x] 配置实验环境，并设置EarthNet训练流程参数，使用`CNN`进行固定步长的输入和预测 。
- [x] 测试了 `RNN` 的效果

## Description

> 卫星云图预测是遥感和图像视觉交叉领域中出现的一个应用问题，旨在根据观测时刻历史资料生成未来临近时段内的云图，以尽可能同步地反映近实时地面观测网络上空的宏观天气变化。

该项目使用 [`conv-lstm`](https://arxiv.org/abs/1506.04214)来预测云图，结果预览如下：

|                                      |                                      |                                      |
| :---------------------------------: | :---------------------------------: | :---------------------------------: |
| 前1帧预测后18帧 | 前4帧预测后15帧 | 前8帧预测后11帧 |
| ![](./demo/pred/pred_pre_seq-1.gif) | ![](./demo/pred/pred_pre_seq-4.gif) | ![](./demo/pred/pred_pre_seq-8.gif) |
| 前12帧预测后11帧 |         使用前16帧预测后3帧          |  |
| ![](./demo/pred/pred_pre_seq-12.gif) | ![](./demo/pred/pred_pre_seq-16.gif) | ![](./demo/pred/pred_pre_seq-18.gif) |



## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r dev-requirments.txt
```



## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

**Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)**

```bash
python src/train.py experiment=h8cloud_rnn
# cpu debug
python src/train.py experiment=h8cloud_rnn logger=csv debug=default
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
