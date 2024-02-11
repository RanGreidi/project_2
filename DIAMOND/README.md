

# DIAMOND - Dual-stage Interference-Aware Multi-flow Optimization of Network Data-signals
Raz Paul<sup>1</sup>, Kobi Cohen, Gil Kedar

<sup>1</sup><code>razpa@post.bgu.ac.il</code> (BGU)

[![arXiv](https://img.shields.io/badge/arXiv-2303.15544-b31b1b.svg)](https://arxiv.org/abs/2303.15544)

-----------------
This repository implements the DIAMOND algorithm, cosists of two stages:
- *Graph Routing via Reinforement Learning (GRRL)* implemented at `stage1_grrl`
- *Noisy Best-Response for optimal Routing Rates (NB3R)* implemented at `stage2_nb3r`

The overall DIAMOND algorithm is implemented at `diamond.py`


-----------------
# Installation
This implementation requires [Pytorch](https://pytorch.org/get-started/locally/).
We have tested DIAMOND using Python 3.9 and Pytorch 1.10 and 1.13.

Note that [Tensorflow](https://www.tensorflow.org/install/) is required for running comparison to other method. We have tested using Tensorflow-gpu 2.7.0.

1. create a [conda](https://docs.conda.io/en/latest/) environment

```commandline
conda create -n 'diamond' python=3.9.*
conda activate diamond
```

2. install requirements
```commandline
pip install -r requirements.txt
```


# Usage

## Train GRRL
- set up running hyperparameters at `stage1_grrl/config.py`
- train the model using running `run.py`
```commandline
python DIAMOND/stage1_grrl/run.py
```

## Evaluate DIAMOND
- set evaluation parameters at `test/inference_vs_competitors.py`
- run comparison
```commandline
python DIAMOND/test/inference_vs_competitors.py
```

## Reproduce paper results
```commandline
python DIAMOND/plots/plot_resutls.py
```
-----------------
code structure for *GRRL* inspired by [chaitjo](https://github.com/chaitjo/learning-tsp/tree/bb28f5795924d99f7e7d945695e884f8d1f7df45) 's repository
