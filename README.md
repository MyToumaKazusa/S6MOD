# Enhancing Online Continual Learning with Plug-and-Play State Space Model and Class-Conditional Mixture of Discretization
The official repository of CVPR2025 paper "Enhancing Online Continual Learning with Plug-and-Play State Space Model and Class-Conditional Mixture of Discretization"
[![arXiv](https://img.shields.io/badge/arXiv-2312.00600-b31b1b.svg)](https://arxiv.org/abs/2412.18177)

![S6MOD Framework](figs/S6MOD.pdf)

## 📒 Updates

* **23 Mar:** We released the code of our paper.

## 🔨 Installation

- **We use the following hardware and software for our experiments:**

- Hardware: NVIDIA Tesla A100 GPUs
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## ➡️ Data Preparation

- **CIFAR-10/100**

Torchvision should be able to handle the CIFAR-10/100 dataset automatically. If not, please download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in the `data` folder.

- **TinyImageNet**

This codebase should be able to handle TinyImageNet dataset automatically and save them in the `data` folder. If not, please refer to [this github gist](https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54).

## 🚀 Training

- **Execute the provided scripts to start training:**
```shell
python main.py --data-root ./data --config ./config/CVPR25/cifar10/ER,c10,m500.yaml
(see more in cmd.txt)
```
- **Training with weight and bias sweep (Recommended)**

Weight and bias sweep is originally designed for hyperparameter search. However, it make the multiple runs much easier. Training can be done with W&B sweep more elegantly, for example:

```shell
wandb sweep sweeps/CVPR/ER,cifar10.yaml
```

Note that you need to set the dataset path in .yaml file by specify `--data-root-dir`. And run the sweep agent with:

```
wandb agent $sweepID
```

The hyperparameters after our hyperparameter search is located at `./sweeps/CVPR`.

## ✏️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{liu2024enhancing,
  title={Enhancing Online Continual Learning with Plug-and-Play State Space Model and Class-Conditional Mixture of Discretization},
  author={Liu, Sihao and Yang, Yibo and Li, Xiaojie and Clifton, David A and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2412.18177},
  year={2024}
}
```

## 👍 Acknowledgments

This codebase builds on [CCLDC](https://github.com/maorong-wang/CCL-DC). Thank you to all the contributors.