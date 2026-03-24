This repository is the official implementation of [Guidestar-Free Adaptive Optics with Asymmetric Apertures](https://weiyunjiang.com/guidestar-free-ao/).

> **[ACM Transactions on Graphics 2026] Guidestar-Free Adaptive Optics with Asymmetric Apertures** <br>
> [Weiyun Jiang](https://weiyunjiang.com/), [Haiyun Guo](https://haiyunguo7.github.io/), [Christopher A. Metzler](https://www.cs.umd.edu/~metzler/), [Ashok Veeraraghavan](https://profiles.rice.edu/faculty/ashok-veeraraghavan)<br>

[![Arxiv](https://img.shields.io/badge/arXiv-2509.21309-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2602.07029)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://weiyunjiang.com/guidestar-free-ao/)

![Teaser](figs/teaser_AO_single_col_web_new.png)
---

## 💻 System & Hardware Requirements

This code was developed and tested on the following configuration:

* **Operating System:** Ubuntu 20.04.6 LTS
* **CUDA Version:** 11.8 (as built with PyTorch)
* **GPU:** NVIDIA A100 80 GB
* **Python:** 3.8
  
---

## 🛠 Quick Start

### Conda Environment
We recommend using **Conda** to manage dependencies and ensure CUDA compatibility.

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate gsf-ao
```

---

## 📊 Data Preparation
### 1. Download the Dataset
First, download the **Places2** dataset from the [official Places2 website](http://places2.csail.mit.edu/download.html).

### 2. Directory Structure
Organize the data under the `data/` directory. The training script expects the following pathing:

```text
guidestar-free-ao/
└── data/
    └── places2/
        └── places365_standard/
            ├── train/                
            │   ├── airfield/
            │   └── ...
            ├── val/                  
            │   ├── airfield/
            │   └── ...
            ├── train.txt
            └── val.txt
```
### 3. Generate Paired PSF-Phase Data 
Finally, generate the synthetic paired PSF-Phase dataset using the following commands:
```bash
bash ./experiments/guidestar/guidestar_data.sh
bash ./experiments/guidestar_free/guidestar_free_data_step1.sh
bash ./experiments/guidestar_free/guidestar_free_data_step2.sh
```
---
## ⚖️ Pretrained Weights

To facilitate quick baseline comparisons, we provide pretrained model weights for the following configurations. These models were trained on the Places2 dataset.

| Experiment | Model Type | Application |
| :--- | :--- | :--- |
| **Guidestar-Free** | Phase U-Net | Wavefront estimation / phase retrieval |
| **Guidestar-Free** | PSF U-Net | Kernel estimation for guidestar-free wavefront shaping |
| **Guidestar** | Phase U-Net | Wavefront estimation with a guide star |

You can download the model checkpoints from our Google Drive:
[📥 Download Pretrained Weights](https://drive.google.com/file/d/1e2GxutCBGbDSLIHtMcvRMSlsKAqJVtYH/view?usp=sharing)

---

## 🚀 Running Experiments

All experiment workflows are organized into subdirectories within the `experiments/` folder.

### 1. Guidestar Experiments

* **Estimated Runtime:** ~1 day
* **Command:**
```bash
bash ./experiments/guidestar/guidestar.sh
```

---

### 2. Guidestar-Free Experiments 
This pipeline handles the guidestar-free workflow, which includes kernel estimation followed by joint optimization.

* **Estimated Runtime:** ~7 days total (~5 days for Step 1, ~2 days for Step 2)
* **Command:**
```bash
bash ./experiments/guidestar_free/guidestar_free.sh
```


## 📂 Repository Structure

The project is organized as follows:

* `data/`: Directory for storing raw datasets and processed patches.
* `experiments/`: Bash scripts for running the full pipelines.
    * `guidestar/`: Scripts for guidestar experiments.
    * `guidestar_free/`: Scripts for the guidestar-free experiments.
* `psfphase_simulator.py`: GPU accelerated simulator for generating phase-psf pairs.
* `dataio.py`: Utilities for data loading and input/output operations.
* `utils.py`: Shared helper functions.
* `train_*.py`: Primary Python training scripts called by the bash scripts.

## Citation

If you find this work useful, please consider citing:
```bibtex
@article{jiang2026guidestar,
    title={Guidestar-Free Adaptive Optics with Asymmetric Apertures},
    author={Jiang, Weiyun and Guo, Haiyun and Metzler, Christopher A. and Veeraraghavan, Ashok},
    journal={ACM Transactions on Graphics (TOG)},
    year={2026}
}
```

## License
   This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
