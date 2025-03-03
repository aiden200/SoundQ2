# Installation steps for SoundQ2

## Repo setup

We recommend using conda, this eases some dependencies with cuda for running all the available submodules in this repo.

```bash
conda create --name soundq2 python=3.10 -y
conda activate soundq2
```

## Start repo submodules

```bash
git submodule sync --recursive
git submodule update --init --recursive
```


### Requirements
- Linux or macOS with Python ≥ 3.10
- `PyTorch ≥ 2.3.1`
- `torch >= 2.3.1`
- `torchvision>=0.18.1`
- Tested on `cuda12.4`

Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 

```
pip3 install torch torchvision torchaudio
```


### Install Grounded SAM2 packages

Then, follow the [Grounded-SAM-2 install instructions](https://github.com/IDEA-Research/Grounded-SAM-2/tree/dd4c5141b75e4838dd486c64f773c43b4db3a07b?tab=readme-ov-file#installation). The official repo's installation didn't work for us. If you have the same problem, you can look at our supplimentary [Grounded-SAM-2 installation guide](GD_SAM2_INSTALL.md) for reference.


### Install soundq2

```bash
pip install -e .
```


