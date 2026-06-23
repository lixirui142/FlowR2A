# Data and Environment Preparation

This guide installs the FlowR2A dependencies and downloads the data and models needed to run evaluation. FlowR2A is built on the [NAVSIM](https://github.com/autonomousvision/navsim) devkit, so the data layout and environment variables follow the NAVSIM convention.

## 1. Clone the repository
```bash
git clone https://github.com/lixirui142/FlowR2A.git
cd FlowR2A
```

## 2. Download the data
Evaluation runs on the NAVSIM **navtest** split, which requires the nuPlan maps and the OpenScene `test` split. Download scripts are provided in `download/`.

**NOTE: Please check the [nuPlan LICENSE](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download
bash download_maps.sh
bash download_test.sh
cd ..
```

Organize the downloaded data exactly as NAVSIM specifies (see the [NAVSIM install doc](https://github.com/autonomousvision/navsim/blob/main/docs/install.md#getting-started-)):
```
<WORKSPACE>/navsim_workspace
├── FlowR2A          # this repo (the devkit)
├── exp              # experiment / cache outputs
└── dataset
    ├── maps
    ├── navsim_logs
    │   └── test
    └── sensor_blobs
        └── test
```

Set the required environment variables (add to your `~/.bashrc`). Replace `<WORKSPACE>` with any directory you prefer:
```bash
export WORKSPACE="<WORKSPACE>"   # e.g. /path/to/your/workspace
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$WORKSPACE/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$WORKSPACE/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$WORKSPACE/navsim_workspace/FlowR2A"
export OPENSCENE_DATA_ROOT="$WORKSPACE/navsim_workspace/dataset"
```

## 3. Install the environment
Create the conda environment and install FlowR2A in editable mode:
```bash
conda env create --name flowr2a -f environment.yml
conda activate flowr2a
pip install -e .
```

> **Reminder:** FlowR2A is developed and tested with `torch==2.1.0` / `torchvision==0.16.0` (pinned in `requirements.txt`). If this does not match your CUDA version, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

Next: [Evaluation](evaluation.md).
