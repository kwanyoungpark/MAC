# Scaling Offline Model-Based RL with Action chunking

This repository contains the official implementation of [Scalable Offline Model-Based RL with Action chunking](TODO).

If you use this code for your research, please consider citing our paper:
```
@article{park2025_MAC,
  title={Scalable Offline Model-Based RL with Action Chunking},
  author={Kwanyoung Park, Seohong Park, Youngwoon Lee, Sergey Levine},
  journal={arXiv Preprint},
  year={2025}
}
```

## Overview
This codebase contains implementations of 1) model-free methods, 2) model-based methods, 3) MAC, and ablated version of MAC for ablation studies. Specifically: 

```
# Baselines (Model-free)
from agents.gciql import GCIQLAgent         # GCIQL
from agents.ngcsacbc import NGCSACBCAgent   # n-step GCSAC+BC
from agents.sharsa import SHARSAAgent       # SHARSA

# Baselines (Model-based)
from agents.fmpc import FMPCAgent           # FMPC
from agents.leq import LEQAgent             # LEQ
from agents.mopo import MOPOAgent           # MOPO
from agents.mobile import MOBILEAgent       # MOBILE

# Our method (MAC)
from agents.mac import MACAgent             # MAC
from agents.mbrs_ac import ACMBRSAgent      # MAC (Gau)
from agents.mbfql import MBFQLAgent         # MAC (FQL)
from agents.model_ac import ACModelAgent    # Model inaccuracy analysis
```

## Installation

Please install the libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Datasets

For downloading the datasets, please follow the instruction of [Horizon Reduction Makes RL Scalable](https://github.com/seohongpark/horizon-reduction).

## Example training scripts 

For MVE-based MBRL algorithms (MAC, LEQ, FMPC) and model-free RL algorithms (SHARSA, GCIQL, n-step GCSAC+BC):

```
# MAC in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mac.py

# SHARSA in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/sharsa.py
```

For MBPO-based algorithms (MOPO, MOBILE):

```
# MOPO in puzzle-4x5-play-oraclerep-v0 (100M)
python main_mbpo.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mopo.py
```

## Acknowledgement

This codebase is built on top of [Horizon Reduction makes RL scalable](https://github.com/seohongpark/horizon-reduction)'s codebase.
