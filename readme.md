# Tabular Reasoning Models

## Abstract

## Getting Started


### Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Create and activate a virtual environment with required dependencies:
```bash
conda env create -f environment.yaml
conda activate unsloth_env
```

## Repository Setup

```
my_repo/
├── configs/
│   └── config.yaml
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── model_module.py
│   ├── rewards/
│   │   └── reward_functions.py
│   └── training/
│       └── grpo_module.py
├── train.py
└── inference.py
```


Run the GRPO training script:
```bash
pyhton train.py
```


Run the SFT training script:
```bash
python sft_train.py
```