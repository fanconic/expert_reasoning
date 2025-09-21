# Knowledge Distillation via Expert Reasoning

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
bash runner_scripts/{$node}_run_gpu_node.sh pyhton train.py
```


Run the SFT training script:
```bash
bash runner_scripts/{$node}_run_gpu_node.sh pyhton sft_train.py
```


# Dataset Builders

These scripts create two Hugging Face `DatasetDict`s and save them to disk.
```bash
python create_countdown_dataset.py \
  --outdir ../data/countdown \
  --test_size 400 \
  --val_size 400 \
  --seed 42
  --max_len 2048
```

```bash
python create_medical_o1_dataset.py \
  --outdir ../../data/medical_o1 \
  --test_size 2000 \
  --val_size 2000 \
  --seed 42
  --max_len 512
```

Run training and evaluation on one single GPU:

Llama-3.1-8B
```bash
bash runner_scripts/0_run_gpu_node.sh sft_train.py --config-path=configs/llama --config-name=sft_8B_config_train
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=sft_8B_config_eval

bash runner_scripts/0_run_gpu_node.sh train.py --config-path=configs/llama --config-name=grpo_8B_config_train
bash runner_scripts/0_run_gpu_node.sh evaluate.py --config-path=configs/llama --config-name=grpo_8B_config_eval
```
