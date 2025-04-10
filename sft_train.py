# sft_train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import os


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("SFT Training Configuration:\n", OmegaConf.to_yaml(cfg))

    # Initialize wandb
    if cfg.training.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.device)
    # Set custom cache directory path

    # Needs to be imported after CUDA_VISIBLE_DEVICES
    from src.models.model_module import load_model_and_tokenizer
    from src.data.dataset import get_dataset
    from src.training.sft_module import run_sft_training

    # Load training, validation, and test datasets (assuming you have these available)
    train_dataset = get_dataset(cfg.dataset.name, split="train")
    val_dataset = get_dataset(
        cfg.dataset.name, split="test"
    )  # Make sure your dataset loader supports this split.
    test_dataset = None  # get_dataset(cfg.dataset.name, split="test")       # Likewise for the test set.

    # Load model and tokenizer from unsloth
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Run SFT training
    trainer = run_sft_training(
        model,
        tokenizer,
        train_dataset,
        cfg,
        val_dataset=val_dataset,
    )
    trainer.evaluate()


if __name__ == "__main__":
    main()
