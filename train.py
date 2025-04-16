import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os


# Example evaluation metric function
def simple_accuracy_metric(ground_truth, prediction):
    # For example, compare if the extracted answer matches the ground truth exactly
    return 1.0 if ground_truth.strip() == prediction.strip() else 0.0


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    if cfg.training.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.device)

    # Needs to be imported after CUDA_VISIBLE_DEVICES
    from src.data.dataset import get_dataset
    from src.models.model_module import load_model_and_tokenizer
    from src.rewards.reward_functions import get_reward_functions
    from src.training.grpo_module import run_grpo_training

    # Load training, validation, and test datasets (assuming you have these available)
    train_dataset = get_dataset(cfg.dataset.name, split="train")
    val_dataset = get_dataset(cfg.dataset.name, split="test") # Make sure your dataset loader supports this split.
    test_dataset = None  # get_dataset(cfg.dataset.name, split="test")       # Likewise for the test set.

    # Load model and tokenizer from unsloth
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Get reward functions
    reward_funcs = get_reward_functions()

    # Run GRPO training with periodic evaluation
    run_grpo_training(
        model,
        tokenizer,
        train_dataset,
        reward_funcs,
        cfg,
        val_dataset=val_dataset,
    )


if __name__ == "__main__":
    main()
