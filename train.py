import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
wandb.login()
from src.models.model_module import load_model_and_tokenizer
from src.utils.utils import set_seed
from src.data.dataset import get_dataset
from src.rewards.reward_functions import get_reward_functions
from src.training.grpo_module import run_grpo_training


@hydra.main(config_path="configs", config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    if cfg.training.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
        )

    # Load training, validation, and test datasets (assuming you have these available)
    no_system = getattr(cfg.dataset, "no_system", False)
    train_dataset = get_dataset(
        cfg.dataset.name, split="train", ratio=cfg.dataset.train_ratio, no_system=no_system
    )
    val_dataset = get_dataset(
        cfg.dataset.name, split="test", ratio=cfg.dataset.val_ratio, no_system=no_system
    )  # Make sure your dataset loader supports this split.
    test_dataset = None  # get_dataset(cfg.dataset.name, split="test")       # Likewise for the test set.

    # Load model and tokenizer from unsloth
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Get reward functions
    reward_funcs, reward_processing_classes = get_reward_functions(cfg.dataset.name)

    # Run GRPO training with periodic evaluation
    run_grpo_training(
        model,
        tokenizer,
        train_dataset,
        reward_funcs,
        cfg,
        val_dataset=val_dataset,
        reward_processing_classes=reward_processing_classes
    )


if __name__ == "__main__":
    main()
