import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
wandb.login()
from src.models.model_module_trl import irl_load_model_and_tokenizer_trl
#from src.models.model_module import irl_load_model_and_tokenizer
from src.data.dataset import get_dataset
from src.training.irl_module import run_irl_training
from src.utils.utils import set_seed
from src.rewards.reward_functions import get_reward_functions


@hydra.main(config_path="configs", config_name="config_irl_train", version_base="1.3")
def main(cfg: DictConfig):
    print("IRL Training Configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    # Initialize wandb
    if cfg.training.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
        )

    # Load training, validation, and test datasets (assuming you have these available)
    train_dataset = get_dataset(
        cfg.dataset.name, split="train", ratio=cfg.dataset.train_ratio
    )
    val_dataset = get_dataset(
        cfg.dataset.name, split="test", ratio=cfg.dataset.val_ratio
    )  # Make sure your dataset loader supports this split.  # Make sure your dataset loader supports this split.
    test_dataset = None  # get_dataset(cfg.dataset.name, split="test")       # Likewise for the test set.

    # Load model and tokenizer from unsloth
    policy_model, reward_model, policy_tokenizer, reward_tokenizer = (
        irl_load_model_and_tokenizer_trl(cfg)
    )

    # Get reward functions
    reward_funcs, reward_processing_classes = get_reward_functions(cfg.dataset.name)

    # Run SFT training
    trainer = run_irl_training(
        policy_model=policy_model,
        reward_model=reward_model,
        policy_tokenizer=policy_tokenizer,
        reward_tokenizer=reward_tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        cfg=cfg,
        val_dataset=val_dataset,
        reward_processing_classes=reward_processing_classes
    )


if __name__ == "__main__":
    main()
