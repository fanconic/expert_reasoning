import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
wandb.login()
import os
#os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"
#os.environ["VLLM_USE_V1"] = "0"


@hydra.main(config_path="configs", config_name="config_irl_train", version_base="1.3")
def main(cfg: DictConfig):
    print("IRL Training Configuration:\n", OmegaConf.to_yaml(cfg))
    if cfg.unsloth:
        from src.models.model_module import irl_load_model_and_tokenizer
        model_tokenizer_loader = irl_load_model_and_tokenizer
    else:
        from src.models.model_module_trl import irl_load_model_and_tokenizer_trl
        model_tokenizer_loader = irl_load_model_and_tokenizer_trl
        
    from src.data.dataset import get_dataset
    from src.training.irl_module import run_irl_training
    from src.utils.utils import set_seed
    from src.rewards.reward_functions import get_reward_functions

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
    no_system = getattr(cfg.dataset, "no_system", False)
    from src.rewards.perturbations import PERTURB_FN_MAP
    
    train_dataset = get_dataset(
        cfg.dataset.name, split="train", ratio=cfg.dataset.train_ratio, no_system=no_system,
        expert_error_rate = getattr(cfg.dataset, "expert_error_rate", 0.0),
        neg_perturb_fns = [PERTURB_FN_MAP[name] for name in cfg.model.neg_perturb_fns],
        num_neg_perturbations_per_expert = cfg.model.num_neg_perturbations_per_expert
    )
    
    val_dataset = get_dataset(
        cfg.dataset.name, split="test", ratio=cfg.dataset.val_ratio, no_system=no_system
    )  # Make sure your dataset loader supports this split.  # Make sure your dataset loader supports this split.
    test_dataset = None  # get_dataset(cfg.dataset.name, split="test")       # Likewise for the test set.

    # Load model and tokenizer from unsloth
    pretrained = False
    frozen_discriminator = getattr(cfg.training, "freeze_discriminator", False)
    discriminator_path = getattr(cfg.model, "frozen_discriminator_path", None)
    policy_model, reward_model, policy_tokenizer, reward_tokenizer = (
        model_tokenizer_loader(cfg, pretrained=pretrained, frozen_discriminator=frozen_discriminator, discriminator_path=discriminator_path)
    )

    # Get reward functions
    reward_funcs, reward_processing_classes = get_reward_functions(cfg.dataset.name)

    # Run AIRL training
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
