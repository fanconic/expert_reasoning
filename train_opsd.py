import os

os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import get_dataset
from src.models.model_module import load_model_and_tokenizer
from src.training.opsd_module import run_opsd_training
from src.utils.utils import set_seed

wandb.login()


@hydra.main(config_path="configs", config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    print("OPSD Training Configuration:\n", OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.training.output_dir, "training_config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to: {config_save_path}")

    set_seed(cfg.seed)

    if cfg.training.report_to == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
        )

    no_system = getattr(cfg.dataset, "no_system", False)
    train_dataset = get_dataset(
        cfg.dataset.name,
        split="train",
        ratio=cfg.dataset.train_ratio,
        no_system=no_system,
    )
    val_dataset = get_dataset(
        cfg.dataset.name,
        split="test",
        ratio=cfg.dataset.val_ratio,
        no_system=no_system,
    )

    if "target" not in train_dataset.column_names:
        raise ValueError(
            "OPSD requires expert demonstrations in a `target` column. "
            "Use a distillation dataset such as gsm8k_kd, mmlu_kd, or medical_kd."
        )

    model, tokenizer = load_model_and_tokenizer(cfg)
    run_opsd_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        cfg=cfg,
        val_dataset=val_dataset,
    )


if __name__ == "__main__":
    main()
