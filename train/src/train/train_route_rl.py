# Use the modified diffusers & peft library
import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../icedit"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np
from datasets import load_dataset

from .data import (
    EditDataset,
    OminiDataset,
    EditDataset_with_Omini,
    EditDataset_AnyEdit
)
from .model import OminiModel
from .callbacks import TrainingCallback, TimingCallback
from .rl_utils import GRPOConfig, default_reward_fn


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def build_reward_fn(reward_cfg: dict):
    """
    Return a RewardFn based on config.  Extend this function to add custom
    rewards without touching anything else.

    reward:
      type: "mse"          # default: negative MSE (higher = better reconstruction)
      # type: "custom"     # placeholder — add your own branch here
    """
    rtype = reward_cfg.get("type", "mse")
    if rtype == "mse":
        return default_reward_fn
    # ── add custom reward functions below ────────────────────────────────────
    # elif rtype == "lpips":
    #     from .rewards import lpips_reward_fn
    #     return lpips_reward_fn
    # elif rtype == "clip":
    #     from .rewards import clip_reward_fn
    #     return clip_reward_fn
    else:
        raise ValueError(f"Unknown reward type: {rtype!r}")


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    if "use_offset_noise" not in config:
        config["use_offset_noise"] = False

    # Initialize dataset
    dataset_cfg = training_config["dataset"]
    dtype = dataset_cfg["type"]

    if dtype == "edit":
        dataset = EditDataset(
            load_dataset("osunlp/MagicBrush"),
            condition_size=dataset_cfg["condition_size"],
            target_size=dataset_cfg["target_size"],
            drop_text_prob=dataset_cfg["drop_text_prob"],
        )
    elif dtype == "omini":
        dataset = OminiDataset(
            json_path=dataset_cfg["json_path"],
            root_path=dataset_cfg["root_path"],
            condition_size=dataset_cfg["condition_size"],
            target_size=dataset_cfg["target_size"],
            drop_text_prob=dataset_cfg["drop_text_prob"],
            specific_task=dataset_cfg.get("specific_task"),
        )
    elif dtype == "edit_with_omini":
        omni = load_dataset("parquet", data_files=os.path.abspath(dataset_cfg["path"]), split="train")
        magic = load_dataset("osunlp/MagicBrush")
        dataset = EditDataset_with_Omini(
            magic, omni,
            condition_size=dataset_cfg["condition_size"],
            target_size=dataset_cfg["target_size"],
            drop_text_prob=dataset_cfg["drop_text_prob"],
        )
    elif dtype == "any_edit":
        dataset = EditDataset_AnyEdit(
            path=dataset_cfg["path"],
            condition_size=dataset_cfg["condition_size"],
            target_size=dataset_cfg["target_size"],
            drop_text_prob=dataset_cfg["drop_text_prob"],
            specific_task=dataset_cfg["specific_task"],
        )
    else:
        raise ValueError(f"Unknown dataset type: {dtype!r}")

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_fill_id=config["flux_path"],
        lora_path=config["lora_path"],
        lora_config=training_config.get("lora_config"),
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Wire up GRPO config — this enables RL routing in training_step
    rl_cfg = training_config.get("rl", {})
    reward_fn = build_reward_fn(rl_cfg.get("reward", {"type": "mse"}))
    trainable_model.rl_config = GRPOConfig(
        num_rollouts=rl_cfg.get("num_rollouts", 4),
        rl_coeff=rl_cfg.get("rl_coeff", 0.05),
        supervised_coeff=rl_cfg.get("supervised_coeff", 1.0),
        stochastic_ratio=rl_cfg.get("stochastic_ratio", 1.0),
        reward_fn=reward_fn,
    )
    if is_main_process:
        print(
            f"[RL] GRPO enabled — G={trainable_model.rl_config.num_rollouts}, "
            f"rl_coeff={trainable_model.rl_config.rl_coeff}, "
            f"reward={rl_cfg.get('reward', {}).get('type', 'mse')}"
        )

    # Callbacks
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else [TimingCallback(print_every_n_steps=10)]
    )

    # Trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        strategy='ddp_find_unused_parameters_true'
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
