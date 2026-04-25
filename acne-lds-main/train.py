
"""Train script that runs training process (5-fold safe + resume safe)."""

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

import importlib
import hydra
from omegaconf import DictConfig

from dataset.acne_data_module import AcneDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import seed_everything


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # =======================
    # Seed
    # =======================
    seed_everything(42)

    # =======================
    # Detect fold id from train file
    # Example: NNEW_trainval_4.txt → fold_id = 4
    # =======================
    fold_id = cfg.path.train_file.split("_")[-1].split(".")[0]
    print(f"\n🚀 Running Fold {fold_id}")

    # =======================
    # Data
    # =======================
    acne_module = AcneDataModule(
        cfg.path.train_file,
        cfg.path.val_file,
        cfg.path.data_path,
        cfg.train_val_params.batch_size,
        cfg.train_val_params.batch_size_val,
    )

    train_loader, val_loader = acne_module.create_loaders()

    # =======================
    # Model
    # =======================
    model_module = importlib.import_module(
        "model." + cfg.train_val_params.model_type
    )
    model = model_module.AcneModel(cfg)

    # =======================
    # Logger
    # =======================
    logger = None
    if cfg.trainer.logger == "wandb":
        logger = WandbLogger(
            project="acne-ldl",
            name=f"{cfg.train_val_params.model_type}_fold_{fold_id}",
            log_model=False,
        )

    # =======================
    # Checkpoint directory (separate per fold)
    # =======================
    base_ckpt_dir = "/content/drive/MyDrive/PBL4_acne_lds/checkpoints"
    ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold_id}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("📁 Checkpoint directory:", ckpt_dir)

    # =======================
    # Checkpoint callbacks
    # =======================
    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-epoch={epoch}-youden={val_you_index:.4f}",
        monitor="val_you_index",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    last_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
    )

    # =======================
    # Trainer
    # =======================
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=[best_ckpt, last_ckpt],
        log_every_n_steps=10,
    )

    # =======================
    # Resume if interrupted
    # =======================
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    if os.path.exists(last_ckpt_path):
        print("🔄 Resuming from:", last_ckpt_path)
        ckpt_path = last_ckpt_path
    else:
        print("🆕 Training from scratch")
        ckpt_path = None

    # =======================
    # Train
    # =======================
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=ckpt_path,
    )

    # =======================
    # Print best result
    # =======================
    print("\n✅ Best checkpoint path:", best_ckpt.best_model_path)
    print("✅ Best val_you_index:", best_ckpt.best_model_score)


if __name__ == "__main__":
    main()