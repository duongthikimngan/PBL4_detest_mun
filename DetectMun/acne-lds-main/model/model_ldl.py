"""Model for baseline (Wu et al.) - LDL without smoothing."""

import torch
from model.base_model import BaseModel


class AcneModel(BaseModel):
    """
    Baseline model (Wu et al. [12]) - Label Distribution Learning
    WITHOUT label distribution smoothing.
    
    This is the baseline approach that uses standard Gaussian label
    distributions for lesion counting, without the smoothing modifications
    proposed in this paper.
    """

    def __init__(self, config):
        super().__init__(config=config)
        # No label smoothing object for baseline

    # =========================
    # Training step
    # =========================
    def training_step(self, batch, batch_idx):
        b_x, b_y, b_l = batch

        # Forward
        cls, cou, cou2cls = self.cnn(b_x)
        device = cls.device

        # Generate label distributions (WITHOUT smoothing - baseline)
        ld, ld_13, ld_4 = self.generate_ld((b_l - 1).detach().cpu())
        ld = ld.to(device)
        ld_13 = ld_13.to(device)
        ld_4 = ld_4.to(device)

        # Avoid log(0)
        cls = torch.clamp(cls, min=1e-7)
        cou = torch.clamp(cou, min=1e-7)
        cou2cls = torch.clamp(cou2cls, min=1e-7)

        # Losses
        loss_cls = self.kl_loss(torch.log(cls), ld_4) * 4.0
        loss_cou = self.kl_loss(torch.log(cou), ld) * 65.0  # Use ld (not smoothed)
        loss_cls_cou = self.kl_loss(torch.log(cou2cls), ld_4) * 4.0

        lam = self.config.train_val_params.lam
        loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

        # Convert predictions back to Hayashi scale
        prob_cls_4 = torch.stack(
            (
                torch.sum(cls[:, :1], 1),
                torch.sum(cls[:, 1:4], 1),
                torch.sum(cls[:, 4:10], 1),
                torch.sum(cls[:, 10:], 1),
            ),
            dim=1,
        )

        preds_cls = torch.argmax(0.5 * (prob_cls_4 + cou2cls), dim=1)
        preds_cnt = torch.argmax(cou, dim=1) + torch.ones(1, device=device)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss_cls", loss_cls, on_epoch=True)
        self.log("train_loss_cou", loss_cou, on_epoch=True)
        self.log("train_loss_cou2cls", loss_cls_cou, on_epoch=True)

        return {
            "loss": loss,
            "preds_cls": preds_cls,
            "preds_cnt": preds_cnt,
            "b_y": b_y,
            "b_l": b_l,
        }

    # =========================
    # Validation step
    # =========================
    def validation_step(self, batch, batch_idx):
        import traceback
        b_x, b_y, b_l = batch

        print(f"DEBUG val b_l: type={type(b_l)}, dtype={b_l.dtype}, device={b_l.device}, shape={b_l.shape}, values={b_l[:3]}")

        cls, cou, cou2cls = self.cnn(b_x)
        device = cls.device

        try:
            ld, ld_13, ld_4 = self.generate_ld((b_l - 1).detach().cpu())
        except Exception as e:
            print(f"REAL ERROR in generate_ld: {e}")
            traceback.print_exc()
            raise
        ld = ld.to(device)
        ld_13 = ld_13.to(device)
        ld_4 = ld_4.to(device)

        cls = torch.clamp(cls, min=1e-7)
        cou = torch.clamp(cou, min=1e-7)
        cou2cls = torch.clamp(cou2cls, min=1e-7)

        loss_cls = self.kl_loss(torch.log(cls), ld_4) * 4.0
        loss_cou = self.kl_loss(torch.log(cou), ld) * 65.0  # Use ld (not smoothed)
        loss_cls_cou = self.kl_loss(torch.log(cou2cls), ld_4) * 4.0

        lam = self.config.train_val_params.lam
        loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

        prob_cls_4 = torch.stack(
            (
                torch.sum(cls[:, :1], 1),
                torch.sum(cls[:, 1:4], 1),
                torch.sum(cls[:, 4:10], 1),
                torch.sum(cls[:, 10:], 1),
            ),
            dim=1,
        )

        preds_cls = torch.argmax(0.5 * (prob_cls_4 + cou2cls), dim=1)
        preds_cnt = torch.argmax(cou, dim=1) + torch.ones(1, device=device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss_cls", loss_cls, on_epoch=True)
        self.log("val_loss_cou", loss_cou, on_epoch=True)
        self.log("val_loss_cou2cls", loss_cls_cou, on_epoch=True)

        return {
            "loss": loss,
            "preds_cls": preds_cls,
            "preds_cnt": preds_cnt,
            "b_y": b_y,
            "b_l": b_l,
        }