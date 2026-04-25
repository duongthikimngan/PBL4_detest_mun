"""Base model with common methods."""

import numpy as np
import torch
import torch.nn as nn
from utils.genLD import genLD
from model.resnet50 import resnet50
from pytorch_lightning import LightningModule
import torchmetrics
from utils.utils import load_obj


class BaseModel(LightningModule):
    """
    Describe model's forward pass, metrics, losses, optimizers etc.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Number of acne classes from config
        self.num_classes = self.config.train_val_params.num_acne_cls

        # Model
        self.cnn = resnet50(num_acne_cls=self.num_classes)

        # Loss
        self.kl_loss = nn.KLDivLoss()

        # =========================
        # Metrics (TorchMetrics >=1.x compatible)
        # =========================
        NUM_GRADING_CLASSES = 4
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=NUM_GRADING_CLASSES,
        )
        NUM_GRADING_CLASSES = 4
        self.prec = torchmetrics.Precision(
            task="multiclass",
            num_classes=NUM_GRADING_CLASSES,
            average="macro"
        )
        NUM_GRADING_CLASSES = 4
        self.specificity = torchmetrics.Specificity(
            task="multiclass",
            num_classes=NUM_GRADING_CLASSES,
            average="macro"
        )
        NUM_GRADING_CLASSES = 4
        self.sensitivity = torchmetrics.Recall(
            task="multiclass",
            num_classes=NUM_GRADING_CLASSES,
            average="macro"
        )
        NUM_GRADING_CLASSES = 4
        self.mcc_cls = torchmetrics.MatthewsCorrCoef(
            task="multiclass",
            num_classes=NUM_GRADING_CLASSES,
        )

        # Regression metrics
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError(squared=False)

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):
        return self.cnn(x)

    # =====================================================
    # Training epoch end
    # =====================================================

    def training_epoch_end(self, training_outs):

        preds_cls = torch.cat([outs["preds_cls"] for outs in training_outs])
        preds_cnt = torch.cat([outs["preds_cnt"] for outs in training_outs])
        b_y = torch.cat([outs["b_y"] for outs in training_outs])
        b_l = torch.cat([outs["b_l"] for outs in training_outs])

        # Compute metrics
        train_sens = self.sensitivity(preds_cls, b_y)
        train_spec = self.specificity(preds_cls, b_y)

        metrics = {
            "train_accuracy": self.accuracy(preds_cls, b_y),
            "train_sensitivity": train_sens,
            "train_prec": self.prec(preds_cls, b_y),
            "train_specificity": train_spec,
            "train_you_index": train_sens + train_spec - 1,
            "train_mae": self.mae(preds_cnt, b_l),
            "train_mse": self.mse(preds_cnt, b_l),
            "train_mcc_class": self.mcc_cls(preds_cls, b_y),
        }

        self.log_dict(metrics, prog_bar=True)

    # =====================================================
    # Validation epoch end
    # =====================================================

    def validation_epoch_end(self, val_outs):

        preds_cls = torch.cat([outs["preds_cls"] for outs in val_outs])
        preds_cnt = torch.cat([outs["preds_cnt"] for outs in val_outs])
        b_y = torch.cat([outs["b_y"] for outs in val_outs])
        b_l = torch.cat([outs["b_l"] for outs in val_outs])

        val_sens = self.sensitivity(preds_cls, b_y)
        val_spec = self.specificity(preds_cls, b_y)

        metrics = {
            "val_accuracy": self.accuracy(preds_cls, b_y),
            "val_sensitivity": val_sens,
            "val_prec": self.prec(preds_cls, b_y),
            "val_specificity": val_spec,
            "val_you_index": val_sens + val_spec - 1,
            "val_mae": self.mae(preds_cnt, b_l),
            "val_mse": self.mse(preds_cnt, b_l),
            "val_mcc_class": self.mcc_cls(preds_cls, b_y),
        }

        self.log_dict(metrics, prog_bar=True)

    # =====================================================
    # Optimizer
    # =====================================================

    def configure_optimizers(self):

        optimizer = load_obj(self.config.optimizers.optim_name)(
            self.cnn.parameters(),
            **self.config.optimizers.params
        )

        scheduler = load_obj(self.config.scheduler.scheduler_name)(
            optimizer,
            **self.config.scheduler.params
        )

        return [optimizer], [scheduler]

    # =====================================================
    # Label Distribution Generation
    # =====================================================

    def generate_ld(self, b_l):

        # Ép về Python list thuần túy, tránh mọi vấn đề CUDA/autograd
        if hasattr(b_l, 'is_cuda') or hasattr(b_l, 'device'):
            b_l_list = [float(x) for x in b_l.reshape(-1)]
        else:
            b_l_list = list(b_l)
        b_l_np = np.array(b_l_list, dtype=np.float32)

        ld = genLD(
            b_l_np,
            self.config.train_val_params.sigma,
            "klloss",
            65
        )

        ld_13 = np.vstack(
            (
                np.sum(ld[:, :5], 1),
                np.sum(ld[:, 5:10], 1),
                np.sum(ld[:, 10:15], 1),
                np.sum(ld[:, 15:20], 1),
                np.sum(ld[:, 20:25], 1),
                np.sum(ld[:, 25:30], 1),
                np.sum(ld[:, 30:35], 1),
                np.sum(ld[:, 35:40], 1),
                np.sum(ld[:, 40:45], 1),
                np.sum(ld[:, 45:50], 1),
                np.sum(ld[:, 50:55], 1),
                np.sum(ld[:, 55:60], 1),
                np.sum(ld[:, 60:], 1),
            )
        ).transpose()

        ld_4 = np.vstack(
            (
                np.sum(ld[:, :5], 1),
                np.sum(ld[:, 5:20], 1),
                np.sum(ld[:, 20:50], 1),
                np.sum(ld[:, 50:], 1),
            )
        ).transpose()

        ld = torch.from_numpy(ld).float()
        ld_13 = torch.from_numpy(ld_13).float()
        ld_4 = torch.from_numpy(ld_4).float()

        if torch.cuda.is_available():
            ld = ld.cuda()
            ld_13 = ld_13.cuda()
            ld_4 = ld_4.cuda()

        return ld, ld_13, ld_4