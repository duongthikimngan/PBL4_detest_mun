"""Script that runs prediction process on ACNE04 images."""

import os
import hydra
from omegaconf import DictConfig
from transforms.acne_transforms import AcneTransformsTorch
from torch.utils.data import DataLoader
from dataset.acne_dataset import AcneDataset
import torch
from model.resnet50 import resnet50
import pandas as pd


@hydra.main(config_path="configs/predict", config_name="default")
def main(config: DictConfig):
    """Define main function."""

    # =======================
    # Model
    # =======================
    num_acne_cls = 13 if config.model_type == "model_ld_smoothing" else 4
    model = resnet50(num_acne_cls=num_acne_cls)

    checkpoint = torch.load(config.path_checkpoint, map_location=torch.device(config.device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    print(f"\n✅ Loaded checkpoint: {config.path_checkpoint}")
    print(f"📋 Model type: {config.model_type} | num_acne_cls: {num_acne_cls}")

    # =======================
    # Dataset & Dataloader
    # =======================
    dset_test = AcneDataset(
        config.path_images,
        config.path_images_metadata,
        transform=AcneTransformsTorch(train=False)
    )
    test_loader = DataLoader(
        dset_test,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )

    print(f"📂 Test samples: {len(dset_test)}")

    # =======================
    # Inference
    # =======================
    cls_test = torch.tensor([], dtype=torch.int32)
    cnt_test = torch.tensor([], dtype=torch.int32)

    with torch.no_grad():
        for step, (b_x, b_y, b_l) in enumerate(test_loader):
            b_x = b_x.to(config.device)
            print(f"  Step {step + 1}/{len(test_loader)}", end="\r")

            cls, cou, cou2cls = model(b_x)

            # Convert predictions back to Hayashi scale if LDS model
            if config.model_type == "model_ld_smoothing":
                cls = torch.stack(
                    (
                        torch.sum(cls[:, :1],    1),
                        torch.sum(cls[:, 1:4],   1),
                        torch.sum(cls[:, 4:10],  1),
                        torch.sum(cls[:, 10:],   1),
                    ),
                    1,
                )

            preds_cls = torch.argmax(0.5 * (cls + cou2cls), dim=1).cpu()
            preds_cnt = (torch.argmax(cou, dim=1) + torch.tensor(1)).cpu()

            cls_test = torch.cat((cls_test, preds_cls))
            cnt_test = torch.cat((cnt_test, preds_cnt))

    print(f"\n✅ Inference done! Total predictions: {len(cls_test)}")

    # =======================
    # Save CSV
    # =======================
    if config.save_preds:
        # Lấy tên file ảnh từ dataset
        image_names = [dset_test.data[i][0] for i in range(len(dset_test))]
        # Lấy ground truth label
        gt_labels   = [dset_test.data[i][1] for i in range(len(dset_test))]

        df = pd.DataFrame({
            "image":          image_names,
            "gt_class":       gt_labels,
            "severity_class": cls_test.numpy(),
            "num_acne":       cnt_test.numpy(),
        })

        # Tạo thư mục lưu
        out_dir = config.get("output_dir",
                  "/content/drive/MyDrive/PBL4_acne_lds/predictions")
        os.makedirs(out_dir, exist_ok=True)

        # Tên file theo fold và thiết lập
        fold_id   = config.get("fold_id", "unknown")
        setup     = config.model_type
        out_path  = os.path.join(out_dir, f"predictions_{setup}_fold_{fold_id}.csv")

        df.to_csv(out_path, index=False)
        print(f"💾 Saved: {out_path}")
        print(df["severity_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()