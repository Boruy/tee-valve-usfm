"""Check USFM input, features, logits and loss before updating any weights."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from usdsgen.data.datasets import build_seg_dataset
from usdsgen.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("external/USFM"))
    parser.add_argument("--dataset", type=Path, default=Path("outputs/usfm_pilot"))
    parser.add_argument("--pretrained", type=Path, default=Path("checkpoints/USFM_latest.pth"))
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str((args.upstream / "configs").resolve()), version_base="1.2"):
        config = compose(config_name="train", overrides=[
            "experiment=task/Seg", "data=Seg/tee_valve", "model=Seg/SegVit",
            f'data.path.root="{args.dataset.resolve().as_posix()}"',
            f"data.img_size={args.img_size}",
            f'model.model_cfg.backbone.pretrained="{args.pretrained.resolve().as_posix()}"',
        ])
    logger = logging.getLogger("usfm_diagnostic")
    dataset, _, _ = build_seg_dataset(config, logger)
    item = dataset[args.index]
    image = item["image"].unsqueeze(0).cuda()
    label = item["mask"].unsqueeze(0).cuda().long()
    print(json.dumps({"input_finite": bool(torch.isfinite(image).all()),
                      "input_range": [float(image.min()), float(image.max())],
                      "label_values": label.unique().tolist(),
                      "image_path": item["img_path"]}, ensure_ascii=False), flush=True)
    model = build_model(config, logger).cuda().eval()
    invalid_parameters = [(name, int((~torch.isfinite(value)).sum()))
                          for name, value in model.named_parameters() if not torch.isfinite(value).all()]
    print(json.dumps({"invalid_parameters": invalid_parameters[:20],
                      "invalid_parameter_count": len(invalid_parameters)}), flush=True)
    with torch.no_grad():
        for precision in ("32", "16", "bf16"):
            enabled = precision != "32"
            dtype = torch.float16 if precision == "16" else torch.bfloat16
            try:
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                    features = model.backbone(image)
                    loss, logits, _ = model.decode_head.forward_with_loss(features, label)
                print(json.dumps({
                    "precision": precision,
                    "feature_finite": [bool(torch.isfinite(feature).all()) for feature in features],
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "loss_finite": bool(torch.isfinite(loss)),
                    "loss": float(loss),
                }), flush=True)
            except Exception as error:
                print(json.dumps({"precision": precision, "error": repr(error)}), flush=True)


if __name__ == "__main__":
    main()
