"""Launch the official USFM SegVit trainer with this project's TEE dataset config."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _output_dir import replace_existing_output


PROJECT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT / "configs/usfm_tee_valve.yaml"
PARTIAL_ENTRY = PROJECT / "code/scripts/train_usfm_partial.py"


def build_command(args: argparse.Namespace) -> list[str]:
    upstream = args.upstream.resolve()
    dataset = args.dataset.resolve()
    if not (upstream / "main.py").is_file():
        raise FileNotFoundError(f"Official USFM source missing: {upstream / 'main.py'}")
    if (dataset / "_INCOMPLETE").exists():
        raise ValueError(f"Dataset export is incomplete; rerun prepare_usfm_dataset.py with --output {dataset}")
    if not (dataset / "export.json").is_file():
        raise FileNotFoundError(f"Run prepare_usfm_dataset.py first: {dataset}")
    info = json.loads((dataset / "export.json").read_text(encoding="utf-8"))
    if args.mode == "test" and info.get("protocol") == "one_training_frame_repeated_across_splits_for_overfit_diagnostic_only":
        raise ValueError("The one-frame memorization set has no independent test data; use it only with --mode train")
    if args.require_patient_split and not info["patient_independence_verified"]:
        raise ValueError("Patient-level split is required; provide an approved patient manifest during export")
    for split in ("training_set", "val_set", "test_set"):
        if not any((dataset / split / "image").glob("*.png")):
            raise ValueError(f"No images in {dataset / split / 'image'}")
    if not args.pretrained or not args.pretrained.is_file():
        raise FileNotFoundError(f"USFM pretraining checkpoint missing: {args.pretrained}; see docs/USFM.md")
    if args.pretrained.with_name(args.pretrained.name + ".INCOMPLETE").exists():
        raise ValueError(f"USFM checkpoint download is incomplete: {args.pretrained}")
    if args.mode == "test" and (not args.resume or not args.resume.is_file()):
        raise FileNotFoundError("A fine-tuned segmentation checkpoint is required: --resume")
    partial = args.freeze_first_blocks is not None
    if partial:
        if args.mode != "train":
            raise ValueError("--freeze-first-blocks is for training; test uses the saved checkpoint")
        if not 1 <= args.freeze_first_blocks <= 12:
            raise ValueError("--freeze-first-blocks must be between 1 and 12; omit it for official full fine-tuning")
        if args.gradient_checkpointing:
            raise ValueError("Disable --gradient-checkpointing when early blocks are frozen")
        if args.accumulation_steps != 1:
            raise ValueError("Partial training requires --accumulation-steps 1; upstream does not accumulate gradients")
        if args.effective_lr is not None:
            raise ValueError("Use --backbone-lr and --head-lr instead of --effective-lr for partial training")
        if args.warmup_epochs not in (None, 0):
            raise ValueError("Partial training requires --warmup-epochs 0 to preserve separate learning rates")
        if args.backbone_lr <= 0 or args.head_lr <= 0:
            raise ValueError("--backbone-lr and --head-lr must be positive")
        if args.log_every < 1:
            raise ValueError("--log-every must be positive")
        if not PARTIAL_ENTRY.is_file():
            raise FileNotFoundError(PARTIAL_ENTRY)
    if args.img_size != 224:
        raise ValueError("Use --img-size 224: the upstream position-bias interpolation produced NaNs at 256 in this setup")
    if args.batch_size < 1 or args.epochs < 1 or args.accumulation_steps < 1:
        raise ValueError("batch-size, epochs and accumulation-steps must be positive")
    if args.effective_lr is not None and args.effective_lr <= 0:
        raise ValueError("effective-lr must be positive")
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        raise ValueError("warmup-epochs must be nonnegative")
    output = args.output.resolve()
    command = [
        sys.executable,
        str(PARTIAL_ENTRY) if partial else "main.py",
        "experiment=task/Seg",
        "data=Seg/tee_valve",
        f'data.path.root="{dataset.as_posix()}"',
        f"data.img_size={args.img_size}",
        f"+model.model_cfg.backbone.use_checkpoint={str(args.gradient_checkpointing).lower()}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        "model=Seg/SegVit",
        "L.devices=1",
        f'hydra.run.dir="{output.as_posix()}"',
        f"mode={args.mode}",
        "tag=TEE_USFM",
        f'model.model_cfg.backbone.pretrained="{args.pretrained.resolve().as_posix()}"',
    ]
    if args.mode == "train":
        command.extend([
            f"train.epochs={args.epochs}",
            f"train.accumulation_steps={args.accumulation_steps}",
            f"train.val_freq={args.val_freq}",
            "train.auto_resume=false",
        ])
        if args.effective_lr is not None:
            # The official trainer scales base_lr by batch_size * world_size / 512,
            # then by accumulation_steps. This launcher uses one device.
            nominal_lr = args.effective_lr * 512 / (args.batch_size * args.accumulation_steps)
            command.append(f"train.base_lr={nominal_lr:.12g}")
        if args.warmup_epochs is not None:
            command.append(f"train.warmup_epochs={args.warmup_epochs}")
        if partial:
            if args.warmup_epochs is None:
                command.append("train.warmup_epochs=0")
            command.extend([
                f"+partial.freeze_first_blocks={args.freeze_first_blocks}",
                f"+partial.backbone_lr={args.backbone_lr:.12g}",
                f"+partial.head_lr={args.head_lr:.12g}",
                f"+partial.log_every={args.log_every}",
            ])
    else:
        command.append(f'model.resume="{args.resume.resolve().as_posix()}"')
        # Checkpoints from partial training have different optimizer parameter
        # groups; testing needs model weights, not optimizer/scheduler state.
        command.append("train.only_resume_model=true")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=PROJECT / "external/USFM")
    parser.add_argument("--dataset", type=Path, default=PROJECT / "outputs/usfm_tee_valve")
    parser.add_argument("--pretrained", type=Path, default=PROJECT / "checkpoints/USFM_latest.pth")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--effective-lr", type=float, help="Approximate actual peak LR after official batch-size scaling (one device)")
    parser.add_argument("--warmup-epochs", type=int, help="Override official warmup length for training")
    parser.add_argument("--freeze-first-blocks", type=int, help="Train late ViT blocks plus FPN/decoder; e.g. 8 freezes blocks 0-7")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Actual LR for trainable late ViT blocks")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Actual LR for FPN and decoder")
    parser.add_argument("--log-every", type=int, default=50, help="Print partial-training progress every N batches")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Reduce GPU memory at the cost of speed")
    parser.add_argument("--require-patient-split", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run after validation; default prints command only")
    args = parser.parse_args()
    command = build_command(args)
    print(subprocess.list2cmdline(command))
    if args.execute:
        if replace_existing_output(
            args.output,
            protected=tuple(path for path in (args.dataset, args.upstream, args.pretrained, args.resume) if path),
        ):
            print(f"Overwriting previous USFM run: {args.output}", flush=True)
        destination = args.upstream.resolve() / "configs/data/Seg/tee_valve.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA_CONFIG, destination)
        subprocess.run(command, cwd=args.upstream.resolve(), check=True)


if __name__ == "__main__":
    main()
