

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage import measure
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import build_inference_loader
from models.pgsrnet import PGSRNet, _to_mask_4d


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROB_THRESHOLD = 0.5
DIST_THRESHOLD = 3.0
NUM_WORKERS = 0
INFMAE_TOTAL_STRIDE = 4 * 2 * 2


@dataclass(frozen=True)
class DatasetConfig:
    root: str
    val_list: str
    image_dir: str
    mask_dir: str
    image_size: int


DATASETS = {
    "irstd1k": DatasetConfig(
        r"E:\datasets\infrared_object_datasets\IRSTD-1k",
        "test.txt",
        "IRSTD1k_Img",
        "IRSTD1k_Label",
        512,
    ),
    "nuaa": DatasetConfig(
        r"E:\datasets\infrared_object_datasets\Dataset\NUAA-SIRST",
        "img_idx/test_NUAA-SIRST.txt",
        "images",
        "masks",
        256,
    ),
    "nudt": DatasetConfig(
        r"E:\datasets\infrared_object_datasets\Dataset\NUDT-SIRST",
        "img_idx/test_NUDT-SIRST.txt",
        "images",
        "masks",
        256,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", default="nuaa", choices=DATASETS)
    data.add_argument("--root", default=None, help="override the configured dataset root")
    data.add_argument("--val-list", "--val_list", dest="val_list", default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--ckpt", default=r'D:\codes\PGSRNet\weights\nuaa.pth')
    model.add_argument("--pretrain-path", default=None, help="needed for adapter-only checkpoints")
    model.add_argument("--device", default="cuda:0")
    model.add_argument("--amp", action="store_true", help="enable mixed-precision inference")

    output = parser.add_argument_group("evaluation and output")
    output.add_argument("--save-dir", default="predict_results")
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def load_checkpoint_file(path: str) -> tuple[Any, dict[str, Any], dict[str, torch.Tensor]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    checkpoint_args = _as_dict(checkpoint.get("args", {})) if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint, checkpoint_args, checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    state_dict = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if not state_dict:
        raise KeyError("no model/state_dict/model_state_dict found in checkpoint")
    return checkpoint, checkpoint_args, state_dict


def build_model(
    args: argparse.Namespace,
    checkpoint_args: dict[str, Any],
    image_size: int,
) -> PGSRNet:
    return PGSRNet(
        crop_size_base=checkpoint_args.get("crop_size_base", 32),
        up_size=checkpoint_args.get("up_size", 32),
        freeze_backbone=True,
        pretrain_path=args.pretrain_path or None,
        target_input_size=image_size,
        use_adapter=True,
        adapter_mlp_ratio_token=checkpoint_args.get("adapter_mlp_ratio_token", 0.25),
        adapter_mlp_ratio_2d=checkpoint_args.get("adapter_mlp_ratio_2d", 0.25),
        adapter_gn_groups=checkpoint_args.get("adapter_gn_groups", 16),
        adapter_k_stat=checkpoint_args.get("adapter_k_stat", 3),
        adapter_detach_gate=checkpoint_args.get("adapter_detach_gate", True),
        thr_s16=checkpoint_args.get("thr_s16", 0.3),
        stride_roi=checkpoint_args.get("stride_roi", 16),
        step_s16=checkpoint_args.get("step_s16", 1),
        max_rois=checkpoint_args.get("max_rois", 25),
        patch_chunk=checkpoint_args.get("patch_chunk", 64),
        saliency_thr=checkpoint_args.get("saliency_thr", 0.0),
        use_bmsr=True,
        use_boundary_refine=True,
        patch_use_imagenet=False,
        pos_weight=checkpoint_args.get("pos_weight", 9.0),
        lam_global_up=checkpoint_args.get("lam_global_up", 1.0),
        lam_global_ref=checkpoint_args.get("lam_global_ref", 1.0),
        lam_patch=checkpoint_args.get("lam_patch", 2.0),
        lam_patch_pred=checkpoint_args.get("lam_patch_pred", 1.0),
        lam_patch_ref=checkpoint_args.get("lam_patch_ref", 1.0),
        use_sal_aux=checkpoint_args.get("use_sal_aux", False),
        sal_mid=checkpoint_args.get("sal_mid", 128),
        lam_sal_f2=checkpoint_args.get("lam_sal_f2", 0.5),
        lam_sal_f4=checkpoint_args.get("lam_sal_f4", 0.5),
        pos_weight_sal=checkpoint_args.get("pos_weight_sal", None),
        detach_base_patch=checkpoint_args.get("detach_base_patch", True),
        use_mt_loss=False,
    )


def infer_checkpoint_input_size(
    state_dict: dict[str, torch.Tensor],
    fallback: int,
) -> int:
    """Recover the encoder training grid from an absolute position embedding."""
    normalized = normalize_state_dict(state_dict)
    for key in (
        "encoder.pos_embed",
        "backbone.pos_embed",
        "backbone.backbone.pos_embed",
    ):
        value = normalized.get(key)
        if value is None or value.ndim != 3:
            continue
        token_count = int(value.shape[1])
        for extra_tokens in (1, 0):
            patch_count = token_count - extra_tokens
            grid_size = int(round(patch_count ** 0.5))
            if grid_size * grid_size == patch_count:
                return grid_size * INFMAE_TOTAL_STRIDE
    return int(fallback)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = dict(state_dict)
    for prefix in ("module.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_model_weights(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    state_dict = normalize_state_dict(state_dict)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    shape_mismatch = [
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    unexpected = [key for key in state_dict if key not in model_state]
    training_only = [key for key in unexpected if key.startswith(("mt_stage1.", "mt_stage2."))]
    unexpected = [key for key in unexpected if key not in training_only]

    print(
        "[checkpoint] "
        f"loaded={len(compatible)}/{len(model_state)}, missing={len(missing)}, "
        f"unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatch)}"
    )
    for label, keys in (("missing", missing), ("unexpected", unexpected), ("mismatch", shape_mismatch)):
        if keys:
            print(f"[{label}] {list(keys)[:10]}")


def build_val_loader(args: argparse.Namespace):
    config = DATASETS[args.dataset]
    root = Path(args.root or config.root)
    val_list = Path(args.val_list or config.val_list)
    if not val_list.is_absolute():
        val_list = root / val_list
    image_size = config.image_size
    val_loader = build_inference_loader(
        root=str(root),
        val_list=str(val_list),
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        crop_size=image_size if args.dataset == "nuaa" else None,
        num_workers=NUM_WORKERS,
    )
    return val_loader, image_size


def object_metrics(prediction: np.ndarray, target: np.ndarray, distance: float) -> tuple[int, float, int]:
    pred_regions = list(measure.regionprops(measure.label(prediction, connectivity=2)))
    gt_regions = list(measure.regionprops(measure.label(target, connectivity=2)))
    unmatched = pred_regions.copy()
    matched_area = 0.0
    detections = 0
    for gt_region in gt_regions:
        gt_center = np.asarray(gt_region.centroid, dtype=np.float32)
        for index, pred_region in enumerate(unmatched):
            pred_center = np.asarray(pred_region.centroid, dtype=np.float32)
            if np.linalg.norm(pred_center - gt_center) < distance:
                matched_area += pred_region.area
                detections += 1
                del unmatched[index]
                break
    false_alarm_area = float(sum(region.area for region in pred_regions) - matched_area)
    return len(gt_regions), false_alarm_area, detections


class MetricMeter:
    def __init__(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.image_iou = 0.0
        self.images = 0
        self.tp = self.fp = self.fn = 0.0
        self.targets = self.detections = 0
        self.false_alarm_area = 0.0
        self.pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, distance: float) -> None:
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target).clamp_max(1).sum(dim=(1, 2, 3))
        self.intersection += intersection.sum().item()
        self.union += union.sum().item()
        per_image = torch.where(union > 0, intersection / union, torch.ones_like(union))
        self.image_iou += per_image.sum().item()
        self.images += pred.shape[0]
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.pixels += target.numel()

        pred_np = pred.detach().cpu().numpy().astype(np.uint8)
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        for index in range(pred.shape[0]):
            count, false_area, detected = object_metrics(
                pred_np[index, 0], target_np[index, 0], distance
            )
            self.targets += count
            self.false_alarm_area += false_area
            self.detections += detected

    def compute(self) -> dict[str, float]:
        precision = self.tp / max(1.0, self.tp + self.fp)
        recall = self.tp / max(1.0, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        false_alarm = self.false_alarm_area / max(1, self.pixels)
        return {
            "IoU": self.intersection / max(1.0, self.union),
            "nIoU": self.image_iou / max(1, self.images),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Pd": self.detections / max(1, self.targets),
            "Fa": false_alarm,
            "Fa_1e6": false_alarm * 1e6,
        }


def sample_stem(batch: dict[str, Any], index: int, fallback: str) -> str:
    for key in ("img_path", "name", "img_name", "id", "filename", "path"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            value = value[index] if index < len(value) else fallback
        return Path(str(value)).stem
    return fallback


def difference_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = np.zeros((*target.shape, 3), dtype=np.uint8)
    output[(pred == 1) & (target == 1)] = (255, 255, 255)
    output[(pred == 1) & (target == 0)] = (255, 0, 0)
    output[(pred == 0) & (target == 1)] = (0, 255, 0)
    return output


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def prepare_output_dirs(args: argparse.Namespace) -> Path:
    output = Path(args.save_dir)
    if not output.is_absolute():
        output = Path(args.ckpt).parent / output
        output.mkdir(parents=True, exist_ok=True)
    # for name in ("pred_bin", "gt_bin", "diff", "panel"):
    #     (output / name).mkdir(parents=True, exist_ok=True)
    return output


def save_batch(
    output: Path,
    batch: dict[str, Any],
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    iteration: int,
) -> None:
    for index in range(images.shape[0]):
        stem = sample_stem(batch, index, f"{iteration:06d}_{index}")
        target = targets[index, 0].detach().cpu().numpy().astype(np.uint8)
        pred = predictions[index, 0].detach().cpu().numpy().astype(np.uint8)
        target_u8, pred_u8 = target * 255, pred * 255
        diff = difference_map(pred, target)
        Image.fromarray(target_u8).save(output / "gt_bin" / f"{stem}.png")
        Image.fromarray(pred_u8).save(output / "pred_bin" / f"{stem}.png")
        Image.fromarray(diff).save(output / "diff" / f"{stem}.png")
        image = denormalize_image(images[index].detach().float())
        top = np.concatenate((image, np.repeat(target_u8[..., None], 3, axis=2)), axis=1)
        bottom = np.concatenate((np.repeat(pred_u8[..., None], 3, axis=2), diff), axis=1)
        Image.fromarray(np.concatenate((top, bottom), axis=0)).save(
            output / "panel" / f"{stem}.png"
        )


def format_summary(metrics: dict[str, float], args: argparse.Namespace, output: Path) -> str:
    return (
        f"dataset={args.dataset} stage=2 prob_thr={PROB_THRESHOLD} "
        f"dist_thr={DIST_THRESHOLD}\n"
        f"IoU={metrics['IoU']:.6f}\n"
        f"nIoU={metrics['nIoU']:.6f}\n"
        f"Precision={metrics['Precision']:.6f}\n"
        f"Recall={metrics['Recall']:.6f}\n"
        f"F1-score={metrics['F1']:.6f}\n"
        f"Pd={metrics['Pd']:.6f}\n"
        f"Fa={metrics['Fa']:.8f} (Fa*1e6={metrics['Fa_1e6']:.4f})\n"
        f"Saved to: {output}\n"
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = prepare_output_dirs(args)
    val_loader, image_size = build_val_loader(args)
    _, checkpoint_args, state_dict = load_checkpoint_file(args.ckpt)
    model_input_size = infer_checkpoint_input_size(state_dict, image_size)
    print(f"[model] checkpoint_input_size={model_input_size}, data_size={image_size}")
    model = build_model(args, checkpoint_args, model_input_size).to(device)
    load_model_weights(model, state_dict)
    model.eval()

    meter = MetricMeter()
    progress = tqdm(val_loader, desc=f"{args.dataset} stage-2", ncols=130)
    for iteration, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = _to_mask_4d(batch["mask"], device=device)
        amp_enabled = bool(args.amp and device.type == "cuda")
        with autocast(enabled=amp_enabled):
            output_dict = model(images, coarse_only=False)
        probabilities = torch.sigmoid(output_dict["logits"])
        predictions = (probabilities > PROB_THRESHOLD).float()
        meter.update(predictions, targets, DIST_THRESHOLD)
        # save_batch(
        #     output,
        #     batch,
        #     images,
        #     targets,
        #     predictions,
        #     iteration,
        # )
        metrics = meter.compute()
        progress.set_postfix(
            IoU=f"{metrics['IoU']:.4f}",
            nIoU=f"{metrics['nIoU']:.4f}",
            F1=f"{metrics['F1']:.4f}",
            Pd=f"{metrics['Pd']:.4f}",
            Fa=f"{metrics['Fa_1e6']:.2f}",
        )

    summary = format_summary(meter.compute(), args, output)
    print(summary)
    (output / "infer_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
