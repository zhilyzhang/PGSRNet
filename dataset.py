"""Minimal inference dataset for IRSTD-1k, NUAA-SIRST, and NUDT-SIRST."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _resolve_file(root: Path, directory: Path, token: str) -> Path:
    token_path = Path(token.strip().replace("\\", "/"))
    candidates = [token_path, root / token_path, directory / token_path]
    for path in candidates:
        if path.is_file():
            return path

    if token_path.suffix == "":
        for extension in IMAGE_EXTENSIONS:
            for path in (directory / f"{token_path}{extension}", root / f"{token_path}{extension}"):
                if path.is_file():
                    return path
    raise FileNotFoundError(f"Cannot resolve file: {token}")


def _read_pairs(root: Path, list_file: Path, image_dir: Path, mask_dir: Path):
    if not list_file.is_file():
        raise FileNotFoundError(f"Validation list not found: {list_file}")

    pairs = []
    with list_file.open("r", encoding="utf-8") as file:
        for line in file:
            items = line.strip().split()
            if not items:
                continue
            image_token = items[0]
            mask_token = items[1] if len(items) > 1 else items[0]
            pairs.append(
                (
                    _resolve_file(root, image_dir, image_token),
                    _resolve_file(root, mask_dir, mask_token),
                )
            )
    if not pairs:
        raise RuntimeError(f"No samples found in: {list_file}")
    return pairs


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)


def _pad_to_size(image: np.ndarray, mask: np.ndarray, size: int):
    height, width = image.shape[:2]
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        mask = cv2.copyMakeBorder(
            mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
    return image, mask


def _target_aware_crop(image: np.ndarray, mask: np.ndarray, size: int):
    """Deterministic NUAA crop that retains the annotated target when possible."""
    height, width = image.shape[:2]
    if height == size and width == size:
        return image, mask

    ys, xs = np.where(mask > 0)
    if ys.size:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        y_low, y_high = max(0, y2 - size), min(height - size, y1)
        x_low, x_high = max(0, x2 - size), min(width - size, x1)
        y0 = int(np.clip(cy - size // 2, y_low, y_high)) if y_low <= y_high else int(
            np.clip(cy - size // 2, 0, height - size)
        )
        x0 = int(np.clip(cx - size // 2, x_low, x_high)) if x_low <= x_high else int(
            np.clip(cx - size // 2, 0, width - size)
        )
    else:
        y0, x0 = (height - size) // 2, (width - size) // 2
    return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


class IRSTDInferenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        list_file: str,
        image_dir: str,
        mask_dir: str,
        crop_size: Optional[int] = None,
    ) -> None:
        root_path = Path(root)
        image_path = Path(image_dir) if os.path.isabs(image_dir) else root_path / image_dir
        mask_path = Path(mask_dir) if os.path.isabs(mask_dir) else root_path / mask_dir
        self.samples = _read_pairs(root_path, Path(list_file), image_path, mask_path)
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = _read_image(image_path)
        mask = _read_mask(mask_path)

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        if self.crop_size is not None:
            image, mask = _pad_to_size(image, mask, self.crop_size)
            image, mask = _target_aware_crop(image, mask, self.crop_size)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return {
            "image": image,
            "mask": mask,
            "img_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_inference_loader(
    root: str,
    val_list: str,
    image_dir: str,
    mask_dir: str,
    crop_size: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = IRSTDInferenceDataset(
        root=root,
        list_file=val_list,
        image_dir=image_dir,
        mask_dir=mask_dir,
        crop_size=crop_size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
