from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path, PurePosixPath


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]
DEFAULT_CACHE_NAME = "ksdd2_aug_cache"


def load_runtime_deps() -> None:
    global np, torch, TF, Image, ElasticTransform, InterpolationMode, tqdm

    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms import ElasticTransform, InterpolationMode
    from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a preprocessed augmentation cache for KolektorSDD2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("KolektorSDD2"),
        help="Local KolektorSDD2 folder containing train/ and test/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_CACHE_NAME),
        help="Local folder to write the cache into.",
    )
    parser.add_argument(
        "--manifest-cache-root",
        default="/content/ksdd2_aug_cache",
        help=(
            "Path that the cache will have in Colab. Use /content/ksdd2_aug_cache "
            "if you will unzip the cache to Colab local disk."
        ),
    )
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elastic-prob", type=float, default=0.6)
    parser.add_argument("--elastic-alpha", type=float, default=35.0)
    parser.add_argument("--elastic-sigma", type=float, default=5.0)
    parser.add_argument("--gaussian-noise-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.02)
    parser.add_argument("--compress-level", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite output-root if it exists.")
    zip_group = parser.add_mutually_exclusive_group()
    zip_group.add_argument(
        "--zip",
        dest="zip",
        action="store_true",
        help="Create output-root.zip. This is enabled by default.",
    )
    zip_group.add_argument(
        "--no-zip",
        dest="zip",
        action="store_false",
        help="Only create the cache folder and skip the zip archive.",
    )
    parser.set_defaults(zip=True)
    return parser.parse_args()


def resolve_data_root(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    candidates = [root, root / "KolektorSDD2"]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    for train_dir in root.rglob("train"):
        candidate = train_dir.parent
        if (candidate / "test").is_dir():
            return candidate

    raise FileNotFoundError(f"Could not find KolektorSDD2 train/test folders under: {root}")


def is_gt_file(path: Path) -> bool:
    return path.stem.lower().endswith("_gt")


def find_ksdd2_mask(img_path: Path) -> Path | None:
    mask_exts = list(dict.fromkeys([img_path.suffix.lower()] + MASK_EXTS))
    for ext in mask_exts:
        mask_path = img_path.with_name(f"{img_path.stem}_GT{ext}")
        if mask_path.exists():
            return mask_path
    return None


def mask_has_positive(mask_path: Path | None) -> bool:
    if mask_path is None:
        return False
    with Image.open(mask_path) as mask_img:
        mask = np.array(mask_img.convert("L"))
    return bool((mask > 0).any())


def collect_train_samples(root: Path) -> list[dict[str, object]]:
    split_dir = root / "train"
    samples: list[dict[str, object]] = []

    for img_path in sorted(split_dir.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if is_gt_file(img_path):
            continue

        mask_path = find_ksdd2_mask(img_path)
        samples.append(
            {
                "image": img_path,
                "mask": mask_path,
                "has_crack": mask_has_positive(mask_path),
            }
        )

    if not samples:
        raise RuntimeError(f"No input images found in {split_dir}")
    return samples


def load_sample_pil(sample: dict[str, object]) -> tuple[Image.Image, Image.Image]:
    img_path = Path(sample["image"])
    mask_path = sample["mask"]

    img = Image.open(img_path).convert("RGB")
    if mask_path is None:
        mask = Image.new("L", img.size, 0)
    else:
        mask = Image.open(Path(mask_path)).convert("L")
        if mask.size != img.size:
            raise ValueError(
                f"Image/mask size mismatch: image={img_path}, mask={mask_path}"
            )
    return img, mask


def elastic_transform_pair(
    img: Image.Image,
    mask: Image.Image,
    alpha: float,
    sigma: float,
) -> tuple[Image.Image, Image.Image]:
    width, height = img.size
    displacement = ElasticTransform.get_params(
        alpha=[alpha, alpha],
        sigma=[sigma, sigma],
        size=[height, width],
    )
    img = TF.elastic_transform(
        img,
        displacement,
        interpolation=InterpolationMode.BILINEAR,
        fill=[0, 0, 0],
    )
    mask = TF.elastic_transform(
        mask,
        displacement,
        interpolation=InterpolationMode.NEAREST,
        fill=[0],
    )
    return img, mask


def apply_training_augmentation(
    img: Image.Image,
    mask: Image.Image,
    args: argparse.Namespace,
) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() < 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    if random.random() < 0.3:
        img = TF.rotate(img, angle=180)
        mask = TF.rotate(mask, angle=180)

    if random.random() < args.elastic_prob:
        img, mask = elastic_transform_pair(img, mask, args.elastic_alpha, args.elastic_sigma)

    if random.random() < 0.4:
        img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))

    return img, mask


def maybe_add_gaussian_noise(img_t: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if random.random() < args.gaussian_noise_prob:
        img_t = (img_t + torch.randn_like(img_t) * args.gaussian_noise_std).clamp(0.0, 1.0)
    return img_t


def colab_cache_path(manifest_cache_root: str, *parts: str) -> str:
    return str(PurePosixPath(manifest_cache_root, *parts))


def save_cached_augmented_sample(
    sample: dict[str, object],
    out_dir: Path,
    sample_idx: int,
    copy_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    img, mask = load_sample_pil(sample)
    try:
        img, mask = apply_training_augmentation(img, mask, args)

        img_t = maybe_add_gaussian_noise(TF.to_tensor(img), args)
        img_np = (
            (img_t.permute(1, 2, 0).numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        mask_np = (np.array(mask) > 0).astype(np.uint8) * 255
    finally:
        img.close()
        mask.close()

    stem = f"{sample_idx:05d}_aug{copy_idx:02d}"
    image_name = f"{stem}.png"
    mask_name = f"{stem}_GT.png"

    Image.fromarray(img_np).save(out_dir / image_name, compress_level=args.compress_level)
    Image.fromarray(mask_np).save(out_dir / mask_name, compress_level=args.compress_level)

    return {
        "image": colab_cache_path(args.manifest_cache_root, "train", image_name),
        "mask": colab_cache_path(args.manifest_cache_root, "train", mask_name),
        "split": "train_preprocessed",
        "has_crack": bool(sample["has_crack"]),
        "source_image": str(sample["image"]),
        "copy_idx": int(copy_idx),
    }


def preprocess_cache_config(num_raw_samples: int, args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_raw_samples": int(num_raw_samples),
        "copies_per_sample": int(args.copies),
        "seed": int(args.seed),
        "elastic_prob": float(args.elastic_prob),
        "elastic_alpha": float(args.elastic_alpha),
        "elastic_sigma": float(args.elastic_sigma),
        "gaussian_noise_prob": float(args.gaussian_noise_prob),
        "gaussian_noise_std": float(args.gaussian_noise_std),
    }


def main() -> None:
    args = parse_args()
    load_runtime_deps()

    if args.copies < 1:
        raise ValueError("--copies must be at least 1")
    if not 0 <= args.compress_level <= 9:
        raise ValueError("--compress-level must be in [0, 9]")

    data_root = resolve_data_root(args.data_root)
    output_root = args.output_root.expanduser().resolve()
    out_dir = output_root / "train"

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = collect_train_samples(data_root)
    config = preprocess_cache_config(len(train_samples), args)

    print(f"Dataset root: {data_root}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Copies per sample: {args.copies}")
    print(f"Local cache output: {output_root}")
    print(f"Manifest cache root for Colab: {args.manifest_cache_root}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    cached_samples = []
    for sample_idx, sample in enumerate(tqdm(train_samples, desc="Building cache")):
        for copy_idx in range(args.copies):
            cached_samples.append(
                save_cached_augmented_sample(sample, out_dir, sample_idx, copy_idx, args)
            )

    manifest = {
        "config": config,
        "samples": cached_samples,
        "elapsed_seconds": time.time() - start,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = manifest["elapsed_seconds"]
    print(f"Done in {elapsed / 60:.1f} min | Cached samples: {len(cached_samples)}")

    if args.zip:
        archive_base = output_root.parent / output_root.name
        zip_path = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
        print(f"Created archive: {zip_path}")


if __name__ == "__main__":
    main()
