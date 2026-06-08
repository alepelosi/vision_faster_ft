from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path, PurePosixPath


try:
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms import ElasticTransform, InterpolationMode
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    np = None
    torch = None
    TF = None
    Image = None
    ElasticTransform = None
    InterpolationMode = None
    tqdm = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]

DEFAULT_DATA_ROOT = Path("KolektorSDD-boxes")
DEFAULT_RESNET_OUTPUT_ROOT = Path("ksdd1_aug_cache")
DEFAULT_SEGFORMER_OUTPUT_ROOT = Path("ksdd1_segformer_aug_cache")
DEFAULT_RESNET_MANIFEST_CACHE_ROOT = "/content/ksdd1_aug_cache"
DEFAULT_SEGFORMER_MANIFEST_CACHE_ROOT = "/content/ksdd1_segformer_aug_cache"
DEFAULT_IMAGE_SIZE = (1408, 512)

DEFAULT_COPIES = 24
DEFAULT_COPIES_PER_EPOCH = 2
DEFAULT_ELASTIC_PROB = 0.3
DEFAULT_ELASTIC_ALPHA = 25.0
DEFAULT_ELASTIC_SIGMA = 5.0
DEFAULT_GAUSSIAN_NOISE_PROB = 0.2
DEFAULT_GAUSSIAN_NOISE_STD = 0.01


def require_runtime_deps() -> None:
    if IMPORT_ERROR is None:
        return
    raise SystemExit(
        "Missing dependency while building the cache. Install the local "
        "requirements first, for example: pip install -r requirements.txt"
    ) from IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a large pre-augmented KolektorSDD1 cache pool locally. "
            "The generated cache is compatible with the SDD1 side-tuning "
            "notebooks."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--resnet",
        action="store_true",
        help="Build the cache expected by the ResNet18-UNet SDD1 side-tuning notebook.",
    )
    model_group.add_argument(
        "--segformer",
        action="store_true",
        help="Build the cache expected by the SegFormer-B0 SDD1 side-tuning notebook.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Local KolektorSDD1 folder containing kosXX folders, or train/test.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Local folder to write the cache into. Defaults depend on --resnet/--segformer.",
    )
    parser.add_argument(
        "--manifest-cache-root",
        default=None,
        help=(
            "Path the cache folder will have in Colab after unzip. The manifest "
            "uses this path for image/mask entries. Defaults depend on --resnet/--segformer."
        ),
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=DEFAULT_COPIES,
        help="Total cached augmented copies to precompute per source image.",
    )
    parser.add_argument(
        "--cache-copies-per-epoch",
        type=int,
        default=DEFAULT_COPIES_PER_EPOCH,
        help=(
            "Recommended number of cached copies to draw per source image each "
            "epoch in the notebook. This is metadata only and does not affect "
            "the cache validity check."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize target in TorchVision order: HEIGHT WIDTH.",
    )
    parser.add_argument("--elastic-prob", type=float, default=DEFAULT_ELASTIC_PROB)
    parser.add_argument("--elastic-alpha", type=float, default=DEFAULT_ELASTIC_ALPHA)
    parser.add_argument("--elastic-sigma", type=float, default=DEFAULT_ELASTIC_SIGMA)
    parser.add_argument("--gaussian-noise-prob", type=float, default=DEFAULT_GAUSSIAN_NOISE_PROB)
    parser.add_argument("--gaussian-noise-std", type=float, default=DEFAULT_GAUSSIAN_NOISE_STD)
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


def apply_model_defaults(args: argparse.Namespace) -> None:
    if args.segformer:
        args.model_cache = "segformer"
        default_output_root = DEFAULT_SEGFORMER_OUTPUT_ROOT
        default_manifest_cache_root = DEFAULT_SEGFORMER_MANIFEST_CACHE_ROOT
    else:
        args.model_cache = "resnet"
        default_output_root = DEFAULT_RESNET_OUTPUT_ROOT
        default_manifest_cache_root = DEFAULT_RESNET_MANIFEST_CACHE_ROOT

    if args.output_root is None:
        args.output_root = default_output_root
    if args.manifest_cache_root is None:
        args.manifest_cache_root = default_manifest_cache_root


def validate_args(args: argparse.Namespace) -> None:
    if args.copies < 1:
        raise ValueError("--copies must be at least 1")
    if args.cache_copies_per_epoch < 1:
        raise ValueError("--cache-copies-per-epoch must be at least 1")
    if args.cache_copies_per_epoch > args.copies:
        raise ValueError("--cache-copies-per-epoch cannot exceed --copies")
    if args.cache_copies_per_epoch == args.copies:
        print(
            "Warning: --cache-copies-per-epoch equals --copies; every epoch will "
            "see all cached copies, so there is no pool-sampling benefit."
        )
    if not 0 <= args.compress_level <= 9:
        raise ValueError("--compress-level must be in [0, 9]")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(args.image_size) != 2:
        raise ValueError("--image-size expects HEIGHT WIDTH")


def is_mask_path(path: Path) -> bool:
    lower = path.stem.lower()
    return lower.endswith("_gt") or lower.endswith("_label") or lower.endswith("_mask")


def has_train_test(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "test").is_dir()


def has_sdd1_groups(root: Path) -> bool:
    if not root.exists():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        has_image = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTS and not is_mask_path(path)
            for path in child.iterdir()
        )
        has_label = any(
            path.is_file() and path.stem.lower().endswith("_label")
            for path in child.iterdir()
        )
        if has_image and has_label:
            return True
    return False


def resolve_data_root(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    candidates = [
        root,
        root / "KolektorSDD-boxes",
        root / "KolektorSDD",
        root / "KolektorSDD1",
    ]
    for candidate in candidates:
        if candidate.exists() and (has_train_test(candidate) or has_sdd1_groups(candidate)):
            return candidate

    for candidate in root.rglob("*"):
        if candidate.is_dir() and (has_train_test(candidate) or has_sdd1_groups(candidate)):
            return candidate

    raise FileNotFoundError(f"Could not find KolektorSDD1 data under: {root}")


def find_mask(image_path: Path) -> Path | None:
    exact_sdd1_mask = image_path.with_name(f"{image_path.stem}_label.bmp")
    if exact_sdd1_mask.exists():
        return exact_sdd1_mask

    stems = [
        f"{image_path.stem}_GT",
        f"{image_path.stem}_gt",
        f"{image_path.stem}_label",
        f"{image_path.stem}_mask",
    ]
    suffixes = list(dict.fromkeys([image_path.suffix.lower()] + MASK_EXTS))
    for stem in stems:
        for suffix in suffixes:
            candidate = image_path.with_name(stem + suffix)
            if candidate.exists() and candidate.resolve() != image_path.resolve():
                return candidate
    return None


def mask_has_positive(mask_path: Path | None) -> bool:
    require_runtime_deps()
    if mask_path is None:
        return False
    with Image.open(mask_path) as mask_img:
        mask = np.array(mask_img.convert("L"))
    return bool((mask > 0).any())


def collect_split_samples(root: Path, split_name: str) -> list[dict[str, object]]:
    split_dir = root / split_name
    samples: list[dict[str, object]] = []
    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if is_mask_path(image_path):
            continue

        mask_path = find_mask(image_path)
        samples.append(
            {
                "image": image_path,
                "mask": mask_path,
                "has_crack": mask_has_positive(mask_path),
                "sample_id": image_path.relative_to(split_dir).with_suffix("").as_posix(),
            }
        )

    if not samples:
        raise RuntimeError(f"No input images found in {split_dir}")
    return samples


def collect_sdd1_samples(root: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for kos_folder in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(kos_folder.glob("*.jpg")):
            if not image_path.is_file() or is_mask_path(image_path):
                continue
            mask_path = find_mask(image_path)
            samples.append(
                {
                    "image": image_path,
                    "mask": mask_path,
                    "has_crack": mask_has_positive(mask_path),
                    "sample_id": f"{kos_folder.name}/{image_path.stem}",
                }
            )

    if not samples:
        raise RuntimeError(f"No SDD1 kosXX/Part*.jpg samples found in {root}")
    return samples


def split_sdd1_samples(
    root: Path,
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples = collect_sdd1_samples(root)
    positive = [sample for sample in samples if sample["has_crack"]]
    negative = [sample for sample in samples if not sample["has_crack"]]
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    def split_class(items: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if not items:
            return [], []
        n_val = max(1, int(round(len(items) * val_fraction)))
        n_val = min(n_val, max(1, len(items) - 1)) if len(items) > 1 else 1
        return items[n_val:], items[:n_val]

    pos_train, pos_val = split_class(positive)
    neg_train, neg_val = split_class(negative)
    if pos_train:
        neg_train = rng.sample(neg_train, min(len(neg_train), 2 * len(pos_train)))

    train_samples = pos_train + neg_train
    val_samples = pos_val + neg_val
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def load_train_samples(root: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    if has_train_test(root):
        samples = collect_split_samples(root, "train")
        split_name = "official train split"
    else:
        samples, val_samples = split_sdd1_samples(root, args.seed, args.val_fraction)
        split_name = "old KolektorSDD stratified sample split"
        val_pos = sum(bool(sample["has_crack"]) for sample in val_samples)
        print(f"Validation holdout from split: {len(val_samples)} images ({val_pos} positive)")

    positives = sum(bool(sample["has_crack"]) for sample in samples)
    negatives = len(samples) - positives
    print(f"Split: {split_name}")
    print(f"Training samples used for cache: {len(samples)} ({positives} positive, {negatives} negative)")
    return samples


def load_sample_pil(
    sample: dict[str, object],
    image_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    require_runtime_deps()
    img_path = Path(sample["image"])
    mask_path = sample["mask"]

    with Image.open(img_path) as img_file:
        img = img_file.convert("RGB")

    if mask_path is None:
        mask = Image.new("L", img.size, 0)
    else:
        with Image.open(Path(mask_path)) as mask_file:
            mask = mask_file.convert("L")
        if mask.size != img.size:
            img.close()
            mask.close()
            raise ValueError(f"Image/mask size mismatch: image={img_path}, mask={mask_path}")

    img = TF.resize(img, image_size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, image_size, interpolation=InterpolationMode.NEAREST)
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
        img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.9, 1.1))
        img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.9, 1.1))

    return img, mask


def maybe_add_gaussian_noise(img_t: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if random.random() < args.gaussian_noise_prob:
        img_t = (img_t + torch.randn_like(img_t) * args.gaussian_noise_std).clamp(0.0, 1.0)
    return img_t


def colab_cache_path(manifest_cache_root: str, *parts: str) -> str:
    return str(PurePosixPath(manifest_cache_root, *parts))


def manifest_row(
    sample: dict[str, object],
    image_name: str,
    mask_name: str,
    sample_idx: int,
    copy_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    image_path = colab_cache_path(args.manifest_cache_root, "train", image_name)
    mask_path = colab_cache_path(args.manifest_cache_root, "train", mask_name)
    sample_id = f"{sample_idx:05d}_aug{copy_idx:02d}"
    label = int(bool(sample["has_crack"]))

    return {
        "image": image_path,
        "mask": mask_path,
        "image_path": image_path,
        "mask_path": mask_path,
        "split": "train_preprocessed",
        "has_crack": bool(sample["has_crack"]),
        "label": label,
        "sample_id": sample_id,
        "source_image": str(sample["image"]),
        "copy_idx": int(copy_idx),
    }


def preprocess_cache_config(num_raw_samples: int, args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_raw_samples": int(num_raw_samples),
        "copies_per_sample": int(args.copies),
        "seed": int(args.seed),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "elastic_prob": float(args.elastic_prob),
        "elastic_alpha": float(args.elastic_alpha),
        "elastic_sigma": float(args.elastic_sigma),
        "gaussian_noise_prob": float(args.gaussian_noise_prob),
        "gaussian_noise_std": float(args.gaussian_noise_std),
    }


def prepare_output_dir(output_root: Path, force: bool) -> tuple[Path, Path]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite it.")
        shutil.rmtree(output_root)

    tmp_root = output_root.parent / f"{output_root.name}._tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "train").mkdir(parents=True, exist_ok=True)
    return output_root, tmp_root


def finalize_output_dir(tmp_root: Path, output_root: Path) -> Path:
    if output_root.exists():
        shutil.rmtree(output_root)
    tmp_root.rename(output_root)
    return output_root


def remove_stale_zip(output_root: Path) -> None:
    zip_path = output_root.parent / f"{output_root.name}.zip"
    if zip_path.exists():
        zip_path.unlink()


def save_cached_sample(
    img_np: np.ndarray,
    mask_np: np.ndarray,
    output_root: Path,
    sample: dict[str, object],
    sample_idx: int,
    copy_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    stem = f"{sample_idx:05d}_aug{copy_idx:02d}"
    image_name = f"{stem}.png"
    mask_name = f"{stem}_GT.png"

    Image.fromarray(img_np).save(output_root / "train" / image_name, compress_level=args.compress_level)
    Image.fromarray(mask_np).save(output_root / "train" / mask_name, compress_level=args.compress_level)
    return manifest_row(sample, image_name, mask_name, sample_idx, copy_idx, args)


def build_augmented_arrays(
    sample: dict[str, object],
    sample_idx: int,
    copy_idx: int,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw_img, raw_mask = load_sample_pil(sample, image_size)
        img, mask = apply_training_augmentation(raw_img, raw_mask, args)
        img_t = maybe_add_gaussian_noise(TF.to_tensor(img), args)
        img_np = (
            (img_t.permute(1, 2, 0).numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        mask_np = (np.array(mask) > 0).astype(np.uint8) * 255
        return img_np, mask_np
    except Exception as exc:
        raise RuntimeError(
            f"Failed while augmenting sample_idx={sample_idx}, copy_idx={copy_idx}, "
            f"image={sample.get('image')}"
        ) from exc


def write_manifest(
    output_root: Path,
    config: dict[str, object],
    samples: list[dict[str, object]],
    elapsed: float,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "config": config,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "cache_pool": {
            "copies_per_epoch": int(args.cache_copies_per_epoch),
            "description": (
                "Runtime sampling hint only. Keep this out of config so changing "
                "the per-epoch draw size does not invalidate the precomputed pool."
            ),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def print_notebook_settings(args: argparse.Namespace) -> None:
    print(f"\nUse these {args.model_cache} notebook settings with this cache pool:")
    print("USE_PREPROCESSED_TRAIN = True")
    print(f"AUGMENTED_COPIES_PER_SAMPLE = {args.copies}")
    print(f"CACHE_COPIES_PER_EPOCH = {args.cache_copies_per_epoch}")
    print(f'PREPROCESS_CACHE_ROOT = Path("{args.manifest_cache_root}")')
    print(f'DRIVE_PREPROCESS_CACHE_ZIP = Path("/content/drive/MyDrive/{args.output_root.name}.zip")')


def main() -> None:
    args = parse_args()
    apply_model_defaults(args)
    require_runtime_deps()
    validate_args(args)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    args.image_size = image_size

    data_root = resolve_data_root(args.data_root)
    output_root, tmp_root = prepare_output_dir(args.output_root, args.force)
    train_samples = load_train_samples(data_root, args)
    config = preprocess_cache_config(len(train_samples), args)

    print(f"Dataset root: {data_root}")
    print(f"Cache target: {args.model_cache}")
    print(f"Output cache: {output_root}")
    print(f"Temporary build cache: {tmp_root}")
    print(f"Manifest cache root: {args.manifest_cache_root}")
    print(f"Image size: {image_size} (HEIGHT, WIDTH)")
    print(f"Cached copies per source image: {args.copies}")
    print(f"Recommended copies drawn per epoch: {args.cache_copies_per_epoch}")
    print(
        "Augmentation: "
        f"elastic_p={args.elastic_prob}, elastic_alpha={args.elastic_alpha}, "
        f"elastic_sigma={args.elastic_sigma}, noise_p={args.gaussian_noise_prob}, "
        f"noise_std={args.gaussian_noise_std}"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cached_samples: list[dict[str, object]] = []
    start = time.time()
    try:
        for sample_idx, sample in enumerate(tqdm(train_samples, desc=f"Building KSDD1 {args.model_cache} cache pool")):
            for copy_idx in range(args.copies):
                img_np, mask_np = build_augmented_arrays(sample, sample_idx, copy_idx, image_size, args)
                cached_samples.append(
                    save_cached_sample(img_np, mask_np, tmp_root, sample, sample_idx, copy_idx, args)
                )

        elapsed = time.time() - start
        write_manifest(tmp_root, config, cached_samples, elapsed, args)
        output_root = finalize_output_dir(tmp_root, output_root)
        tmp_root = None
    except Exception:
        if tmp_root is not None and tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    print(f"\nWrote {len(cached_samples)} cached samples to {output_root}")

    if args.zip:
        remove_stale_zip(output_root)
        archive_base = output_root.parent / output_root.name
        zip_path = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
        print(f"Created archive: {zip_path}")

    print(f"Done in {elapsed / 60:.1f} min")
    print_notebook_settings(args)


if __name__ == "__main__":
    main()
