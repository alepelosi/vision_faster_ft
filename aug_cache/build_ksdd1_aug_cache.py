from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]

DEFAULT_RESNET_CACHE_NAME = "ksdd1_aug_cache"
DEFAULT_SEGFORMER_CACHE_NAME = "ksdd1_segformer_aug_cache"
DEFAULT_IMAGE_SIZE = (512, 1408)


@dataclass(frozen=True)
class CacheTarget:
    name: str
    output_root: Path
    manifest_cache_root: str


def load_runtime_deps() -> None:
    global np, torch, TF, Image, ElasticTransform, InterpolationMode, tqdm

    try:
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        from torchvision.transforms import ElasticTransform, InterpolationMode
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency while building the cache. Install the augmentation "
            "requirements first, for example: pip install torch torchvision pillow numpy tqdm"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create preprocessed augmentation cache(s) for KolektorSDD1."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("KolektorSDD-boxes"),
        help="Local KolektorSDD1 folder containing kosXX folders.",
    )
    parser.add_argument(
        "--cache-kind",
        choices=("resnet", "segformer", "both"),
        default="both",
        help="Which notebook cache format/path to create.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override output folder when --cache-kind is resnet or segformer. "
            "Ignored for --cache-kind both."
        ),
    )
    parser.add_argument(
        "--resnet-output-root",
        type=Path,
        default=Path(DEFAULT_RESNET_CACHE_NAME),
        help="Local folder for the ResNet side-tuning cache.",
    )
    parser.add_argument(
        "--segformer-output-root",
        type=Path,
        default=Path(DEFAULT_SEGFORMER_CACHE_NAME),
        help="Local folder for the SegFormer side-tuning cache.",
    )
    parser.add_argument(
        "--manifest-cache-root",
        default=None,
        help=(
            "Override manifest cache root when --cache-kind is resnet or segformer. "
            "Use the final Colab path, for example /content/ksdd1_aug_cache."
        ),
    )
    parser.add_argument(
        "--resnet-manifest-cache-root",
        default="/content/ksdd1_aug_cache",
        help="Path the ResNet cache will have in Colab.",
    )
    parser.add_argument(
        "--segformer-manifest-cache-root",
        default="/content/ksdd1_segformer_aug_cache",
        help="Path the SegFormer cache will have in Colab.",
    )
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--elastic-prob", type=float, default=0.6)
    parser.add_argument("--elastic-alpha", type=float, default=45.0)
    parser.add_argument("--elastic-sigma", type=float, default=5.0)
    parser.add_argument("--gaussian-noise-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.02)
    parser.add_argument("--compress-level", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache folder(s).")
    zip_group = parser.add_mutually_exclusive_group()
    zip_group.add_argument(
        "--zip",
        dest="zip",
        action="store_true",
        help="Create .zip archive(s). This is enabled by default.",
    )
    zip_group.add_argument(
        "--no-zip",
        dest="zip",
        action="store_false",
        help="Only create cache folder(s), without zip archive(s).",
    )
    parser.set_defaults(zip=True)
    return parser.parse_args()


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
        root / "KolektorSDD2",
    ]
    for candidate in candidates:
        if candidate.exists() and (has_train_test(candidate) or has_sdd1_groups(candidate)):
            return candidate

    for candidate in root.rglob("*"):
        if candidate.is_dir() and (has_train_test(candidate) or has_sdd1_groups(candidate)):
            return candidate

    raise FileNotFoundError(f"Could not find KolektorSDD1 kosXX folders under: {root}")


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
    img_path = Path(sample["image"])
    mask_path = sample["mask"]

    img = Image.open(img_path).convert("RGB")
    if mask_path is None:
        mask = Image.new("L", img.size, 0)
    else:
        mask = Image.open(Path(mask_path)).convert("L")
        if mask.size != img.size:
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
    target: CacheTarget,
    image_name: str,
    mask_name: str,
    sample_idx: int,
    copy_idx: int,
) -> dict[str, object]:
    image_path = colab_cache_path(target.manifest_cache_root, "train", image_name)
    mask_path = colab_cache_path(target.manifest_cache_root, "train", mask_name)
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


def build_targets(args: argparse.Namespace) -> list[CacheTarget]:
    if args.cache_kind == "resnet":
        output_root = args.output_root or args.resnet_output_root
        manifest_root = args.manifest_cache_root or args.resnet_manifest_cache_root
        return [CacheTarget("resnet", output_root, manifest_root)]
    if args.cache_kind == "segformer":
        output_root = args.output_root or args.segformer_output_root
        manifest_root = args.manifest_cache_root or args.segformer_manifest_cache_root
        return [CacheTarget("segformer", output_root, manifest_root)]
    return [
        CacheTarget("resnet", args.resnet_output_root, args.resnet_manifest_cache_root),
        CacheTarget("segformer", args.segformer_output_root, args.segformer_manifest_cache_root),
    ]


def prepare_output_dirs(targets: list[CacheTarget], force: bool) -> None:
    seen_roots: set[Path] = set()
    for target in targets:
        output_root = target.output_root.expanduser().resolve()
        if output_root in seen_roots:
            raise ValueError(f"Duplicate output root requested: {output_root}")
        seen_roots.add(output_root)

        if output_root.exists():
            if not force:
                raise FileExistsError(f"{output_root} already exists. Use --force to overwrite it.")
            shutil.rmtree(output_root)
        (output_root / "train").mkdir(parents=True, exist_ok=True)


def save_augmented_arrays(
    img_np: np.ndarray,
    mask_np: np.ndarray,
    targets: list[CacheTarget],
    sample: dict[str, object],
    sample_idx: int,
    copy_idx: int,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    stem = f"{sample_idx:05d}_aug{copy_idx:02d}"
    image_name = f"{stem}.png"
    mask_name = f"{stem}_GT.png"
    rows_by_target: dict[str, list[dict[str, object]]] = {}

    for target in targets:
        output_root = target.output_root.expanduser().resolve()
        Image.fromarray(img_np).save(output_root / "train" / image_name, compress_level=args.compress_level)
        Image.fromarray(mask_np).save(output_root / "train" / mask_name, compress_level=args.compress_level)
        rows_by_target.setdefault(target.name, []).append(
            manifest_row(sample, target, image_name, mask_name, sample_idx, copy_idx)
        )

    return rows_by_target


def main() -> None:
    args = parse_args()
    load_runtime_deps()

    if args.copies < 1:
        raise ValueError("--copies must be at least 1")
    if not 0 <= args.compress_level <= 9:
        raise ValueError("--compress-level must be in [0, 9]")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    args.image_size = image_size

    data_root = resolve_data_root(args.data_root)
    targets = build_targets(args)
    prepare_output_dirs(targets, args.force)

    train_samples = load_train_samples(data_root, args)
    config = preprocess_cache_config(len(train_samples), args)

    print(f"Dataset root: {data_root}")
    print(f"Image size: {image_size}")
    print(f"Copies per sample: {args.copies}")
    for target in targets:
        print(f"{target.name} cache: {target.output_root.expanduser().resolve()}")
        print(f"{target.name} manifest root: {target.manifest_cache_root}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows_by_target = {target.name: [] for target in targets}
    start = time.time()
    for sample_idx, sample in enumerate(tqdm(train_samples, desc="Building KSDD1 cache")):
        for copy_idx in range(args.copies):
            img, mask = load_sample_pil(sample, image_size)
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

            new_rows = save_augmented_arrays(
                img_np,
                mask_np,
                targets,
                sample,
                sample_idx,
                copy_idx,
                args,
            )
            for target_name, rows in new_rows.items():
                rows_by_target[target_name].extend(rows)

    elapsed = time.time() - start
    for target in targets:
        output_root = target.output_root.expanduser().resolve()
        manifest = {
            "config": config,
            "samples": rows_by_target[target.name],
            "elapsed_seconds": elapsed,
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{target.name}: wrote {len(rows_by_target[target.name])} cached samples")

        if args.zip:
            archive_base = output_root.parent / output_root.name
            zip_path = shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=output_root.parent,
                base_dir=output_root.name,
            )
            print(f"{target.name}: created archive {zip_path}")

    print(f"Done in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
