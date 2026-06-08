# KolektorSDD Fine-Tuning Experiments

This repository contains the notebooks and cache-building scripts used for crack/defect segmentation experiments on KolektorSDD2 and KolektorSDD1. The main comparison is between full fine-tuning, parameter-efficient fine-tuning, and side-tuning under a domain-shift setting.

The notebooks are designed for Google Colab with a CUDA GPU. Local scripts are included to precompute expensive augmentation caches before running the notebooks.

## Repository Layout

```text
.
|-- KolektorSDD2_ResNet18_UNet.ipynb
|-- KolektorSDD2_SegformerB0.ipynb
|-- SideTuning/
|   |-- KolektorSDD2_ResNet18_UNet_SideTuning.ipynb
|   |-- KolektorSDD2_SegformerB0_SideTuning.ipynb
|   |-- KolektorSDD1_ResNet18_UNet_SideTuning.ipynb
|   `-- KolektorSDD1_SegformerB0_SideTuning.ipynb
|-- build_ksdd2_aug_cache.py
|-- build_ksdd1_aug_cache.py
|-- build_ksdd1_cache_pool.py
|-- requirements.txt
`-- .gitignore
```

Local datasets, generated caches, cache zip files, virtual environments, and macOS metadata are ignored by Git.

## What Each File Does

`KolektorSDD2_ResNet18_UNet.ipynb`
: ResNet18 encoder with a UNet decoder for KolektorSDD2. Available strategies are `full_ft`, `decoder_only`, `convlora_decoder`, `staged`, `ssf_decoder`and `last_block_decoder`.

`KolektorSDD2_SegformerB0.ipynb`
: SegFormer-B0 for KolektorSDD2. Available strategies are `full_ft`, `decoder_only`, `lora`, `adaptformer`, and `ssf`.

`SideTuning/KolektorSDD2_ResNet18_UNet_SideTuning.ipynb`
: ResNet18-UNet side-tuning on KolektorSDD2. The frozen base model is loaded from an SDD2 checkpoint and a lightweight side branch is trained.

`SideTuning/KolektorSDD2_SegformerB0_SideTuning.ipynb`
: SegFormer-B0 side-tuning on KolektorSDD2. Available strategies are `side_tuning` and `side_tuning_last_stage`.

`SideTuning/KolektorSDD1_ResNet18_UNet_SideTuning.ipynb`
: ResNet18-UNet side-tuning on KolektorSDD1 to simulate domain shift from an SDD2-trained base model.

`SideTuning/KolektorSDD1_SegformerB0_SideTuning.ipynb`
: SegFormer-B0 side-tuning on KolektorSDD1 to simulate the same domain shift experiment.

`build_ksdd2_aug_cache.py`
: Precomputes augmented KolektorSDD2 training samples into a manifest-backed cache compatible with the SDD2 notebooks.

`build_ksdd1_cache_pool.py`
: Standalone script for the KSDD1 side-tuning cache-pool experiments. Use `--resnet` or `--segformer` to build the matching cache. It builds 24 pre-augmented copies per source image locally and records 2 copies per epoch as the intended runtime draw size.

`requirements.txt`
: Minimal local dependencies for the augmentation cache scripts.

## Datasets

The notebooks expect the datasets to be available in Google Drive or copied to Colab local disk.

KolektorSDD2 layout:

```text
KolektorSDD2/
|-- train/
|   |-- image.png
|   `-- image_GT.png
`-- test/
    |-- image.png
    `-- image_GT.png
```

KolektorSDD1 layout:

```text
KolektorSDD-boxes/
|-- kos01/
|   |-- Part0.jpg
|   `-- Part0_label.bmp
|-- kos02/
`-- ...
```

The `.gitignore` excludes `KolektorSDD2/` and `KolektorSDD-boxes/`, so the datasets should not be committed.

## Local Cache Generation

Create a local environment for the cache scripts:

```bash
cd "/path/to/vision_faster_ft"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Build the KolektorSDD2 cache:

```bash
python aug_cache/build_ksdd2_aug_cache.py \
  --data-root ./KolektorSDD2 \
  --force
```

This creates:

```text
ksdd2_aug_cache/
ksdd2_aug_cache.zip
```

Build the KolektorSDD1 ResNet cache pool:

```bash
python build_ksdd1_cache_pool.py \
  --resnet \
  --data-root ./KolektorSDD-boxes \
  --force
```

This creates:

```text
ksdd1_aug_cache/
ksdd1_aug_cache.zip
```

Build the KolektorSDD1 SegFormer cache pool:

```bash
python build_ksdd1_cache_pool.py \
  --segformer \
  --data-root ./KolektorSDD-boxes \
  --force
```

This creates:

```text
ksdd1_segformer_aug_cache/
ksdd1_segformer_aug_cache.zip
```

Use these cache-pool settings in the SDD1 side-tuning notebooks:

```python
USE_PREPROCESSED_TRAIN = True
AUGMENTED_COPIES_PER_SAMPLE = 24
CACHE_COPIES_PER_EPOCH = 2
```

Upload the zip files to Google Drive and unzip them in Colab so the folders exist at:

```text
/content/ksdd2_aug_cache
/content/ksdd1_aug_cache
/content/ksdd1_segformer_aug_cache
```

The notebooks can also build the cache online, but precomputing avoids paying the elastic/noise augmentation cost during training.

## Recommended Experiment Order

1. Run `KolektorSDD2_ResNet18_UNet.ipynb` with `strategy = "full_ft"` to train the ResNet source model.
2. Save or copy its best checkpoint to:

```text
/content/drive/MyDrive/best_ksdd2_full_ft.pth
```

3. Run `KolektorSDD2_SegformerB0.ipynb` with `strategy = "full_ft"` to train the SegFormer source model.
4. Save or copy its best checkpoint to:

```text
/content/drive/MyDrive/best_ksdd2_segformer_full_ft.pth
```

5. Run the SDD2 strategy comparisons in the two main notebooks.
6. Run the SDD2 side-tuning notebooks if you want side-tuning baselines on the source domain.
7. Run the SDD1 side-tuning notebooks to simulate domain shift from the frozen SDD2 base models to KolektorSDD1.

## Running The Notebooks

Open a notebook in Colab, enable a GPU runtime, and run cells from top to bottom.

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

The notebooks install their own Colab dependencies near the top. Keep `REQUIRE_CUDA = True` for the intended GPU workflow.

Common paths used by the notebooks:

```text
/content/drive/MyDrive/KolektorSDD2
/content/drive/MyDrive/KolektorSDD-boxes
/content/drive/MyDrive/best_ksdd2_full_ft.pth
/content/drive/MyDrive/best_ksdd2_segformer_full_ft.pth
/content/drive/MyDrive/results/<strategy>/
```

If the dataset is copied to local Colab disk, the notebooks use:

```text
/content/KolektorSDD2
/content/KolektorSDD-boxes
```

## Choosing Strategies

ResNet18-UNet SDD2 notebook:

```python
strategy = "full_ft"
```

Other useful ResNet strategy names include:

```text
decoder_only
staged_decoder
ssf_decoder
convlora_decoder
full_ft
```

SegFormer-B0 SDD2 notebook:

```python
strategy = "ssf"
```

Available SegFormer strategies:

```text
full_ft
decoder_only
lora
adaptformer
ssf
```

ResNet side-tuning notebooks:

```python
strategy = "side_tuning"
```

Available ResNet side-tuning strategies:

```text
side_tuning
side_tuning_l4
```

SegFormer side-tuning notebooks:

```python
strategy = "side_tuning"
```

Available SegFormer side-tuning strategies:

```text
side_tuning
side_tuning_last_stage
```

For fair comparison, keep the epoch count, threshold settings, loss settings, and cache configuration aligned across runs.

## Outputs

Each notebook trains a model, evaluates validation metrics, runs threshold/min-area sweeps, creates qualitative prediction plots, and saves artifacts.

The final export cell writes to:

```text
/content/drive/MyDrive/results/<strategy>/
```

Typical exported files include:

```text
epochs.csv
threshold_area_sweep.csv
best_threshold_area.csv
best_result.csv
learning curve PNGs
prediction PNGs
positive-only prediction PNGs
```

The training checkpoint is also saved by each notebook, usually with a name containing the dataset, model, and strategy.

## KolektorSDD1 Domain-Shift Notes

The SDD1 side-tuning notebooks use the older KolektorSDD preprocessing. The target image size is 512 x 1408 pixels in width x height notation. Because TorchVision resize expects height x width, the code uses:

```text
IMG_SIZE = (1408, 512)
```

They resize images and masks before augmentation, convert grayscale strips to RGB with PIL, estimate the foreground prior from SDD1 masks, use lower minimum-area thresholds for thin cracks, and keep the newer augmentation setup:

```text
horizontal flip
vertical flip
180 degree rotation
elastic deformation
Gaussian noise
brightness/contrast jitter in [0.9, 1.1]
```

The SDD1 side-tuning notebooks intentionally load SDD2 full fine-tuning checkpoints as frozen base models. This is the domain-shift setup: train on SDD2 first, then adapt only the lightweight side branch on SDD1.

## Repository Hygiene

Do not commit:

```text
KolektorSDD2/
KolektorSDD-boxes/
ksdd*_aug_cache/
ksdd*_aug_cache.zip
venv/
.DS_Store
```

These are already listed in `.gitignore`. Keep the repository focused on notebooks, scripts, configuration, and documentation.
