# Downloading Datasets (`data/` is git-ignored)

Do **not** commit dataset archives or images to Git. Keep all datasets under `data/` locally (or in Google Drive `/MyDrive/Anomaly_inspect/data/` when using Colab via `notebooks/colab_setup.py`).

## 1. MVTec AD (Primary, ~5 GB)

- **License:** CC BY-NC-SA 4.0 (non-commercial research/educational use).
- **Download:** Obtain the official archive from the [MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) (or an official mirror / Kaggle mirror if the direct download requires registration). Note that Anomalib `2.7.1`'s automatic MVTec downloader currently fails upstream (HTTP 404), so download manually.
- **Expected layout:**

```text
data/mvtec_ad/
├── bottle/
│   ├── train/good/*.png
│   ├── test/
│   │   ├── good/*.png
│   │   ├── broken_large/*.png
│   │   └── ...
│   └── ground_truth/
│       └── broken_large/*_mask.png
├── cable/
... (15 categories total)
```

15 categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

## 2. VisA (Secondary, ~10 GB)

- **License:** Check the current license in the [official VisA / Amazon spot-diff repository](https://github.com/amazon-science/spot-diff) before use or redistribution.
- **Download:** Download `VisA_20220922.tar` from the official repository link (`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`) and extract into `data/visa/`.
- **Expected layout (either official `1cls.csv` or MVTec-converted `visa_pytorch/`):**

```text
data/visa/
├── split_csv/
│   └── 1cls.csv
├── candle/
│   └── Data/
│       ├── Images/{Normal,Anomaly}/*
│       └── Masks/Anomaly/*.png
... (12 categories total)
```

12 categories: `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, `macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum`.

## 3. Verify Loaded Counts

After placing a dataset under `data/`, verify discovery and counts with:

```bash
.venv/bin/python -m research.data.datasets --dataset mvtec --root data/mvtec_ad
.venv/bin/python -m research.data.datasets --dataset visa --root data/visa
```
