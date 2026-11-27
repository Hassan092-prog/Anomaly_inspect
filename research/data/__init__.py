"""Dataset wrappers and deterministic split generators."""

from research.data.datasets import (
    DATASET_CATEGORIES,
    MVTEC_CATEGORIES,
    VISA_CATEGORIES,
    AnomalyImageDataset,
    DatasetError,
    SampleRecord,
    category_counts,
    load_dataset_samples,
    load_mvtec_samples,
    load_visa_samples,
    preprocess_image_and_mask,
)
from research.data.make_splits import (
    CategorySplit,
    LeakageError,
    SplitError,
    assert_no_test_leakage,
    create_category_splits,
    load_split_records,
    sample_k_shot,
    split_train_val,
)

__all__ = [
    "AnomalyImageDataset",
    "CategorySplit",
    "DATASET_CATEGORIES",
    "DatasetError",
    "LeakageError",
    "MVTEC_CATEGORIES",
    "SampleRecord",
    "SplitError",
    "VISA_CATEGORIES",
    "assert_no_test_leakage",
    "category_counts",
    "create_category_splits",
    "load_dataset_samples",
    "load_mvtec_samples",
    "load_split_records",
    "load_visa_samples",
    "preprocess_image_and_mask",
    "sample_k_shot",
    "split_train_val",
]
