import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).parents[1]

ASSETS_PATH = BASE_DIR / "assets"

DATA_PATH = BASE_DIR / "data"

TRAIN_VAL_DATA_PATH = DATA_PATH / "train_val.csv"

TRAIN_CLEANED_DATA_PATH = DATA_PATH / "train_cleaned.csv"

VALIDATION_DATA_PATH = DATA_PATH / "validation.csv"

TEST_DATA_PATH = DATA_PATH / "test.csv"

FULL_DATASET_PATH = DATA_PATH / "full_dataset.csv"

CLEANED_TEST_DATA_PATH = DATA_PATH / "cleaned_test.csv"

CLEANED_TRAIN_DATA_PATH = DATA_PATH / "cleaned_train.csv"

CLEANED_FULL_DATASET_PATH = DATA_PATH / "cleaned_full_dataset.csv"

CLEANED_PRODUCTS_PATH = DATA_PATH / "cleaned_products.csv"

CLEANED_ACTUAL_CLASSES_PATH = DATA_PATH / "cleaned_actual_classes.csv"

CLEANED_CLASSES_PATH = DATA_PATH / "cleaned_classes.csv"

GPC_PATH = DATA_PATH / "gpc.xlsx"

GPC_TRAIN_PATH = DATA_PATH / "gpc_train.csv"

CLEANED_GPC_PATH = DATA_PATH / "cleaned_gpc.csv"

LABELED_GPC_PATH = DATA_PATH / "gpc_labeled.csv"

LABELED_GPC_PRED_PATH = DATA_PATH / "gpc_labeled_pred.csv"

SAMPLE_PRODUCTS_PATH = DATA_PATH / "products_sample.csv"

LABELED_PRODUCTS_PATH = DATA_PATH / "labeled_products.csv"

PRODUCT_TEST_EMBEDDINGS_PATH = DATA_PATH / "product_test_embeddings.csv"

PRODUCT_TRAIN_EMBEDDINGS_PATH = DATA_PATH / "product_train_embeddings.csv"

PRODUCT_FULL_DATASET_EMBEDDINGS_PATH = DATA_PATH / "product_full_dataset_embeddings.csv"

PRODUCT_FULL_DATASET_EMBEDDINGS_QWEN_PATH  = DATA_PATH / "product_full_dataset_embeddings_qwen.csv"

CLASS_EMBEDDINGS_PATH = DATA_PATH / "class_embeddings.csv"

CLASS_EMBEDDINGS_PATH_QWEN = DATA_PATH / "class_embeddings_qwen.csv"

CONFIG_PATH = BASE_DIR / "config"

E5_LARGE_INSTRUCT_CONFIG_PATH = CONFIG_PATH / "e5_large_instruct_config.json"

OPUS_TRANSLATION_CONFIG_PATH = CONFIG_PATH / "opus_translation_config.json"

TFIDF_CLASSIFIER_CONFIG_PATH = CONFIG_PATH / "tfidf_classifier_config.json"

GPC_EMBEDDING_XGB_CONFIG_PATH = CONFIG_PATH / "gpc_embedding_xgb_config.json"

GPC_HIERARCHICAL_CLASSIFIER_CONFIG = CONFIG_PATH / "gpc_hierarchical_classifier_config.json"

ENV_PATH = CONFIG_PATH / ".env"

MODEL_PATH = BASE_DIR / "models"

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}