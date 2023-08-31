# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import sentiment_model

# Project Directories
PACKAGE_ROOT = Path(sentiment_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
TRAINED_TOKENIZER_DIR = PACKAGE_ROOT / "trained_tokenizer"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    model_name: str
    model_save_file: str
    tokenizer_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
        
    target: str
    unused_fields: List[str]
    
    score_var: str
    sentiment_var: str
    reviewtext_var: str
    sentiment_mappings: Dict[int, str]
    
    test_size: float
    val_size: float
    shuffle: bool
    random_state: int
    
    num_words: int
    max_review_length: int
    truncating_type: str
    padding_type: str

    top_words: int
    embedding_vector_length: int
    mask_zero: bool
    lstm_layer_dim: int
    dense_layer_dim: int
    activation: str
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    accuracy_metric: str
    verbose: int
    earlystop: int
    monitor: str
    save_best_only: bool


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()