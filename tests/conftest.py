import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.processing.data_manager import load_dataset, load_tokenizer

tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
tokenizer = load_tokenizer(file_name = tokenizer_file_name)

@pytest.fixture
def sample_input_data():
    
    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    # divide training set and rest
    X_train, X_validation, y_train, y_validation = train_test_split(
        
        data[config.model_config.reviewtext_var].values,     # predictors
        data[config.model_config.target].values,             # target
        test_size = config.model_config.test_size,
        random_state=config.model_config.random_state,   # set the random seed here for reproducibility
        shuffle=config.model_config.shuffle
    )
    
    # divide validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        
        X_validation,
        y_validation,
        test_size = config.model_config.val_size,
        random_state=config.model_config.random_state,
        shuffle=config.model_config.shuffle
    )
    
    # Convert train, validation, and test text to sequences of numbers
    X_test_tok = tokenizer.texts_to_sequences(X_test)
    
    # Pad the sequences with zeros and make all of the same dimension
    X_test_pad = pad_sequences(X_test_tok, 
                               maxlen = config.model_config.max_review_length, 
                               padding=config.model_config.padding_type,
                               truncating=config.model_config.truncating_type)
    
    return X_test_pad, y_test