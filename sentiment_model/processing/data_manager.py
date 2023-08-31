import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path

import pandas as pd
import re
import io
import json
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from sentiment_model.config.core import config
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, TRAINED_TOKENIZER_DIR, config

##  Pre-Pipeline Preparation

# Extract year and month from the date column and create two another columns

def get_sentiment(dataframe: pd.DataFrame, sentiment_var: str, score_var: str):

    df = dataframe.copy()
    # Add new features 'Sentiment'
    df[sentiment_var] = df[score_var].apply(lambda x: 0 if x<3 else 1)        
    return df


def strip_html(text: str):
    text = re.sub('<.*?>', ' ', text)
    text = re.sub('\s+\s', ' ', text)
    text = text.strip()
    return text


def remove_punctuations(text: str):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text


def get_stopwords():    

    nltk.download('stopwords')
    nltk.download('punkt')
    # setting english stopwords
    stopword_list = nltk.corpus.stopwords.words('english')
    # Exclude 'not' and its other forms from the stopwords list
    updated_stopword_list = []
    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
            pass
        else:
            updated_stopword_list.append(word)

    return updated_stopword_list

updated_stopword_list = get_stopwords()


def remove_stopwords(text, is_lower_case=False):
    
    # splitting strings into tokens (list of words)
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in updated_stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in updated_stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text


def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:

    # Add 'Sentiment' column
    df = get_sentiment(dataframe = df, sentiment_var = config.model_config.sentiment_var, score_var = config.model_config.score_var)
    # Remove duplicates considering 'Sentiment' and 'Text' columns
    df = df.drop_duplicates(subset=[config.model_config.sentiment_var, config.model_config.reviewtext_var])
    # Remove the html strips
    df[config.model_config.reviewtext_var] = df[config.model_config.reviewtext_var].apply(strip_html)  
    # Remove punctuations
    df[config.model_config.reviewtext_var] = df[config.model_config.reviewtext_var].apply(remove_punctuations)
    # Remove stopwords
    print("Removing stopwords...")
    df[config.model_config.reviewtext_var] = df[config.model_config.reviewtext_var].apply(remove_stopwords)
    print("Stopwords removed!")
    # Drop unnecessary fields
    for field in config.model_config.unused_fields:
        if field in df.columns:
            df.drop(labels = field, axis=1, inplace=True)    
    
    return df


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    #dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe = pd.read_csv(file_name)
    transformed = pre_pipeline_preparation(df = dataframe)

    return transformed


# Define a function to return a commmonly used callback_list
def callbacks_and_save_model():
    callback_list = []
    
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep = [save_file_name])

    # Default callback
    callback_list.append(ModelCheckpoint(filepath = str(save_path),
                                                         save_best_only = config.model_config.save_best_only,
                                                         monitor = config.model_config.monitor))

    if config.model_config.earlystop > 0:
        callback_list.append(EarlyStopping(patience = config.model_config.earlystop))

    return callback_list


def load_model(*, file_name: str) -> models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = models.load_model(filepath = file_path)
    return trained_model


def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def remove_old_tokenizer(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old tokenizer.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_TOKENIZER_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_tokenizer(*, tokenizer_to_persist: Tokenizer):
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
    save_path = TRAINED_TOKENIZER_DIR / save_file_name
    
    remove_old_tokenizer(files_to_keep = [save_file_name])

    tokenizer_json = tokenizer_to_persist.to_json()    
    with io.open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def load_tokenizer(*, file_name: str):
    
    file_path = TRAINED_TOKENIZER_DIR / file_name
    with open(file_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer