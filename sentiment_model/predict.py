import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_model, load_tokenizer

model_file_name = f"{config.app_config.model_save_file}{_version}"
clf_model = load_model(file_name = model_file_name)

tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
tokenizer = load_tokenizer(file_name = tokenizer_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, list, dict, tf.Tensor]) -> dict:
    """Make a prediction using a saved model """
    
    results = {"predictions": None, "version": _version}
    
    predictions = clf_model.predict(input_data, verbose = 0)
    pred_labels = []
    for i in predictions:
        pred_labels.append(config.model_config.sentiment_mappings[int(predictions + 0.5)])
        
    results = {"predictions": pred_labels, "version": _version}
    print(results)

    return results


if __name__ == "__main__":

    text = 'I liked the product. It was great in taste and very enjoyful.'   
    text = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text, 
                             maxlen = config.model_config.max_review_length, 
                             padding= config.model_config.padding_type, 
                             truncating= config.model_config.truncating_type)

    make_prediction(input_data = text_pad)
