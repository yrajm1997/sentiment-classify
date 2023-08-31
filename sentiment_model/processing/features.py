from tensorflow.keras.preprocessing.text import Tokenizer
from sentiment_model.config.core import config

# Performing the data augmentation as series of transformations
def get_tokenizer(num_words):
    # Tokenization
    tokenizer = Tokenizer(num_words= num_words)    
    return tokenizer
