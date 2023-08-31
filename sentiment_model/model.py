import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

from sentiment_model.config.core import config

# Create a function that returns a model
def create_model(max_review_length, top_words, embedding_vector_length, mask_zero, lstm_layer_dim, activation, dense_layer_dim, optimizer, loss, metrics):

    # Input layer
    inputs = Input(shape=(max_review_length,))
    # Embedding layer
    emb = Embedding(input_dim = top_words, output_dim = embedding_vector_length, mask_zero= mask_zero)(inputs)
    # LSTM layer
    lstm = LSTM(lstm_layer_dim)(emb)
    # Output layer
    outputs = Dense(dense_layer_dim, activation=activation)(lstm)
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


# Create model
classifier = create_model(max_review_length = config.model_config.max_review_length, 
                          top_words = config.model_config.top_words, 
                          embedding_vector_length = config.model_config.embedding_vector_length, 
                          mask_zero = config.model_config.mask_zero, 
                          lstm_layer_dim = config.model_config.lstm_layer_dim, 
                          activation = config.model_config.activation, 
                          dense_layer_dim = config.model_config.dense_layer_dim, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric])
