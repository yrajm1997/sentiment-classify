import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_model.config.core import config
from sentiment_model.model import classifier
from sentiment_model.processing.data_manager import load_dataset, save_tokenizer, callbacks_and_save_model
from sentiment_model.processing.features import get_tokenizer

def run_training() -> None:
    
    """
    Train the model.
    """
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
    
    # Tokenization
    tokenizer = get_tokenizer(num_words= config.model_config.num_words)
    tokenizer.fit_on_texts(X_train)
    
    # Save tokenizer
    save_tokenizer(tokenizer_to_persist = tokenizer)
    
    # Convert train, validation, and test text to sequences of numbers
    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_val_tok = tokenizer.texts_to_sequences(X_val)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
    
    # Pad the sequences with zeros and make all of the same dimension
    X_train_pad = pad_sequences(X_train_tok, 
                                maxlen = config.model_config.max_review_length, 
                                padding=config.model_config.padding_type, 
                                truncating=config.model_config.truncating_type)

    X_val_pad = pad_sequences(X_val_tok, 
                              maxlen = config.model_config.max_review_length, 
                              padding=config.model_config.padding_type, 
                              truncating=config.model_config.truncating_type)

    X_test_pad = pad_sequences(X_test_tok, 
                               maxlen = config.model_config.max_review_length, 
                               padding=config.model_config.padding_type,
                               truncating=config.model_config.truncating_type)


    # Model fitting
    classifier.fit(X_train_pad, y_train,
                   epochs=config.model_config.epochs,
                   batch_size=config.model_config.batch_size,
                   validation_data = (X_val_pad, y_val),
                   callbacks = callbacks_and_save_model(),
                   #verbose = config.model_config.verbose
                   )
    
    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(X_test_pad, y_test)
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

    
if __name__ == "__main__":
    run_training()