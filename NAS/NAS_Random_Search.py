import logging
import random
import re
import time
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (LSTM, GRU, Add, Dense, Dropout,
                                     GlobalAveragePooling1D, Input,
                                     LayerNormalization, MultiHeadAttention)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Data Loading and Preprocessing Functions ---

def ensure_nltk_resources():
    """Downloads required NLTK data if not found."""
    resources = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt")]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"NLTK resource '{resource_name}' not found. Downloading...")
            nltk.download(resource_name, quiet=True)

def clean_text(text):
    """Cleans raw text data for processing."""
    text = str(text)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|bit.ly/\S+', '', text)  # Remove URLs
    text = re.sub(r'[\w.-]+@[\w.-]+', '', text)  # Remove email addresses
    text = re.sub(r'[^a-zA-Z.,!?/:;"\'\s]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'[_]+', '', text)
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    except Exception:
        logger.warning("Could not perform stopword removal.")
    return text.lower()

def load_and_prepare_data(data_path='Sample_of_Dermo_Questions.csv', max_features=1500):
    """Loads, preprocesses, and splits the dataset."""
    logger.info(f"Loading and preparing data from '{data_path}'...")
    ensure_nltk_resources()
    try:
        df = pd.read_csv(data_path)
        df['Question'] = df['Question'].fillna('').astype(str)
        df['Label'] = df['Label'].fillna('').astype(str)
        df['cleaned_Question'] = df['Question'].apply(clean_text)

        label_encoder = LabelEncoder()
        df['drug_encoded'] = label_encoder.fit_transform(df['Label'])
        num_classes = len(label_encoder.classes_)

        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        x_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_Question']).toarray()
        y_encoded = df['drug_encoded'].values

        # Reshape for LSTM/GRU input: (samples, timesteps, features)
        x_reshaped = np.expand_dims(x_tfidf, axis=1)

        x_train, x_val, y_train, y_val = train_test_split(
            x_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Data prepared. Training set: {x_train.shape}, Validation set: {x_val.shape}")
        logger.info(f"Number of classes: {num_classes}")
        
        return x_train, x_val, y_train, y_val, num_classes
    except FileNotFoundError:
        logger.error(f"CRITICAL: Data file not found at '{data_path}'. Please ensure it is in the correct directory.")
        return None, None, None, None, None

# --- Neural Architecture Search (NAS) Components ---

def get_hyperparameter_search_space():
    """Defines the search space for the model architecture."""
    return {
        'lstm_units': [128, 192, 256, 320],
        'gru_units': [96, 128, 192, 256],
        'attention_heads': [2, 4, 8],
        'attention_key_dim': [64, 96, 128],
        'dense_units': [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }

def sample_random_hyperparameters(search_space):
    """Randomly samples a set of hyperparameters from the defined space."""
    return {key: random.choice(values) for key, values in search_space.items()}

def build_dynamic_model(input_shape, num_classes, params):
    """
    Builds a Keras model based on a given set of hyperparameters.
    
    Args:
        input_shape (tuple): The shape of the input data (e.g., (1, 1500)).
        num_classes (int): The number of output classes.
        params (dict): A dictionary of hyperparameters for the model.
    """
    inputs = Input(shape=input_shape)

    # Core RNN layers
    lstm_output = LSTM(params['lstm_units'], return_sequences=True)(inputs)
    gru_output = GRU(params['gru_units'], return_sequences=True)(lstm_output)

    # Attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=params['attention_heads'],
        key_dim=params['attention_key_dim']
    )(query=gru_output, value=gru_output, key=gru_output)
    
    attention_output = Add()([attention_output, gru_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Final classifier head
    pooled_output = GlobalAveragePooling1D()(attention_output)
    dense_output = Dense(params['dense_units'], activation='relu')(pooled_output)
    dense_output = Dropout(params['dropout_rate'])(dense_output)
    outputs = Dense(num_classes, activation='softmax')(dense_output)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_nas_random_search(num_models_to_test=20, epochs_per_model=10):
    """
    Orchestrates the Neural Architecture Search process.
    """
    logger.info("--- Starting Neural Architecture Search with Random Search ---")
    
    # 1. Load Data
    x_train, x_val, y_train, y_val, num_classes = load_and_prepare_data()
    if x_train is None:
        return

    # 2. Define Search Space
    search_space = get_hyperparameter_search_space()
    
    best_accuracy = 0.0
    best_hyperparameters = None
    results_history = []
    
    input_shape = x_train.shape[1:]

    # 3. Run the search loop
    for i in range(num_models_to_test):
        start_time = time.time()
        
        # Sample a random architecture
        current_params = sample_random_hyperparameters(search_space)
        logger.info(f"\n--- [Model {i+1}/{num_models_to_test}] Testing Architecture ---")
        logger.info(f"Hyperparameters: {current_params}")
        
        # Build the model
        model = build_dynamic_model(input_shape, num_classes, current_params)
        
        # Train the model
        logger.info(f"Training for {epochs_per_model} epochs...")
        history = model.fit(
            x_train, y_train,
            epochs=epochs_per_model,
            validation_data=(x_val, y_val),
            batch_size=32,
            verbose=0  # Set to 1 or 2 for more detailed logs during training
        )
        
        # Evaluate the model
        val_accuracy = max(history.history['val_accuracy']) * 100
        training_time = time.time() - start_time
        
        logger.info(f"Finished Training. Time: {training_time:.2f}s. Max Validation Accuracy: {val_accuracy:.2f}%")
        
        results_history.append({'id': i+1, 'accuracy': val_accuracy, 'params': current_params})
        
        # Check for new best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hyperparameters = current_params
            logger.info(f"*** New Best Architecture Found! Accuracy: {best_accuracy:.2f}% ***")
            
    # 4. Print Final Results
    logger.info("\n\n--- NAS Random Search Complete ---")
    if best_hyperparameters:
        logger.info(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        logger.info("Best Hyperparameters Found:")
        for key, value in best_hyperparameters.items():
            logger.info(f"  - {key}: {value}")
    else:
        logger.warning("No successful training runs were completed.")

if __name__ == "__main__":
    # --- Configuration ---
    # Number of different random architectures to sample and test
    NUM_MODELS_TO_TEST = 25 
    # Number of epochs to train each architecture for
    EPOCHS_PER_MODEL = 15

    run_nas_random_search(
        num_models_to_test=NUM_MODELS_TO_TEST,
        epochs_per_model=EPOCHS_PER_MODEL
    )
