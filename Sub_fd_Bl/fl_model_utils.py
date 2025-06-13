# fl_model_utils.py
import json
import hashlib
import numpy as np
# import tensorflow as tf # TensorFlow is imported within the model function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# --- New LSTM-GRU-Attention Model (Now the only model) ---
def build_lstm_gru_attention_model(input_shape, num_classes):
    """
    Defines LSTM-GRU Model with Multi-Head Attention.
    input_shape: e.g., (1, num_tfidf_features)
    num_classes: Number of output classes for classification.
    """
    inputs = Input(shape=input_shape) 

    lstm_output = LSTM(256, return_sequences=True)(inputs)
    gru_output = GRU(192, return_sequences=True)(lstm_output)

    attention_output = MultiHeadAttention(num_heads=2, key_dim=96)(query=gru_output, value=gru_output, key=gru_output)
    attention_output = Add()([attention_output, gru_output]) 
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    pooled_output = GlobalAveragePooling1D()(attention_output)
    dense_output = Dense(64, activation='relu')(pooled_output)
    dense_output = Dropout(0.3)(dense_output)
    outputs = Dense(num_classes, activation='softmax')(dense_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- Utility Functions (General for Keras models) ---
def get_model_weights_keras(model):
    return model.get_weights()

def set_model_weights_keras(model, weights):
    model.set_weights(weights)

def digest_model_weights_keras(weights_list):
    serializable_weights = [w.tolist() for w in weights_list]
    serialized_weights = json.dumps(serializable_weights, sort_keys=True).encode()
    return hashlib.sha256(serialized_weights).hexdigest()

def calculate_weights_l2_norm(weights_list):
    # Keras get_weights returns a list of ndarrays (weights and biases for each layer)
    norm = 0.0
    for w_array in weights_list: 
        norm += np.linalg.norm(w_array)**2
    return np.sqrt(norm)

def calculate_update_l2_norm(global_weights, client_weights):
    norm = 0.0
    for g_w, c_w in zip(global_weights, client_weights):
        diff = c_w - g_w
        norm += np.linalg.norm(diff)**2
    return np.sqrt(norm)