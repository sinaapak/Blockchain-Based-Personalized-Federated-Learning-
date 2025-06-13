# orchestration.py
import logging
import time
import threading
import random
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import requests
from cryptography.hazmat.primitives import serialization

from pbft_node import PBFTNode
from fl_server import FLServer
from fl_client import FLClient
from fl_model_utils import build_lstm_gru_attention_model

logger = logging.getLogger(__name__)

# --- clean_text function remains unchanged ---
def clean_text(text):
    text = str(text) 
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|bit.ly/\S+', '', text)
    text = re.sub(r'[\w.-]+@[\w.-]+', '', text)
    text = re.sub(r'[^a-zA-Z.,!?/:;"\'\s]', '', text)
    text = re.sub(r'[_]+', '', text)
    try:
        from nltk.corpus import stopwords as nltk_stopwords 
        stop_words = set(nltk_stopwords.words("english"))
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    except LookupError:
        logger.warning("NLTK stopwords not found. Skipping stopword removal.")
    except ImportError:
        logger.warning("nltk.corpus.stopwords not available. Skipping stopword removal.")
    text = text.lower()
    return text

class SimulationOrchestrator:
    # --- __init__ and other methods up to run_federated_learning remain unchanged ---
    def __init__(self, num_pbft_nodes, num_fl_clients, num_faulty_pbft, num_malicious_clients,
                 client_data_split_size=100, data_path='Combined_Serum_Questions.csv'):
        self.num_pbft_nodes = num_pbft_nodes
        self.num_fl_clients = num_fl_clients
        self.num_faulty_pbft = num_faulty_pbft
        self.num_malicious_clients = num_malicious_clients
        self.client_data_split_size = client_data_split_size
        self.data_path = data_path

        self.pbft_nodes = {}
        self.fl_clients = {}
        self.fl_server = None
        self.node_threads = []
        self.pbft_peers_config = {}
        self.current_pbft_primary_address = None

        self.label_encoder = None
        self.tfidf_vectorizer = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model_input_shape = None
        self.num_classes = None
        self.test_data_tuple = (None, None)

        self._load_and_preprocess_text_data()
        self.active_model_builder = build_lstm_gru_attention_model
    
    # --- All other methods from the original file are included here unchanged... ---
    # _download_nltk_resources, _load_and_preprocess_text_data, _get_current_pbft_primary_address,
    # setup_pbft_network, start_pbft_nodes, setup_fl_server, start_fl_server, 
    # setup_fl_clients, start_fl_clients, run_federated_learning...
    def _download_nltk_resources(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("NLTK 'stopwords' not found. Downloading...")
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("NLTK 'punkt' not found. Downloading...")
            nltk.download('punkt', quiet=True)

    def _load_and_preprocess_text_data(self):
        logger.info(f"Loading and preprocessing text data from: {self.data_path}")
        self._download_nltk_resources()
        try:
            df = pd.read_csv(self.data_path)
            df['Question'] = df['Question'].fillna('').astype(str)
            df['Label'] = df['Label'].fillna('').astype(str)
            df['cleaned_Question'] = df['Question'].apply(clean_text)
            self.label_encoder = LabelEncoder()
            df['drug_encoded'] = self.label_encoder.fit_transform(df['Label'])
            self.num_classes = len(self.label_encoder.classes_)
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1500)
            X_tfidf = self.tfidf_vectorizer.fit_transform(df['cleaned_Question']).toarray()
            y_encoded = df['drug_encoded'].values
            if X_tfidf.shape[0] == 0:
                raise ValueError("No data available after preprocessing.")
            stratify_param = y_encoded if self.num_classes > 1 and len(y_encoded) > self.num_classes / 0.2 else None
            X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded = train_test_split(
                X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
            )
            self.X_train = np.expand_dims(X_train_tfidf, axis=1)
            self.X_test = np.expand_dims(X_test_tfidf, axis=1)
            self.y_train = y_train_encoded
            self.y_test = y_test_encoded
            self.model_input_shape = (self.X_train.shape[1], self.X_train.shape[2]) 
            self.test_data_tuple = (self.X_test, self.y_test)
            logger.info(f"Text data prepared: X_train {self.X_train.shape}, y_train {self.y_train.shape}, X_test {self.X_test.shape}, y_test {self.y_test.shape}")
        except FileNotFoundError:
            logger.error(f"CRITICAL: Data file '{self.data_path}' not found.")
            raise
        except Exception as e:
            logger.error(f"CRITICAL: Error during text data preprocessing: {e}", exc_info=True)
            raise

    def _get_current_pbft_primary_address(self):
        if not self.pbft_nodes: return None
        node0 = self.pbft_nodes.get(0)
        if node0:
            current_primary_id = node0.current_view % self.num_pbft_nodes
            return self.pbft_peers_config.get(current_primary_id)
        return list(self.pbft_peers_config.values())[0] if self.pbft_peers_config else None

    def setup_pbft_network(self):
        logger.info(f"Setting up {self.num_pbft_nodes} PBFT nodes...")
        base_port = 6000
        for i in range(self.num_pbft_nodes):
            self.pbft_peers_config[i] = f'http://127.0.0.1:{base_port + i}'
        for i in range(self.num_pbft_nodes):
            self.pbft_nodes[i] = PBFTNode(node_id=i, N=self.num_pbft_nodes, f=self.num_faulty_pbft, host='127.0.0.1', port=base_port + i, peers=self.pbft_peers_config.copy())
        all_public_keys = {node_id: node.public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode() for node_id, node in self.pbft_nodes.items()}
        for node in self.pbft_nodes.values():
            node.public_keys = all_public_keys
        logger.info("PBFT nodes configured and public keys exchanged.")

    def start_pbft_nodes(self):
        logger.info("Starting PBFT node Flask apps...")
        for node in self.pbft_nodes.values():
            thread = threading.Thread(target=node.app.run, kwargs={'host': node.host, 'port': node.port, 'threaded': True})
            thread.daemon = True
            thread.start()
            self.node_threads.append(thread)
        time.sleep(2)

    def setup_fl_server(self):
        logger.info("Setting up FL Server...")
        self.fl_server = FLServer(num_clients=self.num_fl_clients, pbft_primary_address_getter=self._get_current_pbft_primary_address, host='127.0.0.1', port=8080, model_builder=self.active_model_builder, model_input_shape=self.model_input_shape, num_classes=self.num_classes)

    def start_fl_server(self):
        if not self.fl_server: return
        thread = threading.Thread(target=self.fl_server.run_server_app)
        thread.daemon = True
        thread.start()
        self.node_threads.append(thread)
        time.sleep(2)

    def setup_fl_clients(self):
        if self.X_train is None or self.y_train is None: return
        logger.info(f"Setting up {self.num_fl_clients} FL clients...")
        num_total_samples = len(self.X_train)
        data_indices = list(range(num_total_samples))
        random.shuffle(data_indices)
        samples_per_client = num_total_samples // self.num_fl_clients
        current_idx = 0
        for i in range(self.num_fl_clients):
            subset_indices = data_indices[current_idx:current_idx + samples_per_client]
            current_idx += samples_per_client
            x_subset = self.X_train[subset_indices]
            y_subset = self.y_train[subset_indices]
            is_malicious = (i < self.num_malicious_clients)
            self.fl_clients[i] = FLClient(client_id=i, server_address=self.fl_server.address, data_subset=(x_subset, y_subset), port_start=5000, is_malicious=is_malicious, model_builder=self.active_model_builder, model_input_shape=self.model_input_shape, num_classes=self.num_classes)
        logger.info("FL clients configured.")

    def start_fl_clients(self):
        if not self.fl_clients: return
        for client in self.fl_clients.values():
            thread = threading.Thread(target=client.run_client_app)
            thread.daemon = True
            thread.start()
            self.node_threads.append(thread)
        time.sleep(2)
        for client in self.fl_clients.values():
            threading.Thread(target=client.register_with_server).start()

    def run_federated_learning(self, num_rounds=5, clients_per_round=None):
        if not self.fl_server or not self.fl_clients: return
        if clients_per_round is None: clients_per_round = self.num_fl_clients
        for fl_round in range(num_rounds):
            self.fl_server.round_num = fl_round + 1
            logger.info(f"\n--- Starting FL Round {self.fl_server.round_num}/{num_rounds} ---")
            eval_client = next((c for c in self.fl_clients.values() if not c.is_malicious and c.num_samples > 0), self.fl_clients.get(0))
            if eval_client:
                eval_client.model.set_weights(self.fl_server.global_model.get_weights())
                accuracy = eval_client.evaluate_model(self.test_data_tuple)
                logger.info(f"Global model accuracy before round {self.fl_server.round_num}: {accuracy:.2f}%")
            self.fl_server.distribute_model()
            collected_updates = self.fl_server.get_collected_updates(min(len(self.fl_server.client_addresses), clients_per_round), 60)
            if collected_updates:
                self.fl_server.aggregate_models(collected_updates)
            time.sleep(7 + self.num_pbft_nodes * 0.5)
            if eval_client:
                eval_client.model.set_weights(self.fl_server.global_model.get_weights())
                accuracy_post = eval_client.evaluate_model(self.test_data_tuple)
                logger.info(f"Global model accuracy after round {self.fl_server.round_num}: {accuracy_post:.2f}%")
    
    # ---  method for personalization ---
    def run_personalization(self, personalization_epochs):
        if not self.fl_clients or not self.fl_server:
            logger.error("FL clients or server not available for personalization.")
            return

        # First, evaluate the final global model as a baseline
        logger.info("\n--- Evaluating Final Global Model (Before Personalization) ---")
        final_global_weights = self.fl_server.global_model.get_weights()
        
        # Use a representative non-malicious client to evaluate the global model
        eval_client = next((c for c in self.fl_clients.values() if not c.is_malicious and c.num_samples > 0), self.fl_clients.get(0))
        
        if eval_client:
            eval_client.model.set_weights(final_global_weights)
            global_accuracy = eval_client.evaluate_model(self.test_data_tuple)
            logger.info(f"Final Global Model Accuracy (evaluated by client {eval_client.client_id}): {global_accuracy:.2f}%")
        else:
            logger.warning("Could not find a suitable client to evaluate the final global model.")

        # Start personalization phase
        logger.info(f"\n--- Starting Personalization Phase (Fine-tuning for {personalization_epochs} epochs) ---")

        # Serialize test data once to send to all clients
        test_data_serializable = (self.test_data_tuple[0].tolist(), self.test_data_tuple[1].tolist())

        personalized_results = {}
        for client_id, client in self.fl_clients.items():
            try:
                # The client's model already has the final global weights from the last round
                payload = {
                    'epochs': personalization_epochs,
                    'test_data': test_data_serializable
                }
                # Use a longer timeout for personalization training
                response = requests.post(f"{client.client_api_address}/personalize", json=payload, timeout=180) 
                
                if response.status_code == 200:
                    result = response.json()
                    acc = result.get('personalized_accuracy')
                    personalized_results[client_id] = acc
                    logger.info(f"Client {client_id} personalized accuracy: {acc:.2f}%")
                else:
                    logger.warning(f"Client {client_id} failed to personalize: {response.status_code} - {response.text}")
                    personalized_results[client_id] = "Error"
            except requests.exceptions.RequestException as e:
                logger.error(f"Could not reach client {client_id} for personalization: {e}")
                personalized_results[client_id] = "Unreachable"

        logger.info("\n--- Personalization Results Summary ---")
        for client_id, acc in sorted(personalized_results.items()):
            is_malicious_str = " (Malicious)" if self.fl_clients[client_id].is_malicious else ""
            logger.info(f"  - Client {client_id}{is_malicious_str}: {acc}")

    # --- print_pbft_blockchain_states, print_pbft_message_counts, and shutdown methods remain unchanged ---
    def print_pbft_blockchain_states(self):
        logger.info("\n--- PBFT Node Blockchain States ---")
        for node_id, node_obj in self.pbft_nodes.items():
            try:
                response = requests.get(f"{node_obj.peers[node_id]}/pbft_chain", timeout=2)
                if response.status_code == 200:
                    chain_data = response.json().get('chain', [])
                    logger.info(f"\n--- PBFT Node {node_id} Blockchain (Length: {len(chain_data)}) ---")
                    for block in chain_data:
                        logger.info(f"  Block {block['index']}: Digest={block['fl_update_digest'][:10]}...")
                else: logger.warning(f"Could not fetch chain from PBFT Node {node_id}: HTTP {response.status_code}")
            except requests.exceptions.RequestException as e: logger.error(f"Error fetching chain from PBFT Node {node_id}: {e}")

    def print_pbft_message_counts(self):
        logger.info("\n--- PBFT Node Message Counts ---")
        for node_id, node_obj in self.pbft_nodes.items():
            try:
                response = requests.get(f"{node_obj.peers[node_id]}/pbft_message_counts", timeout=2)
                if response.status_code == 200:
                    counts = response.json(); logger.info(f"PBFT Node {node_id} message counts: {counts}")
                else: logger.warning(f"Could not fetch message counts from PBFT Node {node_id}: HTTP {response.status_code}")
            except requests.exceptions.RequestException as e: logger.error(f"Error fetching message counts from PBFT Node {node_id}: {e}")

    def shutdown(self):
        logger.info("Attempting to shutdown simulation gracefully...")