# fl_client.py
import logging
import requests
import json
import numpy as np
import random
import threading
from flask import Flask, request, jsonify

from fl_model_utils import get_model_weights_keras, set_model_weights_keras

logger = logging.getLogger(__name__)

class FLClient:
    # --- __init__ remains mostly unchanged ---
    def __init__(self, client_id, server_address, data_subset, 
                 model_builder, model_input_shape, num_classes,
                 host='127.0.0.1', port_start=5000, 
                 is_malicious=False, poisoning_rate=0.1):
        self.client_id = client_id
        self.server_address = server_address
        
        self.model_builder = model_builder
        self.model_input_shape = model_input_shape
        self.num_classes = num_classes
        self.model = self.model_builder(self.model_input_shape, self.num_classes)
        
        self.x_train, self.y_train = data_subset if data_subset and len(data_subset) == 2 else (None, None)
        self.num_samples = len(self.x_train) if self.x_train is not None and len(self.x_train) > 0 else 0
        
        self.is_malicious = is_malicious
        self.poisoning_rate = poisoning_rate if is_malicious else 0.0

        self.host = host
        self.port = port_start + client_id
        self.client_api_address = f"http://{self.host}:{self.port}"

        self.app = Flask(f'fl_client_{client_id}_app_{self.port}')
        self._register_routes()
        self.current_round = -1

        if self.is_malicious: 
            logger.warning(f"Client {self.client_id} ({self.client_api_address}) is MALICIOUS. Poisoning: {self.poisoning_rate*100}%. Samples: {self.num_samples}")
        else:
            logger.info(f"Client {self.client_id} ({self.client_api_address}) initialized. Samples: {self.num_samples}")

    def _register_routes(self):
        @self.app.route('/receive_model', methods=['POST'])
        def receive_model_route():
            try:
                data = request.get_json()
                weights_data = data.get('weights')
                round_num = data.get('round_num', -1)
                self.current_round = round_num
                logger.info(f"Client {self.client_id}: Received global model for round {self.current_round}.")
                weights = [np.array(w) for w in weights_data]
                set_model_weights_keras(self.model, weights)
                if self.num_samples > 0: 
                    training_thread = threading.Thread(target=self.local_train_and_send_update, args=(self.current_round,))
                    training_thread.start()
                    return jsonify({'message': f'Model for round {self.current_round} received, training started.'}), 200
                else:
                    return jsonify({'message': f'Model for round {self.current_round} received, no data to train.'}), 200
            except Exception as e:
                logger.error(f"Client {self.client_id}: Error in receive_model_route: {e}", exc_info=True)
                return jsonify({'message': 'Error processing model: ' + str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def status_route():
            return jsonify({
                'client_id': self.client_id, 'status': 'active', 'address': self.client_api_address,
                'current_round': self.current_round, 'is_malicious': self.is_malicious, 'num_samples': self.num_samples
            }), 200

        # --- New route for personalization ---
        @self.app.route('/personalize', methods=['POST'])
        def personalize_route():
            try:
                data = request.get_json()
                epochs = data.get('epochs', 1)
                
                # The orchestrator sends the test data for evaluation
                test_data_json = data.get('test_data')
                if not test_data_json:
                    return jsonify({'message': 'Test data is required for personalization evaluation'}), 400

                # Reconstruct test data from JSON
                x_test = np.array(test_data_json[0])
                y_test = np.array(test_data_json[1])
                test_data_tuple = (x_test, y_test)

                accuracy = self.personalize_model(epochs, test_data_tuple)
                
                return jsonify({'client_id': self.client_id, 'personalized_accuracy': accuracy}), 200
            except Exception as e:
                logger.error(f"Client {self.client_id}: Error in personalize_route: {e}", exc_info=True)
                return jsonify({'message': 'Error during personalization: ' + str(e)}), 500

    # --- New method to perform personalization ---
    def personalize_model(self, epochs, test_data):
        if self.num_samples == 0:
            logger.warning(f"Client {self.client_id}: Cannot personalize, no local data available.")
            return 0.0

        logger.info(f"Client {self.client_id}: Starting personalization. Fine-tuning for {epochs} epochs on {self.num_samples} samples.")
        
        # The client's self.model already holds the final global weights.
        # We now fine-tune it on local data.
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=32, verbose=0)
        
        logger.info(f"Client {self.client_id}: Personalization training finished.")
        
        # Evaluate the newly personalized model on the provided test data
        return self.evaluate_model(test_data)

    # --- Other methods remain unchanged ---
    def run_client_app(self):
        logger.info(f"Starting FL Client {self.client_id} Flask app on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True)

    def register_with_server(self):
        try:
            response = requests.post(f"{self.server_address}/register_client", json={
                'client_id': self.client_id, 'address': self.client_api_address
            }, timeout=5)
            if response.status_code == 200: logger.info(f"Client {self.client_id} successfully registered.")
            else: logger.warning(f"Client {self.client_id} failed to register: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Client {self.client_id} registration failed: {e}")

    def local_train_and_send_update(self, fl_round, epochs=1, batch_size=32):
        if self.x_train is None or self.y_train is None or self.num_samples == 0:
            logger.warning(f"Client {self.client_id}: No data to train on for round {fl_round}."); return
        
        x_local, y_local = np.copy(self.x_train), np.copy(self.y_train)
        if self.is_malicious and self.poisoning_rate > 0:
            num_to_poison = min(int(self.poisoning_rate * len(y_local)), len(y_local))
            if num_to_poison > 0:
                poison_indices = random.sample(range(len(y_local)), num_to_poison)
                for idx in poison_indices:
                    original_label = y_local[idx]
                    possible_labels = [l for l in range(self.num_classes) if l != original_label]
                    if possible_labels: y_local[idx] = random.choice(possible_labels)
                logger.info(f"Client {self.client_id} (malicious) poisoned {num_to_poison} labels.")
        
        self.model.fit(x_local, y_local, epochs=epochs, batch_size=batch_size, verbose=0)
        logger.info(f"Client {self.client_id}: Finished local training for round {fl_round}.")
        
        local_weights_serializable = [w.tolist() for w in get_model_weights_keras(self.model)]
        update_payload = {'client_id': self.client_id, 'weights': local_weights_serializable, 'num_samples': self.num_samples, 'fl_round': fl_round}
        
        try:
            response = requests.post(f"{self.server_address}/receive_update", json=update_payload, timeout=10)
            logger.info(f"Client {self.client_id} sent update for round {fl_round}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Client {self.client_id} failed to send update for round {fl_round}: {e}")

    def evaluate_model(self, test_data):
        if test_data[0] is None or test_data[1] is None or len(test_data[0]) == 0: return 0.0
        try:
            loss, accuracy = self.model.evaluate(test_data[0], test_data[1], verbose=0)
            logger.info(f"Client {self.client_id} model evaluation - Acc: {accuracy*100:.2f}%, Loss: {loss:.4f}")
            return accuracy * 100
        except Exception as e:
            logger.error(f"Client {self.client_id} failed during model evaluation: {e}", exc_info=True)
            return 0.0