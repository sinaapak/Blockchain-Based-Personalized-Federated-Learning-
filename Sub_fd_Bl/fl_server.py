import logging
import requests
import json
import numpy as np
import queue
import time # For FLServer app
from flask import Flask, request, jsonify # For FLServer app

from fl_model_utils import get_model_weights_keras, set_model_weights_keras, digest_model_weights_keras, calculate_update_l2_norm
# fl_server.py
# ... other imports ...
logger = logging.getLogger(__name__)

class FLServer:
    def __init__(self, num_clients, pbft_primary_address_getter, 
                 model_builder, model_input_shape, num_classes, # <<< THESE ARE THE CRUCIAL NEW PARAMETERS
                 host='127.0.0.1', port=8080): 
        self.host = host
        self.port = port
        self.address = f"http://{host}:{port}"
        
        self.model_builder = model_builder
        self.model_input_shape = model_input_shape
        self.num_classes = num_classes
        self.global_model = self.model_builder(self.model_input_shape, self.num_classes) # Use the passed parameters
        logger.info(f"FLServer initialized with model built by {model_builder.__name__}, input_shape: {model_input_shape}, num_classes: {num_classes}")

        self.num_clients = num_clients
        self.client_addresses = {}  
        self.pbft_primary_address_getter = pbft_primary_address_getter 
        self.round_num = 0
        self.aggregated_updates_digests = [] 
        self.global_weights_history = []
        self.client_updates_queue = queue.Queue()

        self.app = Flask(f'fl_server_app_{port}') 
        self._register_routes()
    
    # ... rest of the FLServer class (methods like _register_routes, run_server_app, etc.)
    # Make sure all methods from the last working version are present here.
    # For example:
    def _register_routes(self):
        @self.app.route('/register_client', methods=['POST'])
        def register_client_route():
            data = request.get_json()
            client_id = data.get('client_id')
            address = data.get('address')
            if client_id is not None and address:
                self.client_addresses[client_id] = address
                logger.info(f"FL Server registered client {client_id} at {address}. Total clients: {len(self.client_addresses)}")
                return jsonify({'message': 'Client registered'}), 200
            return jsonify({'message': 'Invalid client data'}), 400

        @self.app.route('/receive_update', methods=['POST'])
        def receive_update_route():
            data = request.get_json()
            client_id = data.get('client_id')
            weights_data = data.get('weights')
            num_samples = data.get('num_samples')
            if client_id is None or weights_data is None or num_samples is None:
                return jsonify({'message': 'Incomplete update data'}), 400
            try:
                weights = [np.array(w) for w in weights_data]
            except Exception as e:
                return jsonify({'message': 'Error parsing weights'}), 400
            logger.info(f"FL Server received update from client {client_id} with {num_samples} samples.")
            self.client_updates_queue.put({'client_id': client_id, 'weights': weights, 'num_samples': num_samples})
            return jsonify({'message': 'Update received by FL Server'}), 200
        
        @self.app.route('/status', methods=['GET']) 
        def status_route():
            return jsonify({
                'status': 'FL Server is active',
                'num_registered_clients': len(self.client_addresses),
                'current_round': self.round_num
            }), 200
            
    def run_server_app(self):
        logger.info(f"Starting FL Server Flask app on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True)

    def distribute_model(self):
        if not self.client_addresses:
            logger.warning("FL Server: No clients registered to distribute model to.")
            return
        global_weights = get_model_weights_keras(self.global_model)
        if not self.global_weights_history or digest_model_weights_keras(self.global_weights_history[-1]) != digest_model_weights_keras(global_weights):
             self.global_weights_history.append(list(global_weights)) 
        global_weights_serializable = [w.tolist() for w in global_weights]
        logger.info(f"FL Server distributing model for round {self.round_num} to {len(self.client_addresses)} clients.")
        for client_id, address in self.client_addresses.items():
            try:
                response = requests.post(f"{address}/receive_model", json={'weights': global_weights_serializable, 'round_num': self.round_num}, timeout=5)
                logger.info(f"FL Server sent model to client {client_id} at {address}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"FL Server failed to send model to client {client_id} at {address}: {e}")

    def aggregate_models(self, client_updates):
        if not client_updates:
            logger.warning("FL Server: No client updates received for aggregation.")
            return None
        current_global_weights = self.global_weights_history[-1] if self.global_weights_history else get_model_weights_keras(self.global_model)
        update_l2_norms = []
        for update in client_updates:
            client_id = update['client_id']
            client_weights_np = [np.array(w) for w in update['weights']]
            l2_norm = calculate_update_l2_norm(current_global_weights, client_weights_np)
            update_l2_norms.append((client_id, l2_norm, update))
            logger.info(f"FL Server: Client {client_id} update L2 norm: {l2_norm:.4f}")
        norms = [norm for _, norm, _ in update_l2_norms]
        if len(norms) < 2:
            l2_threshold = float('inf')
            logger.info("FL Server: Not enough updates for L2 norm threshold. No L2 filtering.")
        else:
            mean_norm = np.mean(norms); std_norm = np.std(norms)
            l2_threshold = mean_norm + 2.0 * std_norm
            logger.info(f"FL Server: L2 norm threshold round {self.round_num}: {l2_threshold:.4f} (Mean: {mean_norm:.4f}, Std: {std_norm:.4f})")
        filtered_updates = []
        rejected_clients = []
        for client_id, l2_norm, update_data in update_l2_norms:
            if l2_norm <= l2_threshold: filtered_updates.append(update_data); logger.info(f"FL Server: Client {client_id} update accepted (L2 norm: {l2_norm:.4f})")
            else: rejected_clients.append(client_id); logger.warning(f"FL Server: Client {client_id} update REJECTED (L2 norm: {l2_norm:.4f} > Threshold: {l2_threshold:.4f})")
        if not filtered_updates:
            logger.warning(f"FL Server: All client updates rejected by L2 norm filtering. Skipping aggregation round {self.round_num}.")
            return None
        first_filtered_weights = [np.array(w) for w in filtered_updates[0]['weights']]
        avg_weights = [np.zeros_like(w) for w in first_filtered_weights]
        total_samples = sum(update['num_samples'] for update in filtered_updates)
        if total_samples == 0: logger.warning("FL Server: Total samples from filtered clients is zero."); return None
        for update in filtered_updates:
            num_samples = update['num_samples']
            client_weights_np = [np.array(w) for w in update['weights']]
            for i, weight_layer in enumerate(client_weights_np):
                avg_weights[i] += weight_layer * (num_samples / total_samples)
        set_model_weights_keras(self.global_model, avg_weights)
        logger.info(f"FL Server: Global model aggregated from {len(filtered_updates)} clients for round {self.round_num}.")
        if rejected_clients: logger.info(f"FL Server: Rejected clients in round {self.round_num}: {rejected_clients}")
        aggregated_digest = digest_model_weights_keras(avg_weights)
        self.aggregated_updates_digests.append(aggregated_digest)
        self.send_fl_update_to_pbft(aggregated_digest)
        return aggregated_digest

    def send_fl_update_to_pbft(self, update_digest):
        pbft_primary_address = self.pbft_primary_address_getter()
        if not pbft_primary_address:
            logger.error(f"FL Server: PBFT primary address N/A. Cannot send digest {update_digest[:10]}."); return
        try:
            logger.info(f"FL Server sending digest {update_digest[:10]} to PBFT primary at {pbft_primary_address}")
            response = requests.post(f"{pbft_primary_address}/fl_update_request", json={'fl_update_digest': update_digest}, timeout=5)
            logger.info(f"FL Server sent digest {update_digest[:10]} to PBFT primary ({pbft_primary_address}): {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"FL Server failed to send digest {update_digest[:10]} to PBFT primary ({pbft_primary_address}): {e}")

    def get_collected_updates(self, expected_updates, timeout_per_client):
        received_updates = []
        start_time = time.time()
        deadline = start_time + (expected_updates * timeout_per_client)
        logger.info(f"FL Server round {self.round_num}: Waiting for {expected_updates} client updates. Timeout: {expected_updates * timeout_per_client:.1f}s")
        while len(received_updates) < expected_updates and time.time() < deadline:
            try:
                update = self.client_updates_queue.get(timeout=1)
                received_updates.append(update)
                logger.info(f"FL Server round {self.round_num}: Collected update from client {update['client_id']}. Total: {len(received_updates)}/{expected_updates}")
            except queue.Empty:
                if time.time() >= deadline: logger.warning(f"FL Server round {self.round_num}: Update collection timed out."); break
            except Exception as e: logger.error(f"FL Server round {self.round_num}: Error getting update from queue: {e}"); break
        if len(received_updates) < expected_updates:
             logger.warning(f"FL Server round {self.round_num}: Collected only {len(received_updates)} of {expected_updates} expected updates.")
        return received_updates