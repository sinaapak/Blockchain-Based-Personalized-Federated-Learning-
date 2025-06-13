# pbft_node.py
import hashlib
import json
import time
import requests
from flask import Flask, jsonify, request
import logging
import random
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from blockchain_components import Blockchain # Import Blockchain

logger = logging.getLogger(__name__)

class PBFTNode:
    def __init__(self, node_id, N, f, host, port, peers):
        self.node_id = node_id
        self.N = N
        self.f = f
        self.quorum_size = 2 * f + 1
        self.host = host
        self.port = port
        self.peers = peers
        self.blockchain = Blockchain()
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.public_keys = {}

        self.current_view = 0
        self.sequence_number = 0 
        
        self.prepared_messages = {} 
        self.committed_messages = {} 
        
        self.is_primary = False
        self.set_primary()

        self.app = Flask(f'pbft_node_{node_id}')
        self.register_routes()
        self.message_counts = {}
        
        self.current_fl_update_digest = None 
        self.stashed_prepare_messages = {} 
        self.sent_commit_flags = set() 
        self.execution_flags = set() 

        logger.info(f"PBFT Node {self.node_id} initialized. Quorum: {self.quorum_size}. Is Primary: {self.is_primary}")

    def set_primary(self):
        self.is_primary = (self.node_id == (self.current_view % self.N))
        # Logger message moved to __init__ or when view changes, to avoid repeat on every call if not needed
        # logger.info(f"PBFT Node {self.node_id}: {'Is PRIMARY' if self.is_primary else 'Is REPLICA'} for view {self.current_view}")


    def sign_message(self, message):
        message_copy = message.copy()
        message_copy.pop('signature', None)
        serialized_message = json.dumps(message_copy, sort_keys=True).encode()
        signature = self.private_key.sign(
            serialized_message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return signature.hex()

    def verify_signature(self, public_key_pem, message_with_signature, signature_hex):
        if not public_key_pem:
            logger.error(f"Node {self.node_id}: No public key available for sender to verify {message_with_signature.get('message_type', 'Unknown type')}.")
            return False
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
            signature = bytes.fromhex(signature_hex)
            message_copy = message_with_signature.copy()
            message_copy.pop('signature', None)
            serialized_message = json.dumps(message_copy, sort_keys=True).encode()
            public_key.verify(
                signature, serialized_message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Node {self.node_id}: Sig verification failed for {message_with_signature.get('message_type', 'Unknown type')} from {message_with_signature.get('sender_id','Unknown sender')}: {e}. Msg: {json.dumps(message_copy)}", exc_info=False) # Set exc_info=True for full traceback if needed
            return False

    def broadcast_message(self, endpoint, message_type, payload):
        message = {
            'sender_id': self.node_id,
            'message_type': message_type,
            'payload': payload,
            'timestamp': time.time()
        }
        message['signature'] = self.sign_message(message)

        for peer_id, peer_address in self.peers.items():
            if peer_id == self.node_id:
                continue
            try:
                # Increased delay slightly to give more processing time and ordering variance
                time.sleep(random.uniform(0.05, 0.20)) 
                response = requests.post(f"{peer_address}/{endpoint}", json=message, timeout=3)
                # Ensure peer_id is consistently typed for dict keys, e.g., always int or always str
                # Assuming peer_id is int from setup.
                self.message_counts.setdefault(peer_id, {}).setdefault(message_type, 0)
                self.message_counts[peer_id][message_type] += 1
                logger.debug(f"PBFT Node {self.node_id} sent {message_type} to {peer_id}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"PBFT Node {self.node_id} failed to send {message_type} to {peer_id} at {peer_address}: {e}")

    def handle_fl_update_request(self, fl_update_digest):
        if not self.is_primary:
            primary_node_id = self.current_view % self.N
            primary_address = self.peers.get(primary_node_id)
            logger.warning(f"Node {self.node_id} (not primary) received FL update. Primary is {primary_node_id}.")
            return {'message': 'Redirect to primary', 'primary_node_id': primary_node_id, 'primary_address': primary_address}, 400

        self.sequence_number += 1
        self.current_fl_update_digest = fl_update_digest

        pre_prepare_payload = {
            'view': self.current_view,
            'sequence_number': self.sequence_number,
            'digest': fl_update_digest,
        }
        logger.info(f"Primary Node {self.node_id} (View {self.current_view}) initiating PRE-PREPARE for digest {fl_update_digest[:10]}..., seq {self.sequence_number}")
        
        if self.sequence_number not in self.prepared_messages: self.prepared_messages[self.sequence_number] = {}
        if fl_update_digest not in self.prepared_messages[self.sequence_number]: self.prepared_messages[self.sequence_number][fl_update_digest] = {}
        self.prepared_messages[self.sequence_number][fl_update_digest][self.node_id] = {'payload': pre_prepare_payload, 'sender_id': self.node_id, 'type': 'pre-prepare', 'signature': 'local_primary'}
        
        self.broadcast_message('pbft_pre_prepare', 'pre_prepare', pre_prepare_payload)
        # Primary should also consider itself "prepared" enough to send a commit if it gets enough prepares
        # This means after broadcasting pre-prepare, it should then act as if it received its own pre-prepare and proceed
        # The line above already stores its "pre-prepare". The _execute_prepare_logic will be triggered by incoming prepares.
        return {'message': 'FL Update received, starting PBFT consensus', 'sequence_number': self.sequence_number}, 200

    def handle_pre_prepare(self, message):
        sender_id = message['sender_id'] 
        payload = message['payload']
        signature_hex = message['signature']

        if sender_id != (self.current_view % self.N):
            logger.warning(f"Node {self.node_id} received PRE-PREPARE from non-primary {sender_id} for view {self.current_view}")
            return jsonify({'message': 'Not from primary'}), 400
        
        sender_public_key_pem = self.public_keys.get(sender_id)
        if not sender_public_key_pem:
            return jsonify({'message': 'Unknown sender public key for PRE-PREPARE'}), 400
        if not self.verify_signature(sender_public_key_pem, message, signature_hex):
            return jsonify({'message': 'Invalid PRE-PREPARE signature'}), 400

        view = payload['view']
        seq_num = payload['sequence_number']
        digest = payload['digest']

        if view != self.current_view:
            return jsonify({'message': 'Wrong view for PRE-PREPARE'}), 400
        
        self.current_fl_update_digest = digest 

        if seq_num not in self.prepared_messages: self.prepared_messages[seq_num] = {}
        if digest not in self.prepared_messages[seq_num]: self.prepared_messages[seq_num][digest] = {}
        
        if sender_id not in self.prepared_messages[seq_num][digest]: # Store PRE-PREPARE if not already seen
            self.prepared_messages[seq_num][digest][sender_id] = message 
            logger.debug(f"Node {self.node_id} stored PRE-PREPARE from primary {sender_id} for (seq:{seq_num}, dig:{digest[:10]}). Prepared store size for digest: {len(self.prepared_messages[seq_num][digest])}")

            logger.info(f"Node {self.node_id} (View {self.current_view}) accepted PRE-PREPARE for (seq:{seq_num}, dig:{digest[:10]}) from {sender_id}, sending own PREPARE.")
            prepare_payload = {
                'view': self.current_view, 'sequence_number': seq_num, 'digest': digest, 'node_id': self.node_id
            }
            self.broadcast_message('pbft_prepare', 'prepare', prepare_payload)

            key_for_stash = (view, seq_num, digest)
            if key_for_stash in self.stashed_prepare_messages:
                stashed_for_key = self.stashed_prepare_messages.pop(key_for_stash) 
                logger.info(f"Node {self.node_id} processing {len(stashed_for_key)} stashed PREPAREs for key ({view},{seq_num},{digest[:10]}).")
                for stashed_msg in stashed_for_key:
                    logger.info(f"Node {self.node_id} re-evaluating stashed PREPARE from {stashed_msg.get('sender_id')} for (seq:{seq_num}, dig:{digest[:10]}).")
                    self._execute_prepare_logic(stashed_msg)
        else:
            logger.debug(f"Node {self.node_id} already processed PRE-PREPARE from {sender_id} for (seq:{seq_num}, dig:{digest[:10]}).")
            
        return jsonify({'message': 'Pre-prepare handled'}), 200

    def _execute_prepare_logic(self, message):
        sender_id = message['sender_id']
        payload = message['payload']
        view = payload['view'] # view from the PREPARE message
        seq_num = payload['sequence_number']
        digest = payload['digest']

        if seq_num not in self.prepared_messages: self.prepared_messages[seq_num] = {}
        if digest not in self.prepared_messages[seq_num]: self.prepared_messages[seq_num][digest] = {}
        
        # Ensure primary's pre-prepare is also counted if this is the primary processing a prepare from replica
        # Or if this is a replica, the pre-prepare from primary is already stored.
        # The sender_id here is the sender of the PREPARE message.
        if sender_id not in self.prepared_messages[seq_num][digest]:
            self.prepared_messages[seq_num][digest][sender_id] = message
            logger.debug(f"Node {self.node_id} added PREPARE from {sender_id} for (seq:{seq_num}, dig:{digest[:10]}) to store. Store size for digest: {len(self.prepared_messages[seq_num][digest])}")
        else:
            logger.debug(f"Node {self.node_id} already has PREPARE from {sender_id} for (seq:{seq_num}, dig:{digest[:10]}).")

        current_prepared_count = 0
        if seq_num in self.prepared_messages and digest in self.prepared_messages[seq_num]:
             current_prepared_count = len(self.prepared_messages[seq_num][digest])
        
        commit_flag_key = (seq_num, digest)
        already_sent_commit_for_this = commit_flag_key in self.sent_commit_flags

        if current_prepared_count >= self.quorum_size and not already_sent_commit_for_this:
            logger.info(f"Node {self.node_id} (View {self.current_view}) Reached PREPARED for (seq:{seq_num}, dig:{digest[:10]}) ({current_prepared_count}/{self.quorum_size}). Sending COMMIT.")
            self.sent_commit_flags.add(commit_flag_key)

            commit_payload = {
                'view': self.current_view, 
                'sequence_number': seq_num, 
                'digest': digest, 
                'node_id': self.node_id
            }
            
            if seq_num not in self.committed_messages: self.committed_messages[seq_num] = {}
            if digest not in self.committed_messages[seq_num]: self.committed_messages[seq_num][digest] = {}
            self.committed_messages[seq_num][digest][self.node_id] = {'payload': commit_payload, 'local_commit_sent_marker': True, 'sender_id': self.node_id, 'signature': 'local_node'}
            
            self.broadcast_message('pbft_commit', 'commit', commit_payload)
            self._check_and_commit_block(seq_num, digest, self.current_view) 
        else:
            if already_sent_commit_for_this:
                logger.debug(f"Node {self.node_id} (View {self.current_view}) already sent COMMIT for (seq:{seq_num}, dig:{digest[:10]}). Prepared: {current_prepared_count}/{self.quorum_size}.")
            else:
                logger.debug(f"Node {self.node_id} (View {self.current_view}) has {current_prepared_count}/{self.quorum_size} prepared for (seq:{seq_num}, dig:{digest[:10]}). Waiting for more.")

    def handle_prepare(self, message): 
        sender_id = message['sender_id']
        payload = message['payload']
        signature_hex = message['signature']

        sender_public_key_pem = self.public_keys.get(sender_id)
        if not sender_public_key_pem:
            return jsonify({'message': 'Unknown sender public key for PREPARE'}), 400
        if not self.verify_signature(sender_public_key_pem, message, signature_hex):
            return jsonify({'message': 'Invalid PREPARE signature'}), 400

        view = payload['view']
        seq_num = payload['sequence_number']
        digest = payload['digest']

        if view != self.current_view:
            return jsonify({'message': 'Wrong view for PREPARE'}), 400

        primary_of_view = self.current_view % self.N
        has_valid_pre_prepare = (
            seq_num in self.prepared_messages and
            digest in self.prepared_messages[seq_num] and
            primary_of_view in self.prepared_messages[seq_num][digest] 
        )

        if not has_valid_pre_prepare and self.node_id != primary_of_view: 
            key_for_stash = (view, seq_num, digest)
            if key_for_stash not in self.stashed_prepare_messages: 
                self.stashed_prepare_messages[key_for_stash] = []
            
            is_duplicate_in_stash = any(m['sender_id'] == sender_id and m['payload']['digest'] == digest for m in self.stashed_prepare_messages[key_for_stash])
            if not is_duplicate_in_stash:
                self.stashed_prepare_messages[key_for_stash].append(message) 
                logger.info(f"Node {self.node_id} stashed PREPARE from {sender_id} for key ({view},{seq_num},{digest[:10]}). Stash size: {len(self.stashed_prepare_messages[key_for_stash])}")
            else:
                logger.debug(f"Node {self.node_id} ignoring duplicate PREPARE from {sender_id} for stashing for key ({view},{seq_num},{digest[:10]}).")
            return jsonify({'message': 'Prepare stashed, awaiting pre-prepare'}), 202 

        self._execute_prepare_logic(message)
        return jsonify({'message': 'Prepare handled'}), 200

    def _check_and_commit_block(self, seq_num, digest, processing_view):
        commit_count = 0
        # Ensure the specific digest dictionary exists before trying to get its length
        if seq_num in self.committed_messages and digest in self.committed_messages[seq_num]:
            commit_count = len(self.committed_messages[seq_num][digest])
        
        execution_flag_key = (seq_num, digest, 'block_added')
        already_executed_locally = execution_flag_key in self.execution_flags

        if self.node_id == 2: # Node 2 specific debug
             logger.info(f"[NODE 2 DEBUG] _check_and_commit_block: For (seq:{seq_num}, dig:{digest[:10]}) -> commit_count={commit_count}, quorum_size={self.quorum_size}, already_executed={already_executed_locally}")

        if commit_count >= self.quorum_size and not already_executed_locally:
            logger.info(f"Node {self.node_id} (View {processing_view}) Reached COMMITTED-LOCAL for (seq:{seq_num}, dig:{digest[:10]}) ({commit_count}/{self.quorum_size}). Adding block.")
            
            previous_block_hash = self.blockchain.last_block.hash
            new_block = self.blockchain.create_block(fl_update_digest=digest, previous_hash=previous_block_hash)
            logger.info(f"Node {self.node_id} ADDED BLOCK {new_block.index} to chain. Digest: {digest[:10]}, Hash: {new_block.hash[:10]}")
            
            self.execution_flags.add(execution_flag_key) 

            # --- Cleanup ---
            if seq_num in self.prepared_messages and digest in self.prepared_messages[seq_num]:
                del self.prepared_messages[seq_num][digest]
                if not self.prepared_messages[seq_num]: del self.prepared_messages[seq_num]
            
            if seq_num in self.committed_messages and digest in self.committed_messages[seq_num]:
                del self.committed_messages[seq_num][digest]
                if not self.committed_messages[seq_num]: del self.committed_messages[seq_num]
            
            self.sent_commit_flags.discard((seq_num, digest))
            
            key_for_stash_to_clear = (processing_view, seq_num, digest) 
            if key_for_stash_to_clear in self.stashed_prepare_messages:
                del self.stashed_prepare_messages[key_for_stash_to_clear]
                logger.debug(f"Node {self.node_id} cleared stashed prepares for executed key {key_for_stash_to_clear[0]},{key_for_stash_to_clear[1]},{key_for_stash_to_clear[2][:10]}")
            return True 
        
        elif already_executed_locally and self.node_id == 2:
             logger.info(f"[NODE 2 DEBUG] _check_and_commit_block: Already executed for (seq:{seq_num}, dig:{digest[:10]})")
        elif self.node_id == 2 and commit_count < self.quorum_size:
             logger.info(f"[NODE 2 DEBUG] _check_and_commit_block: Not enough commits for (seq:{seq_num}, dig:{digest[:10]}) count: {commit_count}/{self.quorum_size}")
        return False

    def handle_commit(self, message): 
        sender_id = message['sender_id']
        payload = message['payload']
        signature_hex = message['signature']

        sender_public_key_pem = self.public_keys.get(sender_id)
        if not sender_public_key_pem:
            return jsonify({'message': 'Unknown sender public key for commit'}), 400
        if not self.verify_signature(sender_public_key_pem, message, signature_hex):
            return jsonify({'message': 'Invalid commit signature'}), 400

        view = payload['view'] # This is the view from the sender of the COMMIT
        seq_num = payload['sequence_number']
        digest = payload['digest']

        # Node should process commit if it's for the current view it's operating under,
        # or if it's lagging but the commit is for a stable past decision.
        # For simplicity in a non-view-changing scenario, we check against self.current_view.
        if view != self.current_view: 
            logger.warning(f"Node {self.node_id} received COMMIT from {sender_id} for (seq:{seq_num}, view:{view}) but self.current_view is {self.current_view}. Discarding.")
            return jsonify({'message': 'Wrong view for commit'}), 400
        
        if self.node_id == 2:
            logger.info(f"[NODE 2 DEBUG] handle_commit: Received verified COMMIT from {sender_id} for (seq:{seq_num}, digest:{digest[:10]})")

        if seq_num not in self.committed_messages: self.committed_messages[seq_num] = {}
        if digest not in self.committed_messages[seq_num]: self.committed_messages[seq_num][digest] = {}
        
        if sender_id not in self.committed_messages[seq_num][digest]:
             self.committed_messages[seq_num][digest][sender_id] = message 
             if self.node_id == 2:
                 logger.info(f"[NODE 2 DEBUG] Stored COMMIT from {sender_id}. commit_messages for (seq:{seq_num}, dig:{digest[:10]}): senders {list(self.committed_messages[seq_num][digest].keys())}. Total count: {len(self.committed_messages[seq_num][digest])}")
        else:
            if self.node_id == 2:
                logger.info(f"[NODE 2 DEBUG] Already had COMMIT from {sender_id} for (seq:{seq_num}, dig:{digest[:10]}). Current commit_messages senders: {list(self.committed_messages[seq_num][digest].keys()) if seq_num in self.committed_messages and digest in self.committed_messages[seq_num] else 'N/A'}.")
        
        self._check_and_commit_block(seq_num, digest, self.current_view) # Use node's current view for decision context

        return jsonify({'message': 'Commit handled'}), 200

    def register_routes(self):
        @self.app.route('/fl_update_request', methods=['POST'])
        def fl_update_request_route():
            data = request.get_json()
            fl_update_digest = data.get('fl_update_digest')
            if not fl_update_digest:
                return jsonify({'message': 'fl_update_digest is required'}), 400
            response, status_code = self.handle_fl_update_request(fl_update_digest)
            return jsonify(response), status_code

        @self.app.route('/pbft_pre_prepare', methods=['POST'])
        def pbft_pre_prepare_route():
            message = request.get_json()
            return self.handle_pre_prepare(message)

        @self.app.route('/pbft_prepare', methods=['POST'])
        def pbft_prepare_route():
            message = request.get_json()
            return self.handle_prepare(message)

        @self.app.route('/pbft_commit', methods=['POST'])
        def pbft_commit_route():
            message = request.get_json()
            return self.handle_commit(message)

        @self.app.route('/pbft_chain', methods=['GET'])
        def get_chain():
            chain_data = []
            for block_obj in self.blockchain.chain:
                chain_data.append(block_obj.__dict__)
            return jsonify({'chain': chain_data, 'length': len(chain_data)}), 200

        @self.app.route('/pbft_public_key', methods=['GET'])
        def get_pbft_public_key():
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            return jsonify({'node_id': self.node_id, 'public_key': public_pem}), 200

        @self.app.route('/pbft_register_node', methods=['POST'])
        def register_pbft_node(): 
            data = request.get_json()
            node_id_reg = data.get('node_id') 
            address = data.get('address')
            public_key_pem = data.get('public_key')

            if not all([node_id_reg is not None, address, public_key_pem]):
                return jsonify({"message":"Invalid registration data"}), 400
            
            if node_id_reg == self.node_id:
                return jsonify({"message": "Cannot register self"}), 200

            self.peers[node_id_reg] = address
            self.public_keys[node_id_reg] = public_key_pem
            logger.info(f"PBFT Node {self.node_id} registered/updated peer {node_id_reg} at {address}.")
            return jsonify({"message": "Node registered"}), 200
        
        @self.app.route('/pbft_message_counts', methods=['GET'])
        def get_pbft_message_counts():
            return jsonify(self.message_counts), 200

    def run(self):
        logger.info(f"PBFT Node {self.node_id} Flask app setup complete, ready to be run on {self.host}:{self.port}")