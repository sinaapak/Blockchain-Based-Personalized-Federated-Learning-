import hashlib
import json
import time
import logging

logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index, timestamp, fl_update_digest, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.fl_update_digest = fl_update_digest # Digest of the aggregated FL model update
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": str(self.timestamp),
            "fl_update_digest": self.fl_update_digest,
            "previous_hash": self.previous_hash,
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_fl_updates = [] # Stores digests of aggregated FL updates awaiting block
        self.create_genesis_block()

    def create_genesis_block(self):
        self.create_block(fl_update_digest='0', previous_hash='0')

    def create_block(self, fl_update_digest, previous_hash):
        block = Block(
            index=len(self.chain) + 1,
            timestamp=time.time(),
            fl_update_digest=fl_update_digest,
            previous_hash=previous_hash,
        )
        self.pending_fl_updates = [] # Clear pending updates after block creation
        self.chain.append(block)
        return block

    @property
    def last_block(self):
        return self.chain[-1]

    def add_fl_update_to_pending(self, update_digest):
        self.pending_fl_updates.append(update_digest)
        return self.last_block.index + 1

    def is_valid_chain(self, chain):
        # Basic chain validation; in a real system, you'd also validate digests
        previous_block = chain[0]
        block_index = 1

        while block_index < len(chain):
            block = chain[block_index]
            if block.previous_hash != previous_block.hash:
                return False
            previous_block = block
            block_index += 1
        return True