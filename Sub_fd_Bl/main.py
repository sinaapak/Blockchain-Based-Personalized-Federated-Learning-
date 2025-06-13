# main.py
import logging
import time
from orchestration import SimulationOrchestrator
import nltk 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    resources = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt")]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
            logger.debug(f"NLTK resource '{resource_name}' found.")
        except LookupError:
            logger.info(f"NLTK resource '{resource_name}' not found. Downloading...")
            nltk.download(resource_name, quiet=False) 

if __name__ == "__main__":
    ensure_nltk_resources()

    NUM_PBFT_NODES = 4
    NUM_FAULTY_PBFT = 1
    NUM_FL_CLIENTS = 5
    NUM_MALICIOUS_CLIENTS = 1
    FL_ROUNDS = 3
    # --- parameter for personalization ---
    PERSONALIZATION_EPOCHS = 5

    CLIENT_DATA_SPLIT_SIZE = 1 
    TEXT_DATA_PATH = 'Sample_of_Dermo_Questions.csv' 

    if NUM_PBFT_NODES < (3 * NUM_FAULTY_PBFT + 1):
        logger.error(f"CRITICAL: PBFT nodes ({NUM_PBFT_NODES}) < required ({3*NUM_FAULTY_PBFT+1}).")
    else:
        logger.info(f"PBFT Config: N={NUM_PBFT_NODES}, f={NUM_FAULTY_PBFT}.")

    logger.info("--- Starting BFL Simulation (Text Model Only) ---")
    logger.info(f"Params: PBFT Nodes={NUM_PBFT_NODES}, FL Clients={NUM_FL_CLIENTS} (Malicious={NUM_MALICIOUS_CLIENTS}), FL Rounds={FL_ROUNDS}")
    logger.info(f"Personalization: Fine-tuning for {PERSONALIZATION_EPOCHS} epochs post-FL.")
    logger.info(f"Dataset: {TEXT_DATA_PATH}")

    orchestrator = SimulationOrchestrator(
        num_pbft_nodes=NUM_PBFT_NODES,
        num_fl_clients=NUM_FL_CLIENTS,
        num_faulty_pbft=NUM_FAULTY_PBFT,
        num_malicious_clients=NUM_MALICIOUS_CLIENTS,
        client_data_split_size=CLIENT_DATA_SPLIT_SIZE, 
        data_path=TEXT_DATA_PATH
    )

    try:
        orchestrator.setup_pbft_network()
        orchestrator.start_pbft_nodes()
        logger.info("PBFT network active. Allowing time for init...")
        time.sleep(3)

        orchestrator.setup_fl_server()
        orchestrator.start_fl_server()
        logger.info("FL Server active. Allowing time for init...")
        time.sleep(2)

        orchestrator.setup_fl_clients()
        orchestrator.start_fl_clients() 
        logger.info("FL Clients active. Allowing time for registration...")
        time.sleep(max(5, NUM_FL_CLIENTS * 0.5)) 

        if orchestrator.fl_server and len(orchestrator.fl_server.client_addresses) > 0:
            logger.info(f"{len(orchestrator.fl_server.client_addresses)} FL clients registered.")
        else:
            logger.warning("No FL clients seem to be registered. Check client and server logs.")
            if NUM_FL_CLIENTS > 0: logger.error("Proceeding, but FL rounds might fail or show no activity.")
        
        clients_for_fl = min(NUM_FL_CLIENTS, len(orchestrator.fl_server.client_addresses) if orchestrator.fl_server else 0)

        if clients_for_fl > 0:
            orchestrator.run_federated_learning(num_rounds=FL_ROUNDS, clients_per_round=clients_for_fl)
        else:
            logger.warning("No clients available or registered to participate in federated learning.")

        logger.info("\n--- Federated Learning Global Training Finished ---")
        time.sleep(2)

        # --- New call to run the personalization phase ---
        orchestrator.run_personalization(personalization_epochs=PERSONALIZATION_EPOCHS)
        logger.info("\n--- Personalization Phase Finished ---")
        time.sleep(3)

        orchestrator.print_pbft_blockchain_states()
        orchestrator.print_pbft_message_counts()

    except KeyboardInterrupt: logger.warning("\n--- Simulation interrupted by user (Ctrl+C) ---")
    except Exception as e: logger.error(f"\n--- An unexpected error occurred: {e} ---", exc_info=True)
    finally:
        orchestrator.shutdown()
        logger.info("\n--- Simulation Complete ---")