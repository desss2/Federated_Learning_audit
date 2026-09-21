import os
import json


'''
LABEL_MODE può assumere tre valori diversi a seconda della tipologia di classificazione che si vuole effettuare:
    • classificazione binaria (attacco/benigno): "binary"
    • classificazione multiclasse (categoria di attacco/ benigno): "category"
    • classificazione multiclasse (tipologia (sotto-categoria) di attacco/benigno): "attack_type"
    lo scenario è definito dalla variabile LABEL_MODE in globals.py
'''
LABEL_MODE = "category"

NUM_CLIENTS = 2

TRAIN_PATH = "CICIoMT2024/WiFi_and_MQTT/attacks/CSV/train"
TEST_PATH = "CICIoMT2024/WiFi_and_MQTT/attacks/CSV/test"
RESULTS_PATH = "./results.npy"

if os.path.exists("/.dockerenv"):
    IDENTITY_MAPPING_FILE = "/app/identity_mapping.json"
else:
    IDENTITY_MAPPING_FILE = "./identity_mapping.json"


RANDOM_SEED=42

TUNE_SIZE = 50_000  # dimensione per il campione su cui fare il tuning

ROUNDS = 5 # Federated training with advanced reputation
REPUTATION_THRESHOLD = 0.3  # Minimum reputation to participate

LOCAL_TRAIN_SIZE = 100_000   # numero dei dataset che ogni client deve utilizzare per il training

EPOCHS = 10
BATCH_SIZE = 32


NOISE_CONFIG = {
    0: 0.0,   # Clean
    1: 0.1,   # 10% noise
    2: 0.3,   # 30% noise
    3: 0.5,   # 50% noise
    4: 0.7    # 70% noise
}

# True per label noise, False per clean
USE_LABEL_NOISE = True
# Simulazione di alterazioni malevole da parte del server
SIMULATE_WRONG_MODEL_HASH = True
SIMULATE_WRONG_AGGREGATION_DECISION = False

SIMULATE_WRONG_IPFS_RECORD = True


RPC_BASE_PORT = 8545
IPFS_API_BASE_PORT = 5001


def get_node_id(node_name):
    if node_name == "server":
        return 0

    return int(node_name.split("_")[1]) + 1


# crea endpoint per blockchain
def get_blockchain_rpc_url(node_name):
    if node_name == "server":
        host = "besu-server"
    else:
        host = f"besu-{node_name.replace('_', '-')}"

    return f"http://{host}:8545"


# crea endpoit per ipfs
def get_ipfs_api_url(node_name):
    if node_name == "server":
        host = "ipfs-server"
    else:
        host = f"ipfs-{node_name.replace('_', '-')}"

    return f"http://{host}:5001"


def get_blockchain_private_key_path(node_name):
    if node_name == "server":
        return os.getenv(
            "BLOCKCHAIN_SERVER_PRIVATE_KEY_PATH",
            "blockchain/network/nodes/server/key.priv"
        )

    return f"blockchain/network/nodes/{node_name}/key.priv"

BLOCKCHAIN_SMART_CONTRACT=os.getenv(
    "BLOCKCHAIN_SMART_CONTRACT",
    "blockchain/contract_address.json"
)


