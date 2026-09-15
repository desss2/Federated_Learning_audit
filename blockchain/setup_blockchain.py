import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from globals import NUM_CLIENTS


NUM_NODES = NUM_CLIENTS + 1

BLOCKCHAIN_DIR = Path(__file__).resolve().parent

CONFIG_DIR = BLOCKCHAIN_DIR / "config"
NETWORK_DIR = BLOCKCHAIN_DIR / "network"

CONFIG_FILE = CONFIG_DIR / "configFile.json"

KEYS_DIR = NETWORK_DIR / "keys"
NODES_DIR = NETWORK_DIR / "nodes"

GENESIS_FILE = NETWORK_DIR / "genesis.json"

IDENTITY_MAPPING_FILE = PROJECT_ROOT / "identity_mapping.json"

DOCKER_COMPOSE_FILE = BLOCKCHAIN_DIR / "docker-compose.yml"




DOCKER_SUBNET = "172.28.0.0/16"

DOCKER_IP_BASE = 2

P2P_PORT = 30303
RPC_PORT = 8545

P2P_HOST_BASE_PORT = 30303
RPC_HOST_BASE_PORT = 8545


# creazione cartelle
def prepare_directories():
    if NETWORK_DIR.exists():
        print(f"Removing existing blockchain network: {NETWORK_DIR}")
        shutil.rmtree(NETWORK_DIR)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)


# creazione file di configurazione
def create_config():
    config = {

        "genesis": {
            "config": {
                "chainId": 2026,
                "berlinBlock": 0,
                "londonBlock": 0,
                "qbft": {
                    "blockperiodseconds": 2,
                    "epochlength": 30000,
                    "requesttimeoutseconds": 4
                }
            },
            "nonce": "0x0",
            "timestamp": "0x0",
            "gasLimit": "0x1fffffffffffff",
            "difficulty": "0x1",
            "mixHash":
                "0x63746963616c2062797a616e74696e65206661756c7420746f6c6572616e6365",
            "coinbase":
                "0x0000000000000000000000000000000000000000",
            "alloc": {}
        },
        "blockchain": {
            "nodes": {
                "generate": True,
                "count": NUM_NODES
            }
        }
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"QBFT configuration created: {CONFIG_FILE}")


# generazione blockchain
def generate_blockchain():

    print()
    print("Generating Besu blockchain configuration...")
    print(f"Clients:     {NUM_CLIENTS}")
    print("Server:      1")
    print(f"Total nodes: {NUM_NODES}")
    print()

    command = [
        "besu",
        "operator",
        "generate-blockchain-config",
        "--config-file",
        str(CONFIG_FILE),
        "--to",
        str(NETWORK_DIR)
    ]
    subprocess.run(command, check=True)


def get_key_directories():

    if not KEYS_DIR.exists():
        raise RuntimeError(f"Directory delle chiavi non trovata: {KEYS_DIR}")

    key_directories = sorted(
        [
            directory
            for directory in KEYS_DIR.iterdir()
            if directory.is_dir()
        ],
        key=lambda path: path.name
    )

    if len(key_directories) != NUM_NODES:
        raise RuntimeError(f"Numero di nodi atteso: {NUM_NODES}, "f"ma sono state trovate {len(key_directories)} coppie di chiavi.")
    return key_directories

# diamo dei fondi a tutti gli account
def fund_accounts(key_directories):

    if not key_directories:
        raise RuntimeError("Nessuna chiave generata da Besu.")

    if not GENESIS_FILE.exists():
        raise FileNotFoundError(
            f"Genesis file non trovato: {GENESIS_FILE}"
        )

    with open(GENESIS_FILE, "r") as f:
        genesis = json.load(f)

    genesis["alloc"] = {}

    for key_directory in key_directories:

        account_address = key_directory.name

        genesis["alloc"][account_address] = {
            "balance": "0x3635C9ADC5DEA00000"
        }

        print(f"Account funded: {account_address}")

    with open(GENESIS_FILE, "w") as f:
        json.dump(genesis, f, indent=4)


def prepare_node_directories(key_directories):
    if NODES_DIR.exists():
        print(f"Removing existing nodes directory: {NODES_DIR}")
        shutil.rmtree(NODES_DIR)

    NODES_DIR.mkdir(parents=True, exist_ok=True)
    nodes = []

    for node_id, key_directory in enumerate(key_directories):
        node_name = "server" if node_id == 0 else f"client_{node_id - 1}"
        node_directory = NODES_DIR / node_name
        node_directory.mkdir(parents=True, exist_ok=True)

        private_key = key_directory / "key.priv"
        if not private_key.exists():
            raise FileNotFoundError(f"Private key non trovata: {private_key}")

        shutil.copy2(private_key, node_directory / "key.priv")
        shutil.copy2(GENESIS_FILE, node_directory / "genesis.json")

        nodes.append({
            "id": node_id,
            "name": node_name,
            "directory": node_directory,
            "address": key_directory.name,
            "docker_ip": f"172.28.0.{DOCKER_IP_BASE + node_id}",
            "p2p_port": P2P_PORT,
            "rpc_port": RPC_PORT,
            "p2p_host_port": P2P_HOST_BASE_PORT + node_id,
            "rpc_host_port": RPC_HOST_BASE_PORT + node_id
        })

    return nodes



def create_identity_mapping(nodes):
    server = nodes[0]

    mapping = {
        "server": {
            "blockchain_node": "server",
            "ethereum_address": nodes[0]["address"],
            "ipfs_node": None,
            "ipfs_peer_id": None
        },
        "clients": {}
    }

    for node in nodes[1:]:

        client_name = node["name"]
        mapping["clients"][client_name] = {
            "partition_id": None,
            "client_id": None,
            "blockchain_node": client_name,
            "ethereum_address":  node["address"],
            "ipfs_node": None,
            "ipfs_peer_id": None
        }

    IDENTITY_MAPPING_FILE.write_text(json.dumps(mapping, indent=4) + "\n")

    print(f"Identity mapping created: {IDENTITY_MAPPING_FILE}")

    return mapping


def create_static_nodes(nodes):

    static_nodes = []

    for node in nodes:

        public_key_file = KEYS_DIR / node["address"] / "key.pub"

        if not public_key_file.exists():
            raise FileNotFoundError(f"Public key non trovata: {public_key_file}")

        public_key = public_key_file.read_text().strip().removeprefix("0x")
        enode = (f"enode://{public_key}" f"@{node['docker_ip']}:{P2P_PORT}")

        static_nodes.append(enode)

    for node in nodes:
        static_nodes_file = (node["directory"] / "static-nodes.json")

        static_nodes_file.write_text("[\n" + ",\n".join(f'    "{enode}"'
                for enode in static_nodes) + "\n]\n")

    print("static-nodes.json created for all nodes.")

def create_docker_compose(nodes):
    """Genera docker-compose.yml per la rete Besu."""

    lines = ["services:", ""]

    for node in nodes:
        node_name = node["name"]
        service_name = f"besu-{node_name.replace('_', '-')}"

        lines.extend([
            f"  {service_name}:",
            "    build:",
            "      context: .",
            "      dockerfile: Dockerfile",
            f"    container_name: {service_name}",

            "    command:",
            "      - --genesis-file=/config/genesis.json",
            "      - --node-private-key-file=/config/key.priv",
            "      - --data-path=/data",
            "      - --p2p-enabled=true",
            "      - --p2p-host=0.0.0.0",
            f"      - --p2p-port={P2P_PORT}",
            "      - --rpc-http-enabled=true",
            "      - --rpc-http-host=0.0.0.0",
            f"      - --rpc-http-port={RPC_PORT}",
            "      - --rpc-http-api=ETH,NET,QBFT,WEB3",
            "      - --host-allowlist=*",
            "      - --min-gas-price=0",
            "      - --logging=INFO",
            "      - --static-nodes-file=/config/static-nodes.json",

            "    volumes:",
            f"      - ./network/nodes/{node_name}/genesis.json:/config/genesis.json:ro",
            f"      - ./network/nodes/{node_name}/key.priv:/config/key.priv:ro",
            f"      - ./network/nodes/{node_name}/static-nodes.json:/config/static-nodes.json:ro",
            f"      - besu_{node_name}_data:/data",

            "    ports:",
            f'      - "{node["rpc_host_port"]}:{RPC_PORT}"',
            f'      - "{node["p2p_host_port"]}:{P2P_PORT}"',
        ])

        # Tutti i client aspettano che il server sia healthy
        if node_name != "server":
            lines.extend([
                "    depends_on:",
                "      besu-server:",
                "        condition: service_healthy",
            ])

        lines.extend([
            "    networks:",
            "      blockchain:",
            f"        ipv4_address: {node['docker_ip']}",
            ""
        ])

    lines.extend([
        "  contract-deployer:",
        "    build:",
        "      context: .",
        "      dockerfile: Dockerfile.deployer",
        "    container_name: contract-deployer",

        "    environment:",
        "      BESU_RPC_URL: http://besu-server:8545",
        "      BESU_PRIVATE_KEY_PATH: /keys/server-key.priv",

        "    volumes:",
        "      - ./network/nodes/server/key.priv:/keys/server-key.priv:ro",
        "      - ./contract_address.json:/app/contract_address.json",

        "    depends_on:",
    ])

    # nodi besu dei client devono essere pronti prima di avviare contract-deployer.
    # N.B: i client aspettano il server. Quindi se il deployer aspetta i client è sicuro che ci sia anche il server
    for node in nodes:
        node_name = node["name"]

        if node_name != "server":
            service_name = f"besu-{node_name.replace('_', '-')}"
            lines.extend([
                f"      {service_name}:",
                "        condition: service_healthy",
            ])

    lines.extend([
        "    networks:",
        "      - blockchain",
        "",
        "networks:",
        "  blockchain:",
        "    name: securecare-blockchain",
        "    ipam:",
        "      config:",
        f"        - subnet: {DOCKER_SUBNET}",
        "",
        "volumes:"
    ])

    for node in nodes:
        lines.append(f"  besu_{node['name']}_data:")

    DOCKER_COMPOSE_FILE.write_text("\n".join(lines))

    print(f"Docker Compose generated: {DOCKER_COMPOSE_FILE}")


def main():

    print()
    print("=" * 70)
    print("SECURECARE - BLOCKCHAIN SETUP")
    print("=" * 70)
    print()

    print(f"Clients:     {NUM_CLIENTS}")
    print("Server:      1")
    print(f"Total nodes: {NUM_NODES}")
    print()


    prepare_directories()

    create_config()

    generate_blockchain()

    key_directories = (get_key_directories())

    fund_accounts(key_directories)

    nodes = (prepare_node_directories(key_directories))

    create_identity_mapping(nodes)

    create_static_nodes(nodes)

    create_docker_compose(nodes)



if __name__ == "__main__":
    main()