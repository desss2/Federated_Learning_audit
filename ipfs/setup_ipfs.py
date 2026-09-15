import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from globals import NUM_CLIENTS

IPFS_DIR = Path(__file__).resolve().parent
NETWORK_DIR = IPFS_DIR / "network"
IDENTITY_MAPPING_FILE = PROJECT_ROOT / "identity_mapping.json"
DOCKER_COMPOSE_FILE = IPFS_DIR / "docker-compose.yml"

NUM_NODES = NUM_CLIENTS + 1

API_PORT = 5001
GATEWAY_PORT = 8080
SWARM_PORT = 4001

DOCKER_SUBNET = "172.29.0.0/16"
DOCKER_IP_BASE = 2

# Porte esposte sull'host
API_HOST_BASE_PORT = 5001
GATEWAY_HOST_BASE_PORT = 8080
SWARM_HOST_BASE_PORT = 4001


def run_command(command, env=None):
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Comando fallito: {' '.join(command)}")
    return result.stdout.strip()


def get_node_name(node_id):
    return "server" if node_id == 0 else f"client_{node_id - 1}"

# crea una cartella per ogni nodo nella directory network
def create_node_repository(node_id):
    node_name = get_node_name(node_id)
    node_directory = NETWORK_DIR / node_name

    if node_directory.exists():
        shutil.rmtree(node_directory)

    node_directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["IPFS_PATH"] = str(node_directory)

    print(f"Initializing {node_name}...")
    run_command(["ipfs", "init", "--profile", "server"], env=env)

    return node_directory


def configure_node(node_id, node_directory, bootstrap_addresses):
    env = os.environ.copy()
    env["IPFS_PATH"] = str(node_directory)

    run_command(["ipfs", "config", "Addresses.API", f"/ip4/0.0.0.0/tcp/{API_PORT}"],env=env)

    run_command(["ipfs", "config", "Addresses.Gateway",f"/ip4/0.0.0.0/tcp/{GATEWAY_PORT}"],env=env)

    run_command(["ipfs", "config", "--json", "Addresses.Swarm",f'["/ip4/0.0.0.0/tcp/{SWARM_PORT}"]'],env=env)

    run_command(["ipfs", "config", "--json", "Swarm.AddrFilters", "[]"],env=env)

    run_command(["ipfs", "bootstrap", "rm", "--all"],env=env)

    for address in bootstrap_addresses:
        run_command(["ipfs", "bootstrap", "add", address],env=env)


def get_peer_id(node_directory):
    env = os.environ.copy()
    env["IPFS_PATH"] = str(node_directory)
    return run_command(["ipfs", "id", "-f", "<id>"], env=env)


def create_identity_mapping(nodes):
    with open(IDENTITY_MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    for node in nodes:
        name = node["name"]
        peer_id = node["peer_id"]

        if name == "server":
            mapping["server"]["ipfs_node"] = name
            mapping["server"]["ipfs_peer_id"] = peer_id
        else:
            if name not in mapping["clients"]:
                raise ValueError(f"Nodo IPFS non trovato nel mapping: {name}")

            mapping["clients"][name]["ipfs_node"] = name
            mapping["clients"][name]["ipfs_peer_id"] = peer_id

    with open(IDENTITY_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
        f.write("\n")

    print(f"Identity mapping updated: {IDENTITY_MAPPING_FILE}")


def create_docker_compose(nodes):
    lines = ["services:", ""]

    for node in nodes:
        name = node["name"]
        service = f"ipfs-{name.replace('_', '-')}"

        lines.extend([
            f"  {service}:",
            "    image: ipfs/kubo:v0.42.0",
            f"    container_name: {service}",
            "    restart: unless-stopped",
            "    command: [\"daemon\", \"--migrate=true\"]",
            "    volumes:",
            f"      - ./network/{name}:/data/ipfs",
            "    environment:",
            "      IPFS_PATH: /data/ipfs",
            "    ports:",
            f'      - "{node["api_host_port"]}:{API_PORT}"',
            f'      - "{node["gateway_host_port"]}:{GATEWAY_PORT}"',
            f'      - "{node["swarm_host_port"]}:{SWARM_PORT}"',
            "    healthcheck:",
            "      test: [\"CMD-SHELL\", \"ipfs id >/dev/null 2>&1\"]",
            "      interval: 2s",
            "      timeout: 5s",
            "      retries: 15",
            "    networks:",
            "      ipfs:",
            f'        ipv4_address: {node["docker_ip"]}',
            ""
        ])

    lines.extend([
        "networks:",
        "  ipfs:",
        "    name: securecare-ipfs",
        "    ipam:",
        "      config:",
        f"        - subnet: {DOCKER_SUBNET}",
        ""
    ])

    DOCKER_COMPOSE_FILE.write_text("\n".join(lines))
    print(f"Docker Compose generated: {DOCKER_COMPOSE_FILE}")

def main():
    print("=" * 60)
    print("SECURECARE - IPFS SETUP")
    print("=" * 60)

    if NETWORK_DIR.exists():
        print(f"Removing existing network: {NETWORK_DIR}")
        shutil.rmtree(NETWORK_DIR)

    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    nodes = []

    for node_id in range(NUM_NODES):
        node_name = get_node_name(node_id)
        node_directory = create_node_repository(node_id)

        nodes.append({
            "id": node_id,
            "name": node_name,
            "directory": node_directory,
            "peer_id": get_peer_id(node_directory),
            "docker_ip": f"172.29.0.{DOCKER_IP_BASE + node_id}",
            "api_port": API_PORT,
            "gateway_port": GATEWAY_PORT,
            "swarm_port": SWARM_PORT,
            "api_host_port": API_HOST_BASE_PORT + node_id,
            "gateway_host_port": GATEWAY_HOST_BASE_PORT + node_id,
            "swarm_host_port": SWARM_HOST_BASE_PORT + node_id
        })

    for node in nodes:
        bootstrap_addresses = []

        for target in nodes:
            if target["id"] == node["id"]:
                continue

            bootstrap_addresses.append(
                f"/ip4/{target['docker_ip']}"
                f"/tcp/{SWARM_PORT}"
                f"/p2p/{target['peer_id']}"
            )

        configure_node(
            node["id"],
            node["directory"],
            bootstrap_addresses
        )

    create_identity_mapping(nodes)
    create_docker_compose(nodes)

    print("\nIPFS network setup completed.")


if __name__ == "__main__":
    main()