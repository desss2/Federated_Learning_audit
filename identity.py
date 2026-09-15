import json
from globals import *


def update_identity_mapping(client_id, partition_id):

    with open(IDENTITY_MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    node_name = f"client_{partition_id}"

    if node_name not in mapping["clients"]:
        raise ValueError(
            f"Nodo blockchain non trovato: {node_name}"
        )

    mapping["clients"][node_name]["partition_id"] = partition_id
    mapping["clients"][node_name]["client_id"] = client_id

    with open(IDENTITY_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"[IDENTITY] partition_id={partition_id} → client_id={client_id} ")