import hashlib
import json
import numpy as np
import time
import requests
from web3 import Web3


def create_client_audit_record(
        client_id,
        partition_id,
        server_round,
        metrics,
        reputation,
        decision,
        reason,
        local_model_hash
):
    return {
        "client_id": client_id,
        "partition_id": partition_id,
        "round": server_round,
        "timestamp": int(time.time()),

        "local_model_hash": local_model_hash,

        "performance": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        },

        "reputation": {
            "combined_score": reputation["combined"],
            "metrics": {
                key: value
                for key, value in reputation.items()
                if key != "combined"
            }
        },

        "decision": decision,
        "reason": reason
    }


def hash_audit_record(record):
    record_json = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        default = lambda x: x.item() if isinstance(x, np.generic) else x
    )

    return hashlib.sha256(
        record_json.encode("utf-8")
    ).hexdigest()

def hash_model_weights(weights):
    hasher = hashlib.sha256()

    for layer in weights:
        hasher.update(layer.tobytes())

    return hasher.hexdigest()


def build_merkle_tree(audit_hashes):
    """
    Costruisce il Merkle Tree a partire dai record di audit
    dei client e restituisce la Merkle Root.
    """

    if not audit_hashes:
        return None

    hashes = audit_hashes.copy()

    while len(hashes) > 1:

        # Se il numero di nodi è dispari, duplichiamo l'ultimo hash.
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])

        next_level = []

        for i in range(0, len(hashes), 2):

            combined = hashes[i] + hashes[i + 1]

            parent_hash = hashlib.sha256(
                combined.encode("utf-8")
            ).hexdigest()

            next_level.append(parent_hash)

        hashes = next_level

    return hashes[0]

def upload_audit_records_to_ipfs(records, ipfs_api_url):

    response = requests.post(
        f"{ipfs_api_url}/api/v0/add",
        files={
            "file": (
                "audit.json",
                json.dumps(
                    records,
                    default=lambda x: x.item()
                    if isinstance(x, np.generic)
                    else x
                ).encode("utf-8")
            )
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["Hash"]


def create_round_transaction(
        server_round,
        n_clients_accepted,
        n_clients_rejected,
        global_model_hash,
        merkle_root,
        cid_transactions
):
    return {
        "round": server_round,
        "timestamp": int(time.time()),

        "n_clients_accepted": n_clients_accepted,
        "n_clients_rejected": n_clients_rejected,

        "global_model_hash": global_model_hash,

        "merkle_root": merkle_root,

        "cid_transactions": cid_transactions
    }


def verify_audit(
        server_round,
        blockchain_rpc_url,
        ipfs_api_url,
        contract_address,
        contract_abi
):
    """
    Verifica indipendentemente l'audit di un round.
    """


    # connessione al corrispondente nodo della blockchain
    w3 = Web3(Web3.HTTPProvider(blockchain_rpc_url))

    if not w3.is_connected():
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": False,
                "ipfs_records_retrieved": False,
                "record_hashes_computed": False,
                "round_consistency": False,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": "Unable to connect to blockchain"
        }

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=contract_abi
    )

    # recupero audit relativo allo specifico round dalla blockchain

    try:
        audit = contract.functions.getAudit(server_round).call()
        blockchain_audit_retrieved = True
    except Exception as e:
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": False,
                "ipfs_records_retrieved": False,
                "record_hashes_computed": False,
                "round_consistency": False,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": f"Unable to retrieve audit from blockchain: {e}"
        }

    blockchain_round = audit[0]
    n_clients_accepted = audit[2]
    n_clients_rejected = audit[3]
    blockchain_merkle_root = audit[5].hex()
    cid_transactions = audit[6]

    # check sul numero di round
    round_consistent = (
            blockchain_round == server_round
    )

    if not round_consistent:
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": blockchain_audit_retrieved,
                "ipfs_records_retrieved": False,
                "record_hashes_computed": False,
                "round_consistency": False,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": "Round mismatch"
        }

    # recupero json da IPFS

    try:
        response = requests.post(
            f"{ipfs_api_url}/api/v0/cat",
            params={"arg": cid_transactions}
        )

        response.raise_for_status()

        records = response.json()

        ipfs_records_retrieved = True

    except Exception as e:
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": blockchain_audit_retrieved,
                "ipfs_records_retrieved": False,
                "record_hashes_computed": False,
                "round_consistency": round_consistent,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": f"Unable to retrieve audit records from IPFS: {e}"
        }

    if not isinstance(records, list):
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": blockchain_audit_retrieved,
                "ipfs_records_retrieved": ipfs_records_retrieved,
                "record_hashes_computed": False,
                "round_consistency": round_consistent,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": "Invalid audit data retrieved from IPFS"
        }

    if not records:
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": blockchain_audit_retrieved,
                "ipfs_records_retrieved": ipfs_records_retrieved,
                "record_hashes_computed": False,
                "round_consistency": round_consistent,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": "Audit contains no records"
        }

    # controllo coerenza dei round dei record del json

    round_consistent = all(
        record.get("round") == server_round
        for record in records
    )

    # calcolo degli hash dei record nel json
    try:

        audit_hashes = [
            hash_audit_record(record)
            for record in records
        ]

        record_hashes_computed = (len(audit_hashes) == len(records))

    except Exception:
        record_hashes_computed = False
        audit_hashes = []

    if not record_hashes_computed:
        return {
            "verified": False,
            "round": server_round,
            "checks": {
                "blockchain_audit_retrieved": blockchain_audit_retrieved,
                "ipfs_records_retrieved": ipfs_records_retrieved,
                "record_hashes_computed": False,
                "round_consistency": round_consistent,
                "merkle_root_matches_blockchain": False,
                "accepted_clients_count_matches_blockchain": False,
                "rejected_clients_count_matches_blockchain": False,
                "total_clients_count_matches_blockchain": False
            },
            "clients": {
                "included": [],
                "excluded": []
            },
            "error": "Unable to compute audit record hashes"
        }

    # ricostruzione merkle tree
    calculated_merkle_root = build_merkle_tree(audit_hashes)

    # check sulla merkle root

    blockchain_merkle_root = blockchain_merkle_root.lower()
    calculated_merkle_root = calculated_merkle_root.lower()

    merkle_verified = (calculated_merkle_root == blockchain_merkle_root)

    # identificazione client inclusi ed esclusi

    included_clients = []
    excluded_clients = []

    for record in records:

        client_info = {
            "client_id": record.get("client_id"),
            "partition_id": record.get("partition_id")
        }

        if record.get("decision") == "ACCEPTED":

            included_clients.append(client_info)

        elif record.get("decision") == "REJECTED":

            reason = record.get("reason", {})

            negative_metrics = [
                metric.get("metric")
                for metric in reason.get("negative_metrics", [])
            ]

            client_info["reason"] = reason.get("motivation")
            client_info["negative_metrics"] = negative_metrics

            excluded_clients.append(client_info)

    # verifica sul conteggio dei client inclusi/esclusi e totali
    accepted_count_verified = (len(included_clients) == n_clients_accepted)
    rejected_count_verified = (len(excluded_clients) == n_clients_rejected)
    total_count_verified = (len(records)== n_clients_accepted + n_clients_rejected)

    # verifica che tutti i check siano soddisfatti

    verified = all([
        blockchain_audit_retrieved,
        ipfs_records_retrieved,
        record_hashes_computed,
        round_consistent,
        merkle_verified,
        accepted_count_verified,
        rejected_count_verified,
        total_count_verified
    ])


    return {
        "verified": verified,

        "round": server_round,

        "checks": {
            "blockchain_audit_retrieved":blockchain_audit_retrieved,
            "ipfs_records_retrieved":ipfs_records_retrieved,
            "record_hashes_computed":record_hashes_computed,
            "round_consistency": round_consistent,
            "merkle_root_matches_blockchain": merkle_verified,
            "accepted_clients_count_matches_blockchain": accepted_count_verified,
            "rejected_clients_count_matches_blockchain": rejected_count_verified,
            "total_clients_count_matches_blockchain": total_count_verified
        },

        "clients": {
            "included": included_clients,
            "excluded": excluded_clients
        }
    }