import sys
import json
import os
from globals import (
    ROUNDS,
    get_blockchain_rpc_url,
    get_ipfs_api_url,
    BLOCKCHAIN_SMART_CONTRACT, USE_LABEL_NOISE
)

from audit import verify_audit, check_audit_exists

AUDIT_READ_ABI = [
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            }
        ],
        "name": "getAudit",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "auditRound",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "nClientsAccepted",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "nClientsRejected",
                "type": "uint256"
            },
            {
                "internalType": "bytes32",
                "name": "globalModelHash",
                "type": "bytes32"
            },
            {
                "internalType": "bytes32",
                "name": "merkleRoot",
                "type": "bytes32"
            },
            {
                "internalType": "string",
                "name": "cidTransactions",
                "type": "string"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            }
        ],
        "name": "auditExists",
        "outputs": [
            {
                "internalType": "bool",
                "name": "",
                "type": "bool"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]


def print_verification_result(result):

    round_num = result["round"]

    print()
    print(f"[VERIFY AUDIT] Round {round_num}")
    print()

    # Audit non registrato
    if result.get("status") == "NOT_RECORDED":

        print("✗ Audit was not recorded on blockchain")
        print()
        print("✗ AUDIT VERIFICATION FAILED")
        print()

        return

    # Errore nel controllo della blockchain
    if result.get("status") == "CHECK_FAILED":

        print("✗ Unable to check audit existence on blockchain")
        print(f"  Error: {result['error']}")
        print()
        print("✗ AUDIT VERIFICATION FAILED")
        print()

        return

    # Verifica completa
    checks = result["checks"]

    # Blockchain
    if checks["round_consistency"]:
        print("✓ Retrieve audit from blockchain successfully")
    else:
        print("✗ Retrieve audit from blockchain failed")

    # IPFS
    if checks["ipfs_records_retrieved"]:
        print("✓ Retrieve audit records from IPFS successfully")
    else:
        print("✗ Retrieve audit records from IPFS failed")

    # Record hashes
    if checks["record_hashes_computed"]:
        print("✓ Evaluation of record hashes successfully")
    else:
        print("✗ Evaluation of record hashes failed")

    # Merkle Root
    if checks["merkle_root_matches_blockchain"]:
        print("✓ Merkle Root matches blockchain")
    else:
        print("✗ Merkle Root does not match blockchain")

    print()

    # Client esclusi
    excluded = result["clients"]["excluded"]

    print(f"Client excluded: {len(excluded)}")

    for client in excluded:

        metrics = ", ".join(client["negative_metrics"])

        print(
            f"  [client_id={client['client_id']} "
            f"partition={client['partition_id']}]"
        )
        print(f"   reason={client['reason']}")
        print(f"   negative_metrics=[{metrics}]")

    print()

    # Client inclusi
    included = result["clients"]["included"]

    print(f"Client included: {len(included)}")

    for client in included:
        print(
            f"  [client_id={client['client_id']}"
            f"/partition={client['partition_id']}]"
        )

    print()

    # Risultato finale
    if result["verified"]:
        print("✓ AUDIT VERIFIED")
    else:
        print("✗ AUDIT VERIFICATION FAILED")

    print()


def verify_client_audits(client_name):

    blockchain_rpc_url = get_blockchain_rpc_url(client_name)
    ipfs_api_url = get_ipfs_api_url(client_name)

    with open(BLOCKCHAIN_SMART_CONTRACT) as f:
        contract_address = json.load(f)["address"]

    results = []

    for round_num in range(1, ROUNDS + 1):
        existence = check_audit_exists(
            server_round=round_num,
            blockchain_rpc_url=blockchain_rpc_url,
            contract_address=contract_address,
            contract_abi=AUDIT_READ_ABI
        )
        
        if not existence["checked"]:

            result = {
                "verified": False,
                "round": round_num,
                "status": "CHECK_FAILED",
                "error": existence["error"]
            }

            print_verification_result(result)
            results.append(result)
            continue
            
        if not existence["exists"]:

            result = {
                "verified": False,
                "round": round_num,
                "status": "NOT_RECORDED",
                "error": "Audit was not recorded on blockchain"
            }

            print_verification_result(result)

            results.append(result)
            continue

        result = verify_audit(
            server_round=round_num,
            blockchain_rpc_url=blockchain_rpc_url,
            ipfs_api_url=ipfs_api_url,
            contract_address=contract_address,
            contract_abi=AUDIT_READ_ABI
        )

        result["status"] = "RECORDED"

        print_verification_result(result)

        results.append(result)
        
    # Salvataggio del risultato finale del singolo client
    if USE_LABEL_NOISE:
        output_path = f"results/{client_name}_audit_verification_noisy.json"
    else:
        output_path = f"results/{client_name}_audit_verification_clean.json"

    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "client_id": client_name,
                "rounds": results
            },
            f,
            indent=4
        )
    
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python verify_client.py <client_name>"
        )

    client_name = sys.argv[1]

    verify_client_audits(client_name)