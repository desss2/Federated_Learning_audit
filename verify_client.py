import sys
import json

from globals import (
    ROUNDS,
    get_blockchain_rpc_url,
    get_ipfs_api_url,
    BLOCKCHAIN_SMART_CONTRACT
)

from audit import verify_audit


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
    }
]


def print_verification_result(result):

    round_num = result["round"]

    print()
    print(f"[VERIFY AUDIT] Round {round_num}")
    print()

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

        print(f"  [client_id={client['client_id']} partition={client['partition_id']}")
        print(f"   reason={client['reason']}")
        print(f"   negative_metrics=[{metrics}]")

    print()

    # Client inclusi
    included = result["clients"]["included"]

    print(f"Client included: {len(included)}")

    for client in included:

        print(f"  [client_id={client['client_id']}/partition={client['partition_id']}]")

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
        result = verify_audit(
            server_round=round_num,
            blockchain_rpc_url=blockchain_rpc_url,
            ipfs_api_url=ipfs_api_url,
            contract_address=contract_address,
            contract_abi=AUDIT_READ_ABI
        )

        print_verification_result(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python verify_client.py <client_name>"
        )

    client_name = sys.argv[1]

    verify_client_audits(client_name)