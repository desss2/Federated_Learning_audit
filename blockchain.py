import json
from pathlib import Path
import time

from web3 import Web3

from globals import *

AUDIT_ABI = [
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            },
            {
                "internalType": "tuple[]",
                "name": "clientAudits",
                "type": "tuple[]",
                "components": [
                    {
                        "internalType": "string",
                        "name": "clientId",
                        "type": "string"
                    },
                    {
                        "internalType": "bytes32",
                        "name": "hash",
                        "type": "bytes32"
                    },
                    {
                        "internalType": "bool",
                        "name": "aggregationDecision",
                        "type": "bool"
                    },
                    {
                        "internalType": "uint256",
                        "name": "reputation",
                        "type": "uint256"
                    }
                ]
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
        "name": "recordAudit",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


CLIENT_UPDATE_ABI = [
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "clientId",
                "type": "string"
            },
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            },
            {
                "internalType": "bytes32",
                "name": "updateHash",
                "type": "bytes32"
            }
        ],
        "name": "registerClientUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "clientId",
                "type": "string"
            },
            {
                "internalType": "uint256",
                "name": "round",
                "type": "uint256"
            }
        ],
        "name": "getClientUpdate",
        "outputs": [
            {
                "internalType": "bytes32",
                "name": "updateHash",
                "type": "bytes32"
            },
            {
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]


def record_audit_on_blockchain(round_transaction, client_audits, blockchain_rpc_url, private_key):

    w3 = Web3(
        Web3.HTTPProvider(blockchain_rpc_url)
    )

    if not w3.is_connected():
        raise RuntimeError(
            "Impossibile connettersi alla blockchain Besu"
        )

    with open(BLOCKCHAIN_SMART_CONTRACT) as f:
        contract_address = json.load(f)["address"]

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(
            contract_address
        ),
        abi=AUDIT_ABI
    )

    account = w3.eth.account.from_key(
        private_key
    )

    sender = account.address

    REPUTATION_SCALE = 10_000

    client_audits_tuple = [
        (
            audit["clientId"],
            bytes.fromhex(audit["hash"]),
            audit["aggregationDecision"],
            int(round(audit["reputation"] * REPUTATION_SCALE))
        )
        for audit in client_audits
    ]

    tx = contract.functions.recordAudit(
        round_transaction["round"],
        client_audits_tuple,
        round_transaction["n_clients_accepted"],
        round_transaction["n_clients_rejected"],
        bytes.fromhex(
            round_transaction["global_model_hash"]
        ),
        bytes.fromhex(
            round_transaction["merkle_root"]
        ),
        round_transaction["cid_transactions"]
    ).build_transaction({
        "from": sender,
        "nonce": w3.eth.get_transaction_count(
            sender,
            "pending"
        ),
        "gas": 500000,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id
    })

    signed_tx = w3.eth.account.sign_transaction(
        tx,
        private_key
    )
    
    
    tx_start = time.perf_counter()

    tx_hash = w3.eth.send_raw_transaction(
        signed_tx.raw_transaction
    )
    
    send_time = time.perf_counter() - tx_start
    
    confirmation_start = time.perf_counter()

    receipt = w3.eth.wait_for_transaction_receipt(
        tx_hash
    )
    
    confirmation_time = time.perf_counter() - confirmation_start
    gas_used = receipt.gasUsed

    return tx_hash.hex(), receipt.blockNumber, receipt.status, gas_used, send_time, confirmation_time


def register_client_update_on_blockchain(
    client_update,
    blockchain_rpc_url,
    private_key
):

    w3 = Web3(
        Web3.HTTPProvider(blockchain_rpc_url)
    )

    if not w3.is_connected():
        raise RuntimeError(
            "Impossibile connettersi alla blockchain Besu"
        )

    with open(BLOCKCHAIN_SMART_CONTRACT) as f:
        contract_address = json.load(f)["address"]

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=CLIENT_UPDATE_ABI
    )

    account = w3.eth.account.from_key(
        private_key
    )

    sender = account.address

    update_hash_bytes = bytes.fromhex(
        client_update["update_hash"]
    )

    if len(update_hash_bytes) != 32:
        raise ValueError(
            "L'hash dell'update deve essere di 32 byte"
        )

    tx = contract.functions.registerClientUpdate(
        client_update["client_id"],
        client_update["round"],
        update_hash_bytes
    ).build_transaction({
        "from": sender,
        "nonce": w3.eth.get_transaction_count(
            sender,
            "pending"
        ),
        "gas": 200000,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id
    })

    signed_tx = w3.eth.account.sign_transaction(
        tx,
        private_key
    )

    tx_start = time.perf_counter()
	
    tx_hash = w3.eth.send_raw_transaction(
        signed_tx.raw_transaction
    )
    
    send_time = time.perf_counter() - tx_start
    
    confirmation_start = time.perf_counter()

    receipt = w3.eth.wait_for_transaction_receipt(
        tx_hash
    )
    
    confirmation_time = time.perf_counter() - confirmation_start
    gas_used = receipt.gasUsed
    
    return tx_hash.hex(), receipt.blockNumber, receipt.status, gas_used, send_time, confirmation_time

def get_client_update_from_blockchain(
    client_id,
    round_num,
    blockchain_rpc_url
):

    w3 = Web3(
        Web3.HTTPProvider(blockchain_rpc_url)
    )

    if not w3.is_connected():
        raise RuntimeError(
            "Impossibile connettersi alla blockchain Besu"
        )

    with open(BLOCKCHAIN_SMART_CONTRACT) as f:
        contract_address = json.load(f)["address"]

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=CLIENT_UPDATE_ABI
    )

    update_hash, timestamp = contract.functions.getClientUpdate(
        str(client_id),
        int(round_num)
    ).call()

    return update_hash.hex(), timestamp
