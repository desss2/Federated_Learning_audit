// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AuditContract {

    struct ClientAudit {
        string clientId;
        bytes32 hash;
        bool aggregationDecision;
        uint256 reputation;
    }

    struct Audit {

        uint256 round;
        uint256 timestamp;
        uint256 nClientsAccepted;
        uint256 nClientsRejected;

        bytes32 globalModelHash;
        bytes32 merkleRoot;

        string cidTransactions;
    }

    // round -> audit
    mapping(uint256 => Audit) private audits;

    // Permette di verificare se un round è già stato registrato
    mapping(uint256 => bool) private roundRecorded;

    uint256 constant REPUTATION_SCALE = 10000;
    uint256 constant REPUTATION_THRESHOLD = 3000;

    event AuditRecorded(
        uint256 indexed round,
        uint256 timestamp,
        uint256 nClientsAccepted,
        uint256 nClientsRejected,
        bytes32 globalModelHash,
        bytes32 merkleRoot,
        string cidTransactions
    );


    function recordAudit(
        uint256 round,
        ClientAudit[] calldata clientAudits,
        uint256 nClientsAccepted,
        uint256 nClientsRejected,
        bytes32 globalModelHash,
        bytes32 merkleRoot,
        string calldata cidTransactions
    ) external {

        require(
            !roundRecorded[round],
            "Audit for this round already exists"
        );

        require(
                clientAudits.length > 0,
                "No client audits provided"
            );

        // Controllo degli hash registrati dai client rispetto agli hash calcolati dal server
        for (uint256 i = 0; i < clientAudits.length; i++) {

            ClientAudit calldata clientAudit = clientAudits[i];

            ClientUpdate memory clientUpdate =
                clientUpdates[round][clientAudit.clientId];

            require(
                clientUpdate.timestamp != 0,
                "Client update does not exist"
            );

            bytes32 clientHash = clientUpdate.updateHash;

            require(
                clientHash == clientAudit.hash,
                "Client hash does not match server hash"
            );

            if (clientAudit.aggregationDecision) {
                require(
                    clientAudit.reputation > REPUTATION_THRESHOLD,
                    "Accepted client has insufficient reputation"
                );
            } else {
                require(
                    clientAudit.reputation <= REPUTATION_THRESHOLD,
                    "Rejected client has sufficient reputation"
                );
            }
        }


        // Se tutti i controlli sono superati,
        // viene registrato l'audit del round
        audits[round] = Audit({
            round: round,
            timestamp: block.timestamp,
            nClientsAccepted: nClientsAccepted,
            nClientsRejected: nClientsRejected,
            globalModelHash: globalModelHash,
            merkleRoot: merkleRoot,
            cidTransactions: cidTransactions
        });



        roundRecorded[round] = true;

        emit AuditRecorded(
            round,
            block.timestamp,
            nClientsAccepted,
            nClientsRejected,
            globalModelHash,
            merkleRoot,
            cidTransactions
        );
    }

    // restituisce solo i dati della transazione, non quelli relativi ai client
    function getAudit(uint256 round)
        external
        view
        returns (
            uint256 auditRound,
            uint256 timestamp,
            uint256 nClientsAccepted,
            uint256 nClientsRejected,
            bytes32 globalModelHash,
            bytes32 merkleRoot,
            string memory cidTransactions
        )
    {
        require(
            roundRecorded[round],
            "Audit for this round does not exist"
        );

        Audit storage audit = audits[round];

        return (
            audit.round,
            audit.timestamp,
            audit.nClientsAccepted,
            audit.nClientsRejected,
            audit.globalModelHash,
            audit.merkleRoot,
            audit.cidTransactions
        );
    }


    function auditExists(uint256 round)
        external
        view
        returns (bool)
    {
        return roundRecorded[round];
    }


    // TRANSAZIONE PER I CLIENT

    struct ClientUpdate {
        bytes32 updateHash;
        uint256 timestamp;
    }

    mapping(uint256 => mapping(string => ClientUpdate)) private clientUpdates;

    function registerClientUpdate(
        string calldata clientId,
        uint256 round,
        bytes32 updateHash
    ) external {

        require(
            clientUpdates[round][clientId].timestamp == 0,
            "Client update for this round already exists"
        );

        clientUpdates[round][clientId] = ClientUpdate({
            updateHash: updateHash,
            timestamp: block.timestamp
        });

        emit ClientUpdateRegistered(
            clientId,
            round,
            updateHash,
            block.timestamp
        );
    }


    function getClientUpdate(
        string calldata clientId,
        uint256 round
    )
        external
        view
        returns (bytes32 updateHash, uint256 timestamp)
    {
        require(
            clientUpdates[round][clientId].timestamp != 0,
            "Client update does not exist"
        );

        ClientUpdate memory update = clientUpdates[round][clientId];

        return (
            update.updateHash,
            update.timestamp
        );
    }

    event ClientUpdateRegistered(
        string indexed clientId,
        uint256 indexed round,
        bytes32 updateHash,
        uint256 timestamp
    );


}