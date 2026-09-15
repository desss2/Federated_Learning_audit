import { network } from "hardhat";
import fs from "fs";
import path from "path";

async function main() {

    // Connessione alla rete Besu
    const { ethers } = await network.connect();

    console.log("Deploying AuditContract...");

    const privateKeyPath = process.env.BESU_PRIVATE_KEY_PATH ||
    path.join(
        process.cwd(),
        "network",
        "nodes",
        "server",
        "key.priv"
    );

    const privateKey = fs.readFileSync(
        privateKeyPath,
        "utf8"
    ).trim();

    const provider = new ethers.JsonRpcProvider(
        process.env.BESU_RPC_URL || "http://127.0.0.1:8545"
    );

    const wallet = new ethers.Wallet(
        privateKey,
        provider
    );

    const AuditContract = await ethers.getContractFactory(
        "AuditContract",
        wallet
    );

    const auditContract = await AuditContract.deploy();

    const deploymentTx = auditContract.deploymentTransaction();

    console.log("Deployment transaction sent.");
    console.log("Transaction hash:", deploymentTx.hash);

    console.log("Waiting for transaction to be included in a block...");

    const receipt = await deploymentTx.wait();

    console.log("Transaction included!");
    console.log("Block:", receipt.blockNumber);

    const address = await auditContract.getAddress();


    console.log("--------------------------------");
    console.log("AuditContract deployed!");
    console.log("Address:", address);
    console.log("--------------------------------");


    // Salvo automaticamente l'indirizzo
    const addressFile = path.join(
        process.cwd(),
        "contract_address.json"
    );

    fs.writeFileSync(
        addressFile,
        JSON.stringify(
            {
                address: address
            },
            null,
            4
        )
    );

    console.log(
        "Address saved in contract_address.json"
    );
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });