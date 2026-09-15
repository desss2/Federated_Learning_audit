import { defineConfig } from "hardhat/config";
import hardhatEthers from "@nomicfoundation/hardhat-ethers";

export default defineConfig({
    plugins: [hardhatEthers],

    solidity: {
        version: "0.8.20",
        settings: {
            evmVersion: "paris"
        }
    },

    networks: {
        besu: {
            type: "http",
            url: process.env.BESU_RPC_URL || "http://127.0.0.1:8545"
        }
    }
});