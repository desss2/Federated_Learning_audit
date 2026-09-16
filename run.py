import subprocess
import sys
import time
import json

from globals import NUM_CLIENTS, USE_LABEL_NOISE, ROUNDS


def run_fl():
    print("\n=== FEDERATED LEARNING ===")

    start_time = time.perf_counter()

    result = subprocess.run(
        ["flwr", "run", ".", "local-deployment", "--stream"],
        check=False
    )

    end_time = time.perf_counter()

    fl_time = end_time - start_time

    print(f"Total execution time: {fl_time:.2f} seconds")

    if USE_LABEL_NOISE:
        report_path = "results/fl_execution_time_noisy.txt"
    else:
        report_path = "results/fl_execution_time_clean.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("=== FEDERATED LEARNING EXECUTION TIME ===\n")
        report.write(f"Number of clients: {NUM_CLIENTS}\n")
        report.write(f"Number of rounds: {ROUNDS}\n")
        report.write(f"Execution time: {fl_time:.2f} seconds\n")
        report.write(f"Execution time: {fl_time / 60:.2f} minutes\n")

    if result.returncode != 0:
        print("\n[ERROR] Federated Learning failed.")
        sys.exit(result.returncode)

def verify_clients():
    print("\n=== AUDIT VERIFICATION ===")

    processes = []

    for partition_id in range(NUM_CLIENTS):
        service_name = f"superexec-clientapp-{partition_id + 1}"

        process = subprocess.Popen(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                service_name,
                "python",
                "verify_client.py",
                f"client_{partition_id}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        processes.append((partition_id, process))

    outputs = []
    failed = []

    for partition_id, process in processes:
        output, _ = process.communicate()

        outputs.append((partition_id, output))

        if process.returncode != 0:
            failed.append(partition_id)
            
    for partition_id, output in outputs:

        header = (
            f"\n{'=' * 20} "
            f"CLIENT {partition_id} "
            f"{'=' * 20}\n"
        )

        print(header, end="")
        print(output, end="")
        
    if failed:
        message = f"\n[ERROR] Verification failed for clients: {failed}\n"

        print(message, end="")

        return 1
            
    aggregated_results = []

    if USE_LABEL_NOISE:
        aggregated_path = "results/audit_verification_noisy.json"
        suffix = "noisy"
    else:
        aggregated_path = "results/audit_verification_clean.json"
        suffix = "clean"

    for partition_id in range(NUM_CLIENTS):

        client_name = f"client_{partition_id}"

        local_json_path = (
            f"results/"
            f"{client_name}_audit_verification_{suffix}.json"
        )

        # Legge direttamente il JSON generato dal client.
        # La cartella results è condivisa tramite volume Docker.
        try:
            with open(local_json_path, "r", encoding="utf-8") as f:
                client_result = json.load(f)

            aggregated_results.append(client_result)
        except Exception as e:

            print(
                f"\n[ERROR] Unable to read verification "
                f"file for {client_name}: {e}"
            )

            return 1

    aggregated_data = {
        "number_of_clients": NUM_CLIENTS,
        "number_of_rounds": ROUNDS,
        "clients": aggregated_results
    }

    with open(aggregated_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, indent=4)

    print(
        f"\n[VERIFY AUDIT] Aggregated results saved to "
        f"{aggregated_path}"
    )

    print("\n=== AUDIT VERIFICATION COMPLETED ===")

    return 0


def main():
    run_fl()
    sys.exit(verify_clients())


if __name__ == "__main__":
    main()