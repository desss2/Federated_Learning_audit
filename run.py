import subprocess
import sys
import time

from globals import NUM_CLIENTS, USE_LABEL_NOISE


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

    if USE_LABEL_NOISE:
        report_path = "results/audit_verification_noisy.npy"
    else:
        report_path = "results/audit_verification_clean.npy"

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=== AUDIT VERIFICATION ===\n")

        for partition_id, output in outputs:
            header = f"\n{'=' * 20} CLIENT {partition_id} {'=' * 20}\n"

            # Terminale
            print(header, end="")
            print(output, end="")

            # File
            report.write(header)
            report.write(output)

        if failed:
            message = f"\n[ERROR] Verification failed for clients: {failed}\n"

            print(message, end="")
            report.write(message)

            return 1

        message = "\n=== AUDIT VERIFICATION COMPLETED ===\n"

        print(message, end="")
        report.write(message)

    print("\n=== AUDIT VERIFICATION COMPLETED ===")
    return 0


def main():
    run_fl()
    sys.exit(verify_clients())


if __name__ == "__main__":
    main()