import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# ===========================================================
# VISUALIZATION AND ANALYSIS
# ===========================================================

# 1. Global metrics over rounds

def plot_global_metrics(global_metrics):

    gm = np.array(global_metrics)
    #rounds = np.arange(1, ROUNDS+1)
    rounds = np.arange(1, len(global_metrics) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Federated Learning Performance with Advanced Reputation System', fontsize=14, fontweight='bold')

    axes[0, 0].plot(rounds, gm[:,0], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].set_title('Global Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, gm[:,1], marker='s', linewidth=2, markersize=6, color='#A23B72')
    axes[0, 1].set_title('Global Precision', fontweight='bold')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rounds, gm[:,2], marker='^', linewidth=2, markersize=6, color='#F18F01')
    axes[1, 0].set_title('Global Recall', fontweight='bold')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(rounds, gm[:,3], marker='d', linewidth=2, markersize=6, color='#C73E1D')
    axes[1, 1].set_title('Global F1-Score', fontweight='bold')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 2. Reputation evolution over rounds
def plot_reputation_evolution(all_round_reputations,
    num_clients,
    reputation_threshold):

    rounds = np.arange(1,len(all_round_reputations) + 1)

    plt.figure(figsize=(14, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    for ci in range(num_clients):
        client_reps = [all_round_reputations[r][ci]['combined'] for r in range(len(all_round_reputations))]
        plt.plot(rounds, client_reps, marker='o', linewidth=2, label=f'Client {ci}', color=colors[ci])

    plt.axhline(y=reputation_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({reputation_threshold})')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Combined Reputation Score', fontsize=12)
    plt.title('Client Reputation Evolution Over Federated Rounds', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_reputation_methods(all_round_reputations,
    num_clients):

    # 3. Reputation method comparison (final round)
    methods = ['weighted_avg', 'beta_reputation', 'fuzzy_trust', 'tanh_utility',
               'exponential_decay', 'entropy_based', 'cosine_similarity']
    method_labels = ['Weighted\nAvg', 'Beta\nRep', 'Fuzzy\nTrust', 'Tanh\nUtility',
                     'Exp\nDecay', 'Entropy', 'Cosine\nSim']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(methods))
    width = 0.15

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for ci in range(num_clients):
        final_scores = [all_round_reputations[-1][ci][m] for m in methods]
        ax.bar(x + ci*width, final_scores, width, label=f'Client {ci}', color=colors[ci], alpha=0.8)

    ax.set_xlabel('Reputation Method', fontsize=12)
    ax.set_ylabel('Reputation Score', fontsize=12)
    ax.set_title('Comparison of Reputation Methods (Final Round)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(method_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred_global, n_classes, labels):
    cm = confusion_matrix(
        y_test,
        y_pred_global,
        labels=np.arange(n_classes)
    )

    print("\n=== CONFUSION MATRIX ===")
    print("Labels:", labels)
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(
        "Confusion Matrix (Global Model, Final Round)",
        fontsize=14,
        fontweight="bold"
    )
    plt.colorbar()

    tick_marks = np.arange(n_classes)

    plt.xticks(
        tick_marks,
        labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        labels
    )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(
        range(cm.shape[0]),
        range(cm.shape[1])
    ):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.show()


def plot_client_performance(client_filenames):

    for client_id, filename in enumerate(client_filenames):

        with open(filename, "r") as f:
            data = json.load(f)

        rounds_data = data["rounds"]
        rounds = [r["round"] for r in rounds_data]

        training = [r["training_time"] for r in rounds_data]
        send = [r["blockchain_send_time"] for r in rounds_data]
        confirmation = [r["blockchain_confirmation_time"] for r in rounds_data]
        cpu_avg = [r["cpu_avg_percent"] for r in rounds_data]
        cpu_max = [r["cpu_max_percent"] for r in rounds_data]
        ram_avg = [r["ram_avg_mb"] for r in rounds_data]
        ram_max = [r["ram_max_mb"] for r in rounds_data]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Client {client_id} - Performance per Round",
                     fontsize=16, fontweight="bold")

        # (a) Training time
        axes[0, 0].plot(rounds, training, marker="o")
        axes[0, 0].set_title("(a) Training Time")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Time (s)")
        axes[0, 0].set_xticks(rounds)
        axes[0, 0].grid(axis="y", alpha=0.3)

        # (b) Blockchain overhead
        ax = axes[0, 1]

        ax.plot(rounds, send, marker="o", label="Send")
        ax.plot(rounds, confirmation, marker="o", label="Confirmation")
        ax.set_title("(b) Blockchain Overhead")
        ax.set_xlabel("Round")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(rounds)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        # (c) CPU usage
        x = np.arange(len(rounds))
        width = 0.35

        axes[1, 0].bar(x - width / 2, cpu_avg, width, label="Average")
        axes[1, 0].bar(x + width / 2, cpu_max, width, label="Maximum")
        axes[1, 0].set_title("(c) CPU Usage")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("CPU Usage (%)")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(rounds)
        axes[1, 0].grid(axis="y", alpha=0.3)
        axes[1, 0].legend()

        # (d) RAM usage
        axes[1, 1].bar(x - width / 2, ram_avg, width, label="Average")
        axes[1, 1].bar(x + width / 2, ram_max, width, label="Maximum")
        axes[1, 1].set_title("(d) RAM Usage")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("RAM (MB)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(rounds)
        axes[1, 1].grid(axis="y", alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


def plot_gas_usage(client_filenames, server_filename):

    client_data = []

    for filename in client_filenames:
        with open(filename, "r") as f:
            client_data.append(json.load(f)["rounds"])

    with open(server_filename, "r") as f:
        server_data = json.load(f)["rounds"]

    rounds = [r["round"] for r in server_data]

    x = np.arange(len(rounds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, data in enumerate(client_data):
        gas = [r["gas_used"] for r in data]

        ax.bar(
            x + (i - 1) * width,
            gas,
            width,
            label=f"Client {i}"
        )

    server_gas = [r["gas_used"] for r in server_data]

    ax.bar(
        x + (len(client_data) - 1) * width,
        server_gas,
        width,
        label="Server"
    )

    ax.set_title("Gas Usage per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Gas Used")
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_server_performance(server_filename):

    with open(server_filename, "r") as f:
        data = json.load(f)

    rounds_data = data["rounds"]
    rounds = [r["round"] for r in rounds_data]

    server_round = [r["server_round_time"] for r in rounds_data]
    client_evaluation = [r["client_evaluation_time"] for r in rounds_data]
    reputation = [r["reputation_time"] for r in rounds_data]

    audit = [r["audit_creation_time"] for r in rounds_data]
    ipfs = [r["ipfs_time"] for r in rounds_data]

    blockchain_send = [r["blockchain_send_time"] for r in rounds_data]
    blockchain_confirmation = [r["blockchain_confirmation_time"] for r in rounds_data]

    cpu_avg = [r["cpu_avg_percent"] for r in rounds_data]
    cpu_max = [r["cpu_max_percent"] for r in rounds_data]

    ram_avg = [r["ram_avg_mb"] for r in rounds_data]
    ram_max = [r["ram_max_mb"] for r in rounds_data]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(
        "Server Performance per Round",
        fontsize=16,
        fontweight="bold"
    )

    # (a) Server round time
    axes[0, 0].plot(
        rounds,
        server_round,
        marker="o"
    )

    axes[0, 0].set_title("(a) Server Round Time")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Time (s)")
    axes[0, 0].set_xticks(rounds)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # (b) Client evaluation + reputation
    axes[0, 1].plot(
        rounds,
        client_evaluation,
        marker="o",
        label="Client Evaluation"
    )

    axes[0, 1].plot(
        rounds,
        reputation,
        marker="o",
        label="Reputation"
    )

    axes[0, 1].set_title("(b) Evaluation and Reputation")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].set_xticks(rounds)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].legend()

    # (c) Audit + IPFS
    axes[0, 2].plot(
        rounds,
        audit,
        marker="o",
        label="Audit Creation"
    )

    axes[0, 2].plot(
        rounds,
        ipfs,
        marker="o",
        label="IPFS"
    )

    axes[0, 2].set_title("(c) Audit and IPFS")
    axes[0, 2].set_xlabel("Round")
    axes[0, 2].set_ylabel("Time (s)")
    axes[0, 2].set_xticks(rounds)
    axes[0, 2].grid(axis="y", alpha=0.3)
    axes[0, 2].legend()

    # (d) Blockchain overhead
    axes[1, 0].plot(
        rounds,
        blockchain_send,
        marker="o",
        label="Send"
    )

    axes[1, 0].plot(
        rounds,
        blockchain_confirmation,
        marker="o",
        label="Confirmation"
    )

    axes[1, 0].set_title("(d) Blockchain Overhead")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].set_xticks(rounds)
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].legend()

    # (e) CPU usage
    x = np.arange(len(rounds))
    width = 0.35

    axes[1, 1].bar(
        x - width / 2,
        cpu_avg,
        width,
        label="Average"
    )

    axes[1, 1].bar(
        x + width / 2,
        cpu_max,
        width,
        label="Maximum"
    )

    axes[1, 1].set_title("(e) CPU Usage")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("CPU Usage (%)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(rounds)
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].legend()

    # (f) RAM usage
    axes[1, 2].bar(
        x - width / 2,
        ram_avg,
        width,
        label="Average"
    )

    axes[1, 2].bar(
        x + width / 2,
        ram_max,
        width,
        label="Maximum"
    )

    axes[1, 2].set_title("(f) RAM Usage")
    axes[1, 2].set_xlabel("Round")
    axes[1, 2].set_ylabel("RAM (MB)")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(rounds)
    axes[1, 2].grid(axis="y", alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


# dimostra la non registrazione delle transazioni qualora server alterasse il record
def plot_audit_verification_status(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    clients = data["clients"]

    rounds = [
        r["round"]
        for r in clients[0]["rounds"]
    ]

    table = []

    for client in clients:

        row = []

        for result in client["rounds"]:
            row.append(
                "✓" if result["status"] == "RECORDED" else "✗"
            )

        table.append(row)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    ax.axis("off")

    columns = [f"Round {r}" for r in rounds]
    rows = [client["client_id"] for client in clients]

    table_plot = ax.table(
        cellText=table,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center"
    )

    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 2.5)

    for (row, col), cell in table_plot.get_celld().items():

        cell.set_linewidth(0.8)
        cell.set_edgecolor("#b0b0b0")

        # Intestazioni colonne
        if row == 0 and col >= 0:

            cell.set_facecolor("white")

            cell.set_text_props(
                fontweight="bold",
                fontsize=11
            )

        # Intestazioni righe
        elif col == -1:

            cell.set_facecolor("white")

            cell.set_text_props(
                fontweight="bold",
                fontsize=11
            )

        # Celle V/X
        elif row > 0 and col >= 0:

            value = table[row - 1][col]

            cell.set_text_props(
                fontweight="bold",
                fontsize=17
            )

            if value == "✓":

                cell.set_facecolor("#e2f0d9")
                cell.get_text().set_color("#548235")

            else:

                cell.set_facecolor("#fce4d6")
                cell.get_text().set_color("#c00000")

    ax.set_title(
        "Audit Recording Status",
        fontsize=15,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    plt.show()

def plot_audit_verification_checks(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    check_labels = {
        "blockchain_audit_retrieved": "Blockchain audit retrieved",
        "ipfs_records_retrieved": "IPFS records retrieved",
        "record_hashes_computed": "Record hashes computed",
        "round_consistency": "Round consistency",
        "merkle_root_matches_blockchain": "Merkle root matches blockchain",
        "accepted_clients_count_matches_blockchain": "Accepted clients count",
        "rejected_clients_count_matches_blockchain": "Rejected clients count",
        "total_clients_count_matches_blockchain": "Total clients count"
    }

    for client in data["clients"]:

        rounds_data = client["rounds"]

        rounds = [
            r["round"]
            for r in rounds_data
        ]

        table = []

        for check in check_labels:

            row = []

            for result in rounds_data:

                if "checks" not in result:
                    row.append("—")
                else:
                    row.append(
                        "✓" if result["checks"].get(check, False)
                        else "✗"
                    )

            table.append(row)

        fig, ax = plt.subplots(figsize=(13, 5.5))

        ax.axis("off")

        columns = [f"Round {r}" for r in rounds]
        rows = list(check_labels.values())

        table_plot = ax.table(
            cellText=table,
            rowLabels=rows,
            colLabels=columns,
            cellLoc="center",
            rowLoc="center",
            loc="center"
        )

        table_plot.auto_set_font_size(False)
        table_plot.set_fontsize(10)
        table_plot.scale(1.15, 2.0)

        for (row, col), cell in table_plot.get_celld().items():

            cell.set_linewidth(0.8)
            cell.set_edgecolor("#b0b0b0")

            # Intestazioni colonne
            if row == 0 and col >= 0:

                cell.set_facecolor("white")

                cell.set_text_props(
                    fontweight="bold",
                    fontsize=11
                )

            # Intestazioni righe
            elif col == -1:

                cell.set_facecolor("white")

                cell.set_text_props(
                    fontweight="bold",
                    fontsize=10
                )

            # Celle dei controlli
            elif row > 0 and col >= 0:

                value = table[row - 1][col]

                cell.set_text_props(
                    fontweight="bold",
                    fontsize=15
                )

                if value == "✓":

                    cell.set_facecolor("#e2f0d9")
                    cell.get_text().set_color("#548235")

                elif value == "✗":

                    cell.set_facecolor("#fce4d6")
                    cell.get_text().set_color("#c00000")

                else:

                    cell.set_facecolor("#f2f2f2")
                    cell.get_text().set_color("#7f7f7f")

        ax.set_title(
            f"Audit Verification Checks - {client['client_id']}",
            fontsize=15,
            fontweight="bold",
            pad=20
        )

        plt.tight_layout()
        plt.show()


def plot_audit_client_selection(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    for client in data["clients"]:

        rounds_data = client["rounds"]

        rounds = [r["round"] for r in rounds_data]

        included_counts = []
        excluded_counts = []
        excluded_details = []

        for result in rounds_data:

            if "clients" not in result:
                included_counts.append(0)
                excluded_counts.append(0)
                continue

            included = result["clients"]["included"]
            excluded = result["clients"]["excluded"]

            included_counts.append(len(included))
            excluded_counts.append(len(excluded))

            for excluded_client in excluded:

                excluded_details.append({
                    "round": result["round"],
                    "client_id": excluded_client["client_id"],
                    "partition_id": excluded_client["partition_id"],
                    "reason": excluded_client["reason"],
                    "negative_metrics": ", ".join(
                        excluded_client["negative_metrics"]
                    )
                })

        # -------------------------------------------------
        # Nessun client escluso:
        # solo grafico
        # -------------------------------------------------

        if not excluded_details:

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            x = np.arange(len(rounds))
            width = 0.35

            ax.bar(
                x - width / 2,
                included_counts,
                width,
                label="Included"
            )

            ax.bar(
                x + width / 2,
                excluded_counts,
                width,
                label="Excluded"
            )

            ax.set_title(
                f"Audit Client Selection - {client['client_id']}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel("Round")
            ax.set_ylabel("Number of Clients")

            ax.set_xticks(x)
            ax.set_xticklabels(rounds)

            max_clients = max(
                included_counts + excluded_counts + [1]
            )

            ax.set_ylim(0, max_clients + 0.5)

            ax.set_yticks(
                range(0, max_clients + 1)
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            ax.legend()

            plt.tight_layout()
            plt.show()

            continue

        # -------------------------------------------------
        # Client esclusi presenti:
        # grafico + tabella
        # -------------------------------------------------

        fig, (ax, ax_table) = plt.subplots(
            2,
            1,
            figsize=(13, 7),
            gridspec_kw={"height_ratios": [2, 1.3]}
        )

        x = np.arange(len(rounds))
        width = 0.35

        ax.bar(
            x - width / 2,
            included_counts,
            width,
            label="Included"
        )

        ax.bar(
            x + width / 2,
            excluded_counts,
            width,
            label="Excluded"
        )

        ax.set_title(
            f"Audit Client Selection - {client['client_id']}",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel("Round")
        ax.set_ylabel("Number of Clients")

        ax.set_xticks(x)
        ax.set_xticklabels(rounds)

        max_clients = max(
            included_counts + excluded_counts + [1]
        )

        ax.set_ylim(0, max_clients + 0.5)

        ax.set_yticks(
            range(0, max_clients + 1)
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        ax.legend()

        # -------------------------------------------------
        # Tabella degli esclusi
        # -------------------------------------------------

        ax_table.axis("off")

        table_data = [
            [
                item["round"],
                item["client_id"],
                item["partition_id"],
                item["reason"],
                item["negative_metrics"]
            ]
            for item in excluded_details
        ]

        columns = [
            "Round",
            "Client ID",
            "Partition",
            "Reason",
            "Negative metrics"
        ]

        table_plot = ax_table.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc="center",
            loc="center"
        )

        table_plot.auto_set_font_size(False)
        table_plot.set_fontsize(9)
        table_plot.scale(1, 1.7)

        for (row, col), cell in table_plot.get_celld().items():

            cell.set_linewidth(0.8)
            cell.set_edgecolor("#b0b0b0")

            if row == 0:

                cell.set_facecolor("white")

                cell.set_text_props(
                    fontweight="bold",
                    fontsize=10
                )

            else:

                cell.set_text_props(
                    fontsize=9
                )

        ax_table.set_title(
            "Excluded Clients Details",
            fontsize=13,
            fontweight="bold",
            pad=10
        )

        plt.tight_layout()
        plt.show()