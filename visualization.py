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