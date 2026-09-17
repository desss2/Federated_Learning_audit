import numpy as np
from sklearn.metrics import classification_report
from visualization import *
from globals import *


def save_results(filename, results):
    np.save(filename, results, allow_pickle=True)


def load_results(filename):
    return np.load(filename, allow_pickle=True).item()


def main():

    print("\n" + "=" * 70)
    print("LOADING FEDERATED LEARNING RESULTS")
    print("=" * 70)


    if USE_LABEL_NOISE:
        result_path = "results/results_noisy.npy"
    else:
        result_path = "results/results_clean.npy"

    results = load_results(result_path)


    global_metrics = results["global_metrics"]
    all_round_reputations = results["all_round_reputations"]
    y_pred_global = results["y_pred_global"]
    y_test = results["y_test"]
    n_classes = results["n_classes"]
    label_classes = results["label_classes"]

    # ===========================================================
    # GENERAL INFORMATION
    # ===========================================================

    print(f"\nNumber of rounds: {len(global_metrics)}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Reputation threshold: {REPUTATION_THRESHOLD}")

    # ===========================================================
    # GLOBAL METRICS PER ROUND
    # ===========================================================

    print("\n" + "=" * 70)
    print("GLOBAL MODEL PERFORMANCE PER ROUND")
    print("=" * 70)

    print( f"{'Round':<10} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")

    print("-" * 70)

    for r, metrics in enumerate(global_metrics, start=1):

        acc, prec, rec, f1 = metrics

        print(f"{r:<10} {acc:<15.4f} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f}")

    # ===========================================================
    # FINAL GLOBAL PERFORMANCE
    # ===========================================================

    final_acc, final_prec, final_rec, final_f1 = global_metrics[-1]

    print("\n" + "=" * 70)
    print("FINAL GLOBAL PERFORMANCE")
    print("=" * 70)

    print(f"Accuracy : {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall   : {final_rec:.4f}")
    print(f"F1-Score : {final_f1:.4f}")

    # ===========================================================
    # REPUTATION PER ROUND
    # ===========================================================

    print("\n" + "=" * 70)
    print("REPUTATION SCORES PER ROUND")
    print("=" * 70)

    for r, round_reputations in enumerate(all_round_reputations,start=1):

        print(f"\n--- Round {r} ---")

        for ci, rep in enumerate(round_reputations):

            print(f"\nClient {ci}:")
            print(f"  Combined          : {rep['combined']:.4f}")
            print(f"  Weighted Avg      : {rep['weighted_avg']:.4f}")
            print(f"  Beta Reputation   : {rep['beta_reputation']:.4f}")
            print(f"    Fuzzy Trust: {rep['fuzzy_trust']:.4f}")
            print(f"    Tanh Utility: {rep['tanh_utility']:.4f}")
            print(f"    Exp Decay: {rep['exponential_decay']:.4f}")
            print(f"    Entropy: {rep['entropy_based']:.4f}")
            print(f"    Cosine Sim: {rep['cosine_similarity']:.4f}")
            print(f"    Consistency: {rep['consistency']:.4f}")
            print(f"    Plausibility: {rep['plausibility']:.4f}")

            status = ("INCLUDED" if rep["combined"] >= REPUTATION_THRESHOLD
                else "EXCLUDED")

            print(f"  Status            : {status}")



    # ===========================================================
    # FINAL REPUTATION
    # ===========================================================

    print("\n" + "=" * 70)
    print("FINAL ROUND REPUTATION")
    print("=" * 70)

    final_reputations = all_round_reputations[-1]

    for ci, rep in enumerate(final_reputations):

        status = ("INCLUDED" if rep["combined"] >= REPUTATION_THRESHOLD
            else "EXCLUDED")

        print(
            f"Client {ci}: reputation={rep['combined']:.4f} -> {status}")

    # ===========================================================
    # VISUALIZATIONS
    # ===========================================================

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Global metrics
    plot_global_metrics(
        global_metrics
    )

    # 2. Reputation evolution
    plot_reputation_evolution(
        all_round_reputations,
        NUM_CLIENTS,
        REPUTATION_THRESHOLD
    )

    # 3. Reputation methods
    plot_reputation_methods(
        all_round_reputations,
        NUM_CLIENTS
    )

    # 4. Confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred_global,
        n_classes,
        label_classes
    )

    if USE_LABEL_NOISE:
        performance_label = "noisy"
    else:
        performance_label = "clean"

    client_filenames = [
        f"results/client_{client_id}_performance_metrics_{performance_label}.json"
        for client_id in range(NUM_CLIENTS)
    ]

    server_filename = f"results/performance_metrics_{performance_label}.json"

    plot_client_performance(client_filenames)
    plot_gas_usage(client_filenames, server_filename)
    plot_server_performance(server_filename)

    audit_filename=f"results/audit_verification_{performance_label}.json"

    plot_audit_verification_status(audit_filename)
    plot_audit_verification_checks(audit_filename)

    plot_audit_client_selection(audit_filename)



    # ===========================================================
    # CLASSIFICATION REPORT
    # ===========================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            y_pred_global,
            labels=np.arange(n_classes),
            target_names=label_classes,
            digits=4
        )
    )


if __name__ == "__main__":
    main()
