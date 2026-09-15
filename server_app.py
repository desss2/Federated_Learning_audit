import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from blockchain import *
from data import *
from evaluation import *
from identity import update_identity_mapping
from reputation import ReputationManager
from result import *
from model import build_model
from audit import *
import joblib
import hashlib
import json
import time

DATA_DIR = os.getenv("DATA_DIR", "./data")

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
setup_info = joblib.load(os.path.join(DATA_DIR, "setup_info.pkl"))

baseline_performance = setup_info["baseline_performance"]
input_size=setup_info["input_size"]
le = setup_info["le"]
scaler = setup_info["scaler"]
n_classes = setup_info["n_classes"]

# costruiamo modello MLP
model = build_model(
    input_size=input_size,
    n_classes=n_classes
)

# prendiamo i parametri iniziali del modello
initial_parameters = fl.common.ndarrays_to_parameters(
    model.get_weights()
)


def evaluate_config(server_round: int):
    return {
        "round": server_round
    }

def fit_config(server_round: int):
    return {
        "round": server_round,
    }


class OutputFedAvg(FedAvg):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        # Reputation manager
        self.reputation_manager = ReputationManager(NUM_CLIENTS)

        self.reputation_manager.baseline_performance = baseline_performance

        # Soglia di reputazione
        self.reputation_threshold = REPUTATION_THRESHOLD

        # Salvataggio risultati
        self.global_metrics = []
        self.all_round_reputations = []
        self.y_pred_global = None

        # pesi prima dell'aggregazione
        self.global_weights = [w.copy() for w in model.get_weights()]


    def collect_client_results(self, results):

        # ordiniamo in base a partition_id
        results = sorted(
            results,
            key=lambda x: int(x[1].metrics.get("partition_id", -1))
        )

        partition_ids = [
            int(fit_res.metrics.get("partition_id", -1))
            for _, fit_res in results
        ]

        if partition_ids != list(range(len(partition_ids))):
            raise ValueError(
                f"Partition IDs non validi o non contigui: {partition_ids}"
            )

        client_ids = []
        client_weights = []
        client_num_examples = []
        round_metrics = []

        for client_proxy, fit_res in results:

            client_ids.append(int(client_proxy.cid))
            local_weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(local_weights)
            client_num_examples.append(fit_res.num_examples)
            metrics = fit_res.metrics
            partition_id = int(metrics.get("partition_id", -1))

            round_metrics.append({
                "client_id": int(client_proxy.cid),
                "partition_id": partition_id,
                "local_accuracy": metrics.get("local_accuracy", 0.0),
                "local_f1": metrics.get("local_f1", 0.0),
            })

        return (
            client_ids,
            client_weights,
            client_num_examples,
            round_metrics,
        )

    def calculate_client_updates(self, client_weights):

        client_updates = []

        for local_weights in client_weights:
            update = [
                local_w - global_w
                for local_w, global_w in zip(
                    local_weights,
                    self.global_weights
                )
            ]

            client_updates.append(update)

        return client_updates

    def evaluate_global_model(
            self,
            aggregated_weights,
            server_round,
    ):

        global_model = build_model(
            input_size=input_size,
            n_classes=n_classes
        )

        global_model.set_weights(aggregated_weights)

        p_global = global_model.predict(
            X_test,
            verbose=0
        )

        y_pred_global = np.argmax(p_global,axis=1)

        self.y_pred_global = y_pred_global

        acc, prec, rec, f1 = evaluate(y_test,y_pred_global)
        self.global_metrics.append((acc, prec, rec, f1))

        print(f"\n[SERVER] Round {server_round} "f"Global Performance:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

        return acc, prec, rec, f1

    def save_final_results(self):

        if USE_LABEL_NOISE:
            filename = "results_noisy.npy"
        else:
            filename = "results_clean.npy"

        results = {
            "global_metrics": self.global_metrics,
            "all_round_reputations": self.all_round_reputations,
            "final_reputation_scores": self.all_round_reputations[-1],
            "y_pred_global": self.y_pred_global,
            "y_test": y_test,
            "n_classes": n_classes,
            "label_classes": le.classes_,
        }

        filepath = os.path.join("/app/results", filename)

        save_results(filepath, results)

        print(f"\n[SERVER] Results saved to {filename}")

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):


        if not results:
            return None, {}

        round_reputations = []
        round_reputation_analysis = []
        client_predictions = []
        client_probabilities = []

        # leggiamo i dati restituiti dal client
        client_ids, client_weights, client_num_examples, round_metrics = self.collect_client_results(results)


        # valutazione dei modelli dei client sul global test
        for i, weights in enumerate(client_weights):
            self.model.set_weights(weights)

            p_test = self.model.predict(self.X_test, verbose=0)
            y_pred_test = np.argmax(p_test, axis=1)

            test_acc, test_prec, test_rec, test_f1 = evaluate(
                self.y_test,
                y_pred_test
            )

            client_predictions.append(y_pred_test)
            client_probabilities.append(p_test)


            round_metrics[i]["accuracy"] = test_acc
            round_metrics[i]["precision"] = test_prec
            round_metrics[i]["recall"] = test_rec
            round_metrics[i]["f1"] = test_f1


        # registriamo il mapping delle identità dei client
        for client in round_metrics:
            update_identity_mapping(client_id=client["client_id"],partition_id=client["partition_id"])

        # media delle accuracy dei client nel round corrente
        global_avg_accuracy = np.mean([
            m["accuracy"]
            for m in round_metrics
        ])

        # otteniamo gli update, come differenza tra ciò che il server ha mandato e ciò che il client ha calcolato
        client_updates=self.calculate_client_updates(client_weights)

        num_clients = len(client_weights)

        # calcolo reputazione
        for i, client_id in enumerate(client_ids):
            rep_scores, rep_analysis = (
                self.reputation_manager
                .update_client_reputation(
                    client_id=client_id,    # id del client
                    metrics=round_metrics[i],
                    client_index=i,         # indice utilizzato per accedere a client_updates, cioè la posizione relativa a client_id in client_updates,
                    client_updates=client_updates,
                    global_avg_accuracy=global_avg_accuracy,
                    y_pred_test=client_predictions[i],
                    p_test=client_probabilities[i],
                    current_round=server_round
                )
            )

            round_reputations.append(rep_scores)
            round_reputation_analysis.append(rep_analysis)

            print(f"[SERVER] Client {client_id} "f"reputation = "f"{rep_scores['combined']:.4f}")

        self.all_round_reputations.append(round_reputations)


        # filtering client sulla base della reputazione
        included_clients = [
            i for i in range(num_clients)
            if self.reputation_manager.should_include_client(round_reputations[i],self.reputation_threshold)
        ]

        if len(included_clients) == 0:
            print("\n [SERVER] WARNING: No clients meet reputation threshold!")
            included_clients = list(range(num_clients))

        print(f"\n  [SERVER] Round {server_round} Included Clients: {[client_ids[i] for i in included_clients]}")

        # audit del round FL
        round_audit = []
        client_audits = []
        audit_hashes = []

        for i, client_id in enumerate(client_ids):

            if i in included_clients:
                decision = "ACCEPTED"
                aggregation_decision = True
                reason = {
                    "motivation": "reputation_above_threshold"
                }
            else:
                decision = "REJECTED"
                aggregation_decision = False
                negative_metrics = round_reputation_analysis[i]["negative_metrics"]

                reason = {
                    "motivation": "reputation_below_threshold",
                    "negative_metrics": [
                        {
                            "metric": metric["metric"],
                            "score": metric["value"],
                            "weight": metric["weight"],
                            "penalty": metric["penalty"]
                        }
                        for metric in negative_metrics
                    ]
                }

            local_model_hash = hash_model_weights(
                client_weights[i]
            )

            partition_id = round_metrics[i]["partition_id"]

            client_id_str = f"client_{partition_id}"

            client_update_hash, client_update_timestamp = get_client_update_from_blockchain(
                    client_id=client_id_str,
                    round_num=server_round,
                    blockchain_rpc_url=get_blockchain_rpc_url(client_id_str)
                )


            record = create_client_audit_record(
                client_id=client_id,
                partition_id=partition_id,
                server_round=server_round,
                metrics=round_metrics[i],
                reputation=round_reputations[i],
                decision=decision,
                reason=reason,
                local_model_hash=local_model_hash
            )

            round_audit.append(record)

            # hash dell'audit del singolo record del client, per la costruzione del Merkle Tree
            record_hash = hash_audit_record(record)
            audit_hashes.append(record_hash)

            client_audits.append({
                "clientId": client_id_str,
                "hash": local_model_hash,
                "reputation": round_reputations[i]["combined"],
                "aggregationDecision": aggregation_decision
            })

        ipfs_api_url = get_ipfs_api_url("server")
        cid_transactions = upload_audit_records_to_ipfs(round_audit, ipfs_api_url)


        # a partire dalla lista degli hash, creiamo il Merkle Tree
        merkle_root = build_merkle_tree(audit_hashes)

        # pesi basati sulla reputazione
        rep_weights = self.reputation_manager.get_aggregation_weights(
            [round_reputations[i] for i in included_clients]
        )

        # aggregazione finale
        aggregated_weights = []
        for layer_idx in range(len(client_weights[0])):

            layer = np.zeros_like(client_weights[0][layer_idx])

            for i, client_idx in enumerate(included_clients):
                layer += (rep_weights[i]* client_weights[client_idx][layer_idx])

            aggregated_weights.append(layer)

        # aggiorniamo il modello globale:
        self.global_weights = [w.copy() for w in aggregated_weights]


        n_clients_accepted = len(included_clients)
        n_clients_rejected = len(client_ids) - n_clients_accepted

        global_model_hash = hash_model_weights(self.global_weights)

        round_transaction = create_round_transaction(
            server_round=server_round,
            n_clients_accepted=n_clients_accepted,
            n_clients_rejected=n_clients_rejected,
            global_model_hash=global_model_hash,
            merkle_root=merkle_root,
            cid_transactions=cid_transactions
        )
        print(f"[AUDIT] Round transaction:\n {json.dumps(round_transaction, indent=4)}")

        # transazione di audit del round
        blockchain_rpc_url = get_blockchain_rpc_url("server")
        path_private_key=get_blockchain_private_key_path("server")
        private_key = Path(path_private_key).read_text().strip()
        tx_hash, block_number, status = record_audit_on_blockchain(round_transaction, client_audits, blockchain_rpc_url, private_key)

        if status ==1:
            print(f"[AUDIT] Blockchain audit transaction  sent: {tx_hash}. Recorded in block: {block_number}")
        else:
            print("[AUDIT] Blockchain audit transaction failed")

        # definiamo i nuovi parametri da mandare:
        new_parameters=fl.common.ndarrays_to_parameters(aggregated_weights)

        # valutazione modello aggregato sul dataset di test

        acc, prec, rec, f1 = (
            self.evaluate_global_model(
                aggregated_weights,
                server_round
            )
        )

        # Alla fine dei round di FL salviamo i risultati
        if server_round == ROUNDS:
            self.save_final_results()


        return new_parameters, {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }


strategy = OutputFedAvg(
    on_evaluate_config_fn=evaluate_config,
    fraction_fit=1.0,
    on_fit_config_fn=fit_config,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=initial_parameters,
)

def server_fn(context):

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(
            num_rounds=ROUNDS
        )
    )


app = ServerApp(
    server_fn=server_fn
)



