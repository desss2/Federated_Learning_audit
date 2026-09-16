import numpy as np
from flwr.client import ClientApp, NumPyClient

from audit import hash_model_weights
from blockchain import *
from data import *
from model import *
from evaluation import *
from globals import *
import joblib
import gc
import threading
import psutil
import time
import json

DATA_DIR = os.getenv("DATA_DIR", "./data")

setup_info = joblib.load(os.path.join(DATA_DIR, "setup_info.pkl"))

input_size = setup_info["input_size"]
n_classes = setup_info["n_classes"]
class_weights = setup_info["class_weights"]


def monitor_resources(process, samples, stop_event, interval=0.2):

    while not stop_event.is_set():

        cpu = process.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024 ** 2)

        samples["cpu"].append(cpu)
        samples["ram"].append(ram)

        time.sleep(interval)


class FlowerClient(NumPyClient):

    def __init__(
            self,
            partition_id,
            X_train,
            y_train,
    ):
        self.partition_id = partition_id

        self.X_train = X_train
        self.y_train = y_train

        self.model = build_model(
            input_size=input_size,
            n_classes=n_classes
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def save_performance_metrics(self, round_metrics):

        if USE_LABEL_NOISE:
            filename = (
                f"client_{self.partition_id}_performance_metrics_noisy.json"
            )
        else:
            filename = (
                f"client_{self.partition_id}_performance_metrics_clean.json"
            )

        filepath = os.path.join("/app/results", filename)

        os.makedirs("/app/results", exist_ok=True)

        # Legge i risultati già presenti
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {
                "rounds": []
            }

        # Aggiunge il nuovo round
        data["rounds"].append(round_metrics)

        # Riscrive il file aggiornato
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4
            )

        print(f"[PERFORMANCE] Metrics saved to {filename}")

    def train_local(self, round_num):

        # split training/validation locale
        mask = np.random.RandomState(1000 + round_num + self.partition_id).rand(len(self.y_train)) < 0.9

        train_indices = np.flatnonzero(mask)
        val_indices = np.flatnonzero(~mask)

        sampled_train_indices = sampling_indices(
            self.y_train,
            train_indices,
            LOCAL_TRAIN_SIZE,
            seed=10000 + round_num * 100 + self.partition_id
        )

        Xc_tr = np.asarray(self.X_train[sampled_train_indices])
        yc_tr = np.asarray(self.y_train[sampled_train_indices])

        Xc_va = np.asarray(self.X_train[val_indices])

        yc_va = np.asarray(self.y_train[val_indices])

        n_train = len(yc_tr)
        
        training_start = time.perf_counter()

        self.model.fit(
            Xc_tr,
            yc_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            class_weight=class_weights
        )
        
        training_time = time.perf_counter() - training_start

        del Xc_tr
        del yc_tr
        del train_indices
        del sampled_train_indices
        gc.collect()

        return Xc_va, yc_va, n_train, training_time


    def evaluate_local(self, Xc_va, yc_va):

        if len(yc_va) > 0:
            p_val = self.model.predict(Xc_va, verbose=0)
            yva_pred = np.argmax(p_val, axis=1)
            result = evaluate(yc_va, yva_pred)

            del p_val
            del yva_pred
            gc.collect()

            return result
        return 0.0, 0.0, 0.0, 0.0


    # training locale
    def fit(self, parameters, config):

        round_num = config.get("round", -1)

        print(f"[CLIENT partition={self.partition_id}] Started Training Round {round_num}")
        
        process = psutil.Process(os.getpid())

        resource_samples = {
        	"cpu": [],
        	"ram": []
    	}

        stop_monitor = threading.Event()

        monitor_thread = threading.Thread(
        	target=monitor_resources,
        	args=(process, resource_samples, stop_monitor),
        	daemon=True
    	)

        monitor_thread.start()


        self.set_parameters(parameters)
        
        # training
        Xc_va, yc_va, n_train, training_time = self.train_local(round_num)
        
        
        # validazione locale
        cacc, cprec, crec, cf1= self.evaluate_local(Xc_va, yc_va)

        del Xc_va
        del yc_va
        gc.collect()

        print(f"  [CLIENT partition={self.partition_id}] [ROUND {round_num}] Local Val: Acc={cacc:.4f}, F1={cf1:.4f}")

        # parametri aggiornati dopo il training locale
        updated_weights = self.get_parameters(config)

        # hash dei pesi aggiornati
        update_hash = hash_model_weights(updated_weights)

        # dati della ClientUpdate
        client_update = {
            "client_id": f"client_{self.partition_id}",
            "round": round_num,
            "update_hash": update_hash,
        }

        node_name = f"client_{self.partition_id}"

        blockchain_rpc_url = get_blockchain_rpc_url(node_name)
        private_key_path = get_blockchain_private_key_path(node_name)
        private_key = Path(private_key_path).read_text().strip()
        tx_hash, block_number, status, gas_used, send_time, confirmation_time = register_client_update_on_blockchain(client_update, blockchain_rpc_url, private_key)
        if status ==1:
            print(f"[AUDIT] Blockchain audit transaction  sent: {tx_hash}. Recorded in block: {block_number}")
        else:
            print("[AUDIT] Blockchain audit transaction failed")
            
        stop_monitor.set()
        monitor_thread.join()

        cpu_avg = np.mean(resource_samples["cpu"])
        cpu_max = np.max(resource_samples["cpu"])

        ram_avg = np.mean(resource_samples["ram"])
        ram_max = np.max(resource_samples["ram"])

        round_metrics = {
            "round": round_num,
            "training_time": training_time,
            "blockchain_send_time": send_time,
            "blockchain_confirmation_time": confirmation_time,
            "gas_used": gas_used,
            "cpu_avg_percent": cpu_avg,
            "cpu_max_percent": cpu_max,
            "ram_avg_mb": ram_avg,
            "ram_max_mb": ram_max
        }

        self.save_performance_metrics(round_metrics)

        return (
            self.get_parameters(config),
            n_train,
            {
                "local_accuracy": cacc,
                "local_f1": cf1,
                "partition_id": self.partition_id,
            }
        )

def client_fn(context):

    partition_id = int(
        context.node_config["partition-id"]
    )

    if USE_LABEL_NOISE:
        dataset_type = "noisy"
    else:
        dataset_type = "clean"

    X_train = np.load(os.path.join(DATA_DIR,f"client_{partition_id}_{dataset_type}_X.npy"),mmap_mode="r")

    y_train = np.load(os.path.join(DATA_DIR,f"client_{partition_id}_{dataset_type}_y.npy"),mmap_mode="r")

    return FlowerClient(
        partition_id=partition_id,
        X_train=X_train,
        y_train=y_train,
    ).to_client()

app = ClientApp(
    client_fn=client_fn
)



