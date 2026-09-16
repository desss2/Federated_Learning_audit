import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import os
from data import load_and_preprocess_data, create_stratified_partitions
from sklearn.utils.class_weight import compute_class_weight
from model import build_model
from globals import *

from pathlib import Path


DOCKER_COMPOSE_FILE = Path("docker-compose.yml")


from pathlib import Path


def create_docker_compose(num_clients):

    services = {}

    services["superlink"] = {
        "image": "flwr/superlink:${FLWR_VERSION:-1.30.0}",
        "command": [
            "--insecure",
            "--isolation",
            "process",
        ],
        "ports": [
            "9093:9093",
        ],
    }


    services["superexec-serverapp"] = {
        "build": {
            "context": "${PROJECT_DIR:-.}",
            "dockerfile_inline": """\
            FROM flwr/superexec:${FLWR_VERSION:-1.30.0}
            USER root
            RUN apt-get update \\
                && apt-get -y --no-install-recommends install \\
                build-essential \\
                && rm -rf /var/lib/apt/lists/*
            USER app
            WORKDIR /app
            COPY --chown=app:app pyproject.toml .
            RUN sed -i 's/.*flwr\\[simulation\\].*//' pyproject.toml \\
                && python -m pip install -U --no-cache-dir .   
            ENTRYPOINT ["flower-superexec"]
            """,
        },
        "command": [
            "--insecure",
            "--plugin-type",
            "serverapp",
            "--appio-api-address",
            "superlink:9091",
        ],
        "environment": {
            "DATA_DIR": "/app/data",
        },
        "volumes": [
            "./data/X_test.npy:/app/data/X_test.npy:ro",
            "./data/y_test.npy:/app/data/y_test.npy:ro",
            "./data/setup_info.pkl:/app/data/setup_info.pkl:ro",
            "./identity_mapping.json:/app/identity_mapping.json",
            "./blockchain/network/nodes/server:/app/blockchain/network/nodes/server:ro",
            "./blockchain/contract_address.json:/app/blockchain/contract_address.json:ro",
            "./results:/app/results",
        ],
        "restart": "on-failure:3",
        "depends_on": [
            "superlink",
        ],
        "networks": [
            "default",
            "securecare-ipfs",
            "securecare-blockchain",
        ],
    }


    for client_id in range(num_clients):

        node_number = client_id + 1

        if USE_LABEL_NOISE:
            data_x_file = f"client_{client_id}_noisy_X.npy"
            data_y_file = f"client_{client_id}_noisy_y.npy"
        else:
            data_x_file = f"client_{client_id}_clean_X.npy"
            data_y_file = f"client_{client_id}_clean_y.npy"

        supernode_name = f"supernode-{node_number}"
        clientapp_port = 9094 + client_id
        services[supernode_name] = {
            "image": "flwr/supernode:${FLWR_VERSION:-1.30.0}",
            "command": [
                "--insecure",
                "--superlink",
                "superlink:9092",
                "--clientappio-api-address",
                f"0.0.0.0:{clientapp_port}",
                "--isolation",
                "process",
                "--node-config",
                f"partition-id={client_id} num-partitions={num_clients}",
            ],
            "depends_on": [
                "superlink",
            ],
        }

        clientapp_name = f"superexec-clientapp-{node_number}"

        services[clientapp_name] = {
            "build": {
                "context": "${PROJECT_DIR:-.}",
                "dockerfile_inline": """\
                FROM flwr/superexec:${FLWR_VERSION:-1.30.0}
                
                USER root
                
                RUN apt-get update \\
                    && apt-get -y --no-install-recommends install \\
                    build-essential \\
                    && rm -rf /var/lib/apt/lists/*
                
                USER app
                
                WORKDIR /app
                
                COPY --chown=app:app pyproject.toml .
                
                RUN sed -i 's/.*flwr\\[simulation\\].*//' pyproject.toml \\
                    && python -m pip install -U --no-cache-dir .
                
                ENTRYPOINT ["flower-superexec"]
                """,
            },
            "command": [
                "--insecure",
                "--plugin-type",
                "clientapp",
                "--appio-api-address",
                f"{supernode_name}:{clientapp_port}",
            ],
            "environment": {
                "DATA_DIR": "/app/data",
            },
            "volumes": [
                "./data/setup_info.pkl:/app/data/setup_info.pkl:ro",
                f"./data/{data_x_file}:/app/data/{data_x_file}:ro",
                f"./data/{data_y_file}:/app/data/{data_y_file}:ro",
                "./verify_client.py:/app/verify_client.py:ro",
                "./globals.py:/app/globals.py:ro",
                "./audit.py:/app/audit.py:ro",
                f"./blockchain/network/nodes/client_{client_id}:/app/blockchain/network/nodes/client_{client_id}:ro",
                "./blockchain/contract_address.json:/app/blockchain/contract_address.json:ro",
                "./results:/app/results",
            ],
            "deploy": {
                "resources": {
                    "limits": {
                        "cpus": "2",
                    }
                }
            },
            "stop_signal": "SIGINT",
            "depends_on": [
                supernode_name,
            ],
            "networks": [
                "default",
                "securecare-ipfs",
                "securecare-blockchain",
            ],
        }

    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required to generate docker-compose.yml")

    compose = {
        "services": services,
        "networks": {
            "securecare-ipfs": {
                "external": True,
            },
            "securecare-blockchain": {
                "external": True,
            },
        },
    }

    output_path = Path("docker-compose.yml")

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            compose,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"Docker Compose generated: {output_path}")


# caricamento e preprocessing del dataset
X_train, X_test, y_train, y_test, le, scaler, n_classes = \
    load_and_preprocess_data(False)

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {
    int(cls): float(np.sqrt(weight))
    for cls, weight in zip(classes, class_weights)
}

print("\n[SETUP] Class weights:")
for cls, weight in class_weight_dict.items():
    print(f"    Class {cls}: {weight:.4f}")


print("[SETUP] Dataset loaded and preprocessed successfully.")

input_size = X_train.shape[1]

baseline_model = build_model(
    input_size=input_size,
    n_classes=n_classes
)

print("[SETUP] Baseline MLP built successfully.")

print("\n[SETUP] Preparing baseline training...")

# modello baseline
baseline_sample_idx = np.random.choice(
    len(X_train),
    size=min(LOCAL_TRAIN_SIZE, len(X_train)),
    replace=False
)

baseline_model.fit(
    X_train[baseline_sample_idx],
    y_train[baseline_sample_idx],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    class_weight=class_weight_dict
)

print("[SETUP] Baseline training completed.")


baseline_pred = baseline_model.predict(X_test, verbose=0)

baseline_performance = accuracy_score(
    y_test,
    np.argmax(baseline_pred, axis=1)
)

os.makedirs("./data", exist_ok=True)

setup_info = {
    "baseline_performance": baseline_performance,
    "input_size":input_size,
    "le": le,
    "scaler": scaler,
    "n_classes": n_classes,
    "class_weights": class_weight_dict
}

joblib.dump(
    setup_info,
    "./data/setup_info.pkl"
)

print("\n[SETUP] Creating stratified client partitions...")

# creazione partizioni dei client
create_stratified_partitions(X_train,y_train,NUM_CLIENTS, noise_config=NOISE_CONFIG)

np.save( "./data/X_test.npy", X_test)
np.save("./data/y_test.npy",y_test)

create_docker_compose(NUM_CLIENTS)
