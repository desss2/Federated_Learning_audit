import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from globals import *



def get_label_from_filename(filename):

    '''
    estrae l'etichetta da porre al record a partire dal nome del file.
    ci sono 3 possibili scenari:
    • classificazione binaria (attacco/benigno)
    • classificazione multiclasse (categoria di attacco/ benigno)
    • classificazione multiclasse (tipologia (sotto-categoria) di attacco/benigno)
    lo scenario è definito dalla variabile LABEL_MODE in globals.py
    '''

    basename = os.path.basename(filename)

    # Rimuoviamo il suffisso relativo al train e test dal nome dei file
    if basename.endswith("_train.pcap.csv"):
        attack_name = basename[:-len("_train.pcap.csv")]
    elif basename.endswith("_test.pcap.csv"):
        attack_name = basename[:-len("_test.pcap.csv")]
    else:
        raise ValueError(f"Unexpected filename format: {basename}")

    # ritorniamo l'etichetta in modo coerente a quanto dichiarato con LABEL_MODE
    if LABEL_MODE == "binary":
        if attack_name == "Benign":
            return "Benign"
        return "Attack"

    elif LABEL_MODE == "category":
        if attack_name == "Benign":
            return "Benign"
        elif "DDoS" in attack_name:
            return "DDoS"
        elif "DoS" in attack_name:
            return "DoS"
        elif attack_name.startswith("Recon-"):
            return "Recon"
        elif "Spoofing" in attack_name:
            return "Spoofing"
        elif attack_name.startswith("MQTT-"):
            return "MQTT"
        else:
            raise ValueError(
                f"Cannot determine category for file: {basename}"
            )

    elif LABEL_MODE == "attack_type":
        # rimuoviamo eventuali numeri finali
        return attack_name.rstrip("0123456789")

    else:
        raise ValueError(
            f"Unknown LABEL_MODE: {LABEL_MODE}"
        )


def load_dataset_from_directory(directory):
    """
    Carica tutti i file csv dalla cartella, ci aggiunge
    l'etichetta basandosi sul nome di essi e genera un unico dataset
    combinando tutti i csv

    """

    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in directory: {directory}"
        )

    dataframes = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        label = get_label_from_filename(file_path)
        df["Label"] = label

        dataframes.append(df)

    dataset = pd.concat(
        dataframes,
        ignore_index=True
    )


    return dataset


def load_data(debug_log):
    '''
    Carica tutti i dataset di training e di test di CICIoMT2024
    '''

    if debug_log:
        print("Loading CICIoMT2024 dataset...")

    # crea un unico dataset di training con le etichette
    train_df = load_dataset_from_directory(TRAIN_PATH)
    # crea un unico dataset di test con le etichette
    test_df = load_dataset_from_directory(TEST_PATH)

    # eliminiamo eventuali righe duplicate nei dataset finali
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    if debug_log:
        print("\nDataset loaded successfully.")

        print("\nTrain classes:")
        print(train_df["Label"].value_counts())

        print("\nTest classes:")
        print(test_df["Label"].value_counts())

    return train_df, test_df


def preprocess_data(train_df, test_df, debug_log):
    """
    Preprocessing dei dataset di training e di test.
    """
    if debug_log:
        print("\nPreprocessing dataset...")

    # Separazione feature / label
    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"]

    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"]

    # Controllo che le feature siano numeriche
    non_numeric_train = X_train.select_dtypes(
        exclude=np.number
    ).columns

    non_numeric_test = X_test.select_dtypes(
        exclude=np.number
    ).columns

    if len(non_numeric_train) > 0:
        raise ValueError(
            f"Non-numeric features in train: "
            f"{list(non_numeric_train)}"
        )

    if len(non_numeric_test) > 0:
        raise ValueError(
            f"Non-numeric features in test: "
            f"{list(non_numeric_test)}"
        )

    # Encoding delle label
    label_encoder = LabelEncoder()

    label_encoder.fit(y_train)

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Standardizzazione
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Conversione ai tipi utilizzati da TensorFlow
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Numero di classi
    n_classes = len(label_encoder.classes_)
    if debug_log:
        print("\nPreprocessing completed.")

        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {n_classes}")
        print(f"Classes: {label_encoder.classes_}")

        print(f"Train samples: {X_train.shape[0]}")
        print(f"Test samples:  {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, label_encoder, scaler, n_classes


def load_and_preprocess_data(debug_log=True):
    '''
    funzione finale per il caricamento del dataset e il suo preprocessing
    '''

    train_df, test_df = load_data(debug_log)

    return preprocess_data(
        train_df,
        test_df,
        debug_log
    )

def sampling_dataset(X_train, y_train, sample_size, seed=None):

    rng = np.random.default_rng(seed)

    tune_indices = []

    for class_id in np.unique(y_train):
        class_indices = np.where(y_train == class_id)[0]

        n_samples = max(
            1,
            round(sample_size * len(class_indices) / len(y_train))
        )

        selected = rng.choice(
            class_indices,
            size=n_samples,
            replace=False
        )

        tune_indices.extend(selected)

    tune_indices = np.array(tune_indices)

    # Nel caso il numero finale sia leggermente diverso da TUNE_SIZE
    rng.shuffle(tune_indices)
    tune_indices = tune_indices[:sample_size]

    X_small = X_train[tune_indices]
    y_small = y_train[tune_indices]


    return X_small, y_small


def sampling_indices(y, candidate_indices, sample_size, seed=None):

    rng = np.random.default_rng(seed)

    candidate_y = y[candidate_indices]

    selected_indices = []

    for class_id in np.unique(candidate_y):

        class_indices = candidate_indices[candidate_y == class_id]

        n_samples = max(
            1,
            round(
                sample_size* len(class_indices)/ len(candidate_indices)
            )
        )

        selected = rng.choice(
            class_indices,
            size=n_samples,
            replace=False
        )

        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)

    rng.shuffle(selected_indices)

    return selected_indices[:sample_size]

def inject_label_noise(y, noise_rate, n_classes, seed):
    """
    Introduce label noise modificando casualmente una percentuale
    delle etichette.
    """

    rng = np.random.default_rng(seed)

    y_noisy = y.copy()

    n_flip = int(len(y) * noise_rate)

    if n_flip == 0:
        return y_noisy

    flip_idx = rng.choice(
        len(y),
        size=n_flip,
        replace=False
    )

    for idx in flip_idx:

        original_label = y_noisy[idx]

        wrong_classes = [
            c for c in range(n_classes)
            if c != original_label
        ]

        y_noisy[idx] = rng.choice(wrong_classes)

    return y_noisy


def create_stratified_partitions(X_train, y_train,num_clients,noise_config=None):
    '''
    Crea le partizioni stratificate dei client.

    Per ogni client vengono salvate:
    - la partizione originale (clean)
    - la stessa partizione con label noise (noisy)
    '''

    if noise_config is None:
        noise_config = {
            client_id: 0.0
            for client_id in range(num_clients)
        }

    skf = StratifiedKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    n_classes = len(np.unique(y_train))


    for client_id, (_, idx) in enumerate(skf.split(X_train, y_train)):


        # partizione originale
        X_client = X_train[idx]
        y_clean = y_train[idx].copy()

        np.save(
            f"./data/client_{client_id}_clean_X.npy",
            X_client
        )

        np.save(
            f"./data/client_{client_id}_clean_y.npy",
            y_clean
        )

        # partizione con label noise
        noise_rate = noise_config.get(client_id, 0.0)



        y_noisy = inject_label_noise(
            y_clean,
            noise_rate=noise_rate,
            n_classes=n_classes,
            seed=RANDOM_SEED + client_id
        )

        np.save(
            f"./data/client_{client_id}_noisy_X.npy",
            X_client
        )

        np.save(
            f"./data/client_{client_id}_noisy_y.npy",
            y_noisy
        )

