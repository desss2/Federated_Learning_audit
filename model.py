from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense


def build_model(input_size, n_classes):
    """
    Costruisce la MLP utilizzata per la classificazione
    """

    model = Sequential([
        Input(shape=(input_size,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


