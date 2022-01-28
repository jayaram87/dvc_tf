import tensorflow as tf
import logging
import io
from src.utils.common import get_timestamp
import os


def _get_model_summary(model):
    with io.StringIO() as stream:
        model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
        summary_str = stream.getvalue()
    return summary_str


def prepare_full_model(learning_rate, classes=10) -> tf.keras.models.Model:
    layers = [
        tf.keras.layers.Flatten(input_shape=(28, 28), name="inputLayer"),
        tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
        tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
        tf.keras.layers.Dense(classes, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(layers)
    model_clf.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    logging.info("custom model is compiled and ready to be trained")

    logging.info(f"full model summary {_get_model_summary(model_clf)}")
    return model_clf


def load_full_model(untrained_full_model_path: str) -> tf.keras.models.Model:
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained models is read from: {untrained_full_model_path}")
    logging.info(f"untrained full model summary: \n{_get_model_summary(model)}")
    return model


def get_unique_path_to_save_model(trained_model_dir: str, model_name: str = "model") -> str:
    timestamp = get_timestamp(name=model_name)
    unique_model_name = f"trained_{model_name}.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)

    return unique_model_path