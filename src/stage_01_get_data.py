import argparse
import os
import logging
import tensorflow as tf
import numpy as np

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def create_directories(list_of_directories: list) -> None:
    for dir_path in list_of_directories:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory is created at {dir_path}")


def get_data(config_path) -> None:
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    create_directories(['data'])
    np.save(os.path.join(os.getcwd(), 'data', 'X_train_full'), X_train_full)
    np.save(os.path.join(os.getcwd(), 'data', 'y_train_full'), y_train_full)
    np.save(os.path.join(os.getcwd(), 'data', 'X_test'), X_test)
    np.save(os.path.join(os.getcwd(), 'data', 'y_test'), y_test)

    logging.info('Mnist data stored in data folder as arrays')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage one started <<<<<")
        get_data(config_path=parsed_args.config)
        logging.info(
            ">>>>> stage one completed! all the data are saved in local <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e