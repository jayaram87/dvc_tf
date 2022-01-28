import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml_file, create_directories
from src.utils.model import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def train_model(config_path: str, params_path: str) -> None:
    config = read_yaml_file(config_path)
    params = read_yaml_file(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["MODEL_DIR"], artifacts["MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    x_valid, x_train, y_valid, y_train = train_valid_generator(artifacts['DATA_DIR'])

    model.fit(
        x_train, y_train,
        epochs=params["EPOCHS"],
        batch_size = params['BATCH_SIZE'],
        validation_data=(x_valid, y_valid),
        callbacks=callbacks)

    ### save the trained model
    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([trained_model_dir])

    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)

    logging.info(f"trained models is saved at: \n{model_file_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage four started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage four completed! training completed <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e