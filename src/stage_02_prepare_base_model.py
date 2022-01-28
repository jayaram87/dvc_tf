import argparse
import os
import logging
from tqdm import tqdm
from src.utils.model import prepare_full_model
from src.utils.common import read_yaml_file, create_directories

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def prepare_base_model(config_path: str, params_path: str) -> None:
    config = read_yaml_file(config_path)
    params = read_yaml_file(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])

    model_path = os.path.join(model_dir_path, model_name)

    model = prepare_full_model(learning_rate=params["LEARNING_RATE"], classes=10)

    model.save(model_path)
    logging.info(f"full untrained model is saved at {model_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage two started <<<<<")
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage two completed! base model is created and saved successfully <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e