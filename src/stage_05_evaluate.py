import argparse
import os
import logging
from src.utils.common import read_yaml_file, save_json, get_timestamp
import numpy as np
import sklearn.metrics as metrics
import math
import tensorflow as tf

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def evaluate(config_path):
    config = read_yaml_file(config_path)
    artifacts = config["artifacts"]

    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["TRAINED_MODEL_DIR"])
    model_path = os.path.join(model_dir_path, 'trained_model.h5')

    model = tf.keras.models.load_model(model_path)
    x_test = np.load(os.path.join(artifacts['DATA_DIR'], 'X_test.npy')) / 255
    y_test = np.load(os.path.join(artifacts['DATA_DIR'], 'y_test.npy'))

    prediction_by_class = model.predict_proba(x_test)
    predictions = prediction_by_class[:, 1]

    PRC_json_path = config["plots"]["PRC"]
    ROC_json_path = config["plots"]["ROC"]

    scores_json_path = config["metrics"]["SCORES"]

    avg_prec = metrics.average_precision_score(y_test, predictions)
    roc_auc = metrics.roc_auc_score(y_test, predictions)

    scores = {
        "avg_prec": avg_prec,
        "roc_auc": roc_auc
    }

    save_json(scores_json_path, scores)

    precision, recall, prc_threshold = metrics.precision_recall_curve(y_test, predictions)
    nth_points = math.ceil(len(prc_threshold) / 1000)
    prc_points = list(zip(precision, recall, prc_threshold))[::nth_points]

    prc_data = {
        "prc": [
            {"precision": p, "recall": r, "threshold": t}
            for p, r, t in prc_points
        ]
    }

    save_json(PRC_json_path, prc_data)

    fpr, tpr, roc_threshold = metrics.roc_curve(y_test, predictions)

    roc_data = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold": t}
            for fp, tp, t in zip(fpr, tpr, roc_threshold)
        ]
    }

    save_json(ROC_json_path, roc_data)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n********************")
        evaluate(config_path=parsed_args.config)
        logging.info(f">>>>> stage evalation completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e


