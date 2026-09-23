"""Train and persist a Titanic survival classifier.

Usage:
    python train.py --n_estimators 150 --max_depth 8
    python train.py --grid_search
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

from data_utils import load_titanic, clean_titanic
from model import TitanicModel
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Titanic survival classifier")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--grid_search", action="store_true",
                         help="Run GridSearchCV instead of a single fit")
    parser.add_argument("--model_out", type=str, default="models/titanic_model.pkl",
                         help="Path to save the trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("Starting training run with args: %s", vars(args))

    df = clean_titanic(load_titanic())
    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    logger.info("Train/test split: %d train rows, %d test rows", len(X_train), len(X_test))

    titanic_model = TitanicModel(n_estimators=args.n_estimators, max_depth=args.max_depth)

    if args.grid_search:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
        }
        logger.info("Running GridSearchCV over %s", param_grid)
        best_params, best_score = titanic_model.grid_search(X_train, y_train, param_grid)
        logger.info("Best params: %s", best_params)
        logger.info("Best CV accuracy: %.4f", best_score)
    else:
        logger.info("Fitting model with n_estimators=%d, max_depth=%s",
                     args.n_estimators, args.max_depth)
        titanic_model.fit(X_train, y_train)

    results = titanic_model.evaluate(X_test, y_test)
    logger.info("Final test accuracy: %.4f", results["accuracy"])
    logger.info("Confusion matrix: %s", results["confusion_matrix"])

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    model_path = Path(args.model_out)
    with open(model_path, "wb") as f:
        pickle.dump(titanic_model.model, f)
    logger.info("Trained model saved to %s", model_path)


if __name__ == "__main__":
    main()