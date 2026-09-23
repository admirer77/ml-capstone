import argparse
import json
from data_prep import load_titanic, clean_titanic
from model import TitanicModel
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Train Titanic survival classifier")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--grid_search", action="store_true",
                         help="Run GridSearchCV instead of a single fit")
    args = parser.parse_args()

    df = clean_titanic(load_titanic())
    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    titanic_model = TitanicModel(n_estimators=args.n_estimators, max_depth=args.max_depth)

    if args.grid_search:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10]
        }
        best_params, best_score = titanic_model.grid_search(X_train, y_train, param_grid)
        print("Best params:", best_params)
        print("Best CV accuracy:", best_score)
    else:
        titanic_model.fit(X_train, y_train)

    results = titanic_model.evaluate(X_test, y_test)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()