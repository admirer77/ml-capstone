from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

class TitanicModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }

    def grid_search(self, X_train, y_train, param_grid, cv=5):
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        return grid.best_params_, grid.best_score_