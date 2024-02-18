from collections import defaultdict

# from decision_tree import DecisionTreeClassifier  # uncomment to use our DT (much slower)
from sklearn.tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class AdaBoostClassifier:
    def __init__(self, n_estimators: int = 5):
        self.num_stumps = n_estimators
        self.eps = 1e-8

        # List object will hold each stump DT and its weight
        self.stumps = []

    def build_stump(self, X, y):
        """
        X: stump training samples (N, D)
        y: stump training labels (N,)
        """
        N, D = X.shape
        # Assuming binary classification for this example
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump.fit(X, y)
        # Calculate stump error and stump weight
        error = 1 - stump.score(X, y)
        stump_weight = 0.5 * np.log((1 - error) / (error + self.eps))
        # Add to overall stumps list
        stump.weight = stump_weight
        self.stumps.append(stump)

        # Update sample weights
        sw = np.array([1 / N for _ in range(N)])  # initial sample weights
        preds = stump.predict(X)
        clf_status = preds == y.squeeze()  # classification status

        # Update incorrect sample weights
        sw[clf_status is False] *= np.exp(stump_weight)
        # Update correct sample weights
        sw[clf_status is True] *= np.exp(-stump_weight)
        # Normalize new sample weights
        sw /= np.sum(sw)

        # Create dataset for next stump
        sample_ixs = np.random.choice(np.arange(N), size=N, replace=True, p=sw)
        return X[sample_ixs], y[sample_ixs]

    def build_stumps(self, X, y):
        for _ in range(self.num_stumps):
            X, y = self.build_stump(X, y)

    def fit(self, X, y):
        self.build_stumps(X, y)

    def predict(self, X):
        stump_preds = []
        for stump in self.stumps:
            preds = stump.predict(X)
            stump_preds.append(preds)

        # Determine the best weighted prediction per stump
        stump_preds = np.array(stump_preds).T  # (N, num_stumps)
        model_preds = []
        for row in stump_preds:
            model_pred = defaultdict(int)
            for i, pred in enumerate(row):
                model_pred[pred] += self.stumps[i].weight
            # Determine the class with the largest score
            ptow = {w: p for p, w in model_pred.items()}
            model_preds.append(ptow[max(ptow)])
        return model_preds

    def score(self, y_true, y_pred):
        """
        Accuracy score on predictions
        vs actual values
        """
        return np.mean((y_true == y_pred))


if __name__ == "__main__":
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y.reshape(-1, 1),
        test_size=0.2,
        random_state=42,
    )

    print("Creating and fitting a Random Forest Classifier...")
    model = RandomForestClassifier()
    model.fit(Xtr, ytr)
    print(
        "Classifier Accuracy:",
        model.score(
            y_true=yte.squeeze(),
            y_pred=model.predict(Xte),
        ),
    )

    print("Creating and fitting a AdaBoost Classifier...")
    model = AdaBoostClassifier()
    model.fit(Xtr, ytr)
    print(
        "Classifier Accuracy:",
        model.score(
            y_true=yte.squeeze(),
            y_pred=model.predict(Xte),
        ),
    )
