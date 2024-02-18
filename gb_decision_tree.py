import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from util import BCELoss

# from decision_tree import DecisionTreeRegressor # uncomment to use our DT (much slower)
from sklearn.tree import DecisionTreeRegressor


class BaseGradBoost:
    """Base gradient boost class functionality"""

    def __init__(self, n_estimators, loss_fn, lr):
        self.n_estimators = n_estimators
        self.loss_fn = loss_fn
        self.lr = lr
        self.estimators = []
        self.base_pred = None

    def fit(self, X, y):
        N, D = X.shape
        self.base_pred = np.random.normal(size=N)
        # Create random guess
        preds = self.base_pred
        for _ in range(self.n_estimators):
            # Get gradient of loss wrt the predictions
            dpreds = self.loss_fn.grad(preds, y)
            # Create a new estimator that learns the gradient updates
            estim = DecisionTreeRegressor(max_depth=3)
            # Negative direction of the gradient will reduce loss
            estim.fit(X, y=(-self.lr * dpreds).reshape(-1, 1))
            # Update the predictions with the model's guess at the gradient updates
            preds += estim.predict(X)
            # Save the created estimator
            self.estimators.append(estim)

    def predict(self, X):
        """
        Loop through each of the gradient boost estimators,
        sum up each of the residuals
        """
        return sum(
            (estim.predict(X) for estim in self.estimators),
            start=self.base_pred,
        )


class GradBoostClassifier(BaseGradBoost):
    def __init__(self, n_estimators=100, loss_fn=BCELoss(), lr=1e-1):
        super().__init__(n_estimators, loss_fn, lr)

    def predict(self, X):
        N, _ = X.shape
        self.base_pred = np.random.normal(size=N)

        raw_pred = super().predict(X)
        probs_pos_cls = self.loss_fn.raw_pred_to_proba(raw_pred)
        return probs_pos_cls > 0.5

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
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Creating and fitting a Gradient Boosted Classifier...")
    model = GradBoostClassifier()
    model.fit(Xtr, ytr)
    print(
        "Classifier Accuracy:",
        model.score(
            y_true=yte.squeeze(),
            y_pred=model.predict(Xte),
        ),
    )
