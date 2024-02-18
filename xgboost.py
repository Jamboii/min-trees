import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from util import BCELoss
from decision_tree import DecisionTreeBase, Node, BestSplit
from gb_decision_tree import BaseGradBoost


class XGBoostRegressionTree(DecisionTreeBase):
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 3,
        reg: float = 1.0,
        gamma: float = 0.0,
    ):
        super().__init__("maximize", max_depth, min_samples_split)
        self.reg = reg
        self.gamma = gamma

    def build_tree(self, dataset, curr_depth=0):
        """
        dataset: (N,D+2) where +2 is the gradients, and hessians
        """
        N, _ = dataset.shape
        if N >= self.min_samples_split and curr_depth <= self.max_depth:
            # Find the best split of the current dataset
            best_split: BestSplit = self.find_best_split(dataset)
            # Create left and right subtrees of this split
            left = self.build_tree(best_split.left_ds, curr_depth + 1)
            right = self.build_tree(best_split.right_ds, curr_depth + 1)
            # Return the node representing this decision
            return Node(
                feat_ix=best_split.feat_ix,
                threshold=best_split.threshold,
                left=left,
                right=right,
            )

        # Create feature value node here
        value = self.calc_feat_value(grad=dataset[:, -2], hess=dataset[:, -1])
        return Node(value=value)

    def find_best_split(self, dataset):
        best_split = BestSplit(score=float("-inf"))
        X, grad, hess = dataset[:, :-2], dataset[:, -2], dataset[:, -1]
        N, D = X.shape
        grad_sum, hess_sum = grad.sum(), hess.sum()
        for feat_ix in range(D):
            # Initialize running gradients and hessians
            grad_left, hess_left = 0, 0

            # Extract out thresholds for this feature index
            # and evaluate in sorted order
            thresholds = X[:, feat_ix]
            sort_ix = np.argsort(thresholds)
            thresh_sort, grad_sort, hess_sort = (
                thresholds[sort_ix],
                grad[sort_ix],
                hess[sort_ix],
            )

            # Loop through each threshold in sorted order to keep a running
            # counter of gradients and hessians per subtree
            for thresh, g, h in zip(thresh_sort, grad_sort, hess_sort):
                # Update running gradients and hessians
                grad_left += g
                hess_left += h
                grad_right = grad_sum - grad_left
                hess_right = hess_sum - hess_left

                # Calculate gain value
                gain = 0.5 * (
                    self.calc_score(grad_left, hess_left)
                    + self.calc_score(grad_right, hess_right)
                    - self.calc_score(grad_sum, hess_sum)
                )
                -self.gamma

                # Evaluate for best split
                if gain > best_split.score:
                    left_ds, right_ds = self.split(dataset, feat_ix, thresh)
                    best_split = BestSplit(
                        left_ds=left_ds,
                        right_ds=right_ds,
                        feat_ix=feat_ix,
                        threshold=thresh,
                        score=gain,
                    )

        return best_split

    def fit(self, X, grad, hess):
        dataset = np.concatenate((X, grad, hess), axis=1)
        self.root = self.build_tree(dataset)

    def calc_score(self, grad_sum, hess_sum):
        return grad_sum**2 / (hess_sum + self.reg)

    def calc_feat_value(self, grad, hess):
        return -grad.sum() / (hess.sum() + self.reg)


class XGBoostModel(BaseGradBoost):
    def __init__(
        self, n_estimators=100, loss_fn=BCELoss(), lr=1e-1, reg=1.0, gamma=0.0
    ):
        super().__init__(n_estimators, loss_fn, lr)
        self.reg = reg
        self.gamma = gamma

    def fit(self, X, y):
        N, D = X.shape
        self.base_pred = np.random.normal(size=N)
        # Create random guess
        preds = self.base_pred
        for _ in range(self.n_estimators):
            # Get gradient of loss wrt the predictions
            dpreds = self.loss_fn.grad(preds, y)[:, None]
            # Get hessian of loss wrt the predictions
            ddpreds = self.loss_fn.hess(preds, y)[:, None]
            # Create a special "XGBooster" tree and train it
            estim = XGBoostRegressionTree(max_depth=3, reg=self.reg, gamma=self.gamma)
            estim.fit(X, grad=dpreds, hess=ddpreds)
            preds += self.lr * estim.predict(X)
            self.estimators.append(estim)

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

    print("Creating and fitting a XGBoost Classifier...")
    model = XGBoostModel()
    model.fit(Xtr, ytr)
    print(
        "Classifier Accuracy:",
        model.score(
            y_true=yte.squeeze(),
            y_pred=model.predict(Xte),
        ),
    )
