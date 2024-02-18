from collections import namedtuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTreeClassifier

# Random Forest Classifier tree
RFCTree = namedtuple(typename="RFCTree", field_names=["dtree", "feat_ixs"])


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 10) -> None:
        self.num_trees = n_estimators
        self.ensemble = []

    def bootstrap(self, dataset: np.ndarray) -> np.ndarray:
        """
        Perform bootstrapping of data, randomly sampling features given a dataset
        The size of the randomly sampled dataset is the same as the input dataset
        It is intentional to have repeat rows
        """
        N, D = dataset.shape
        return dataset[np.random.choice(np.arange(N), size=N, replace=True), :]

    def select_feats(self, bootstrap: np.ndarray, num_feat: int) -> np.ndarray:
        """
        Randomly select a number of features to grab from the bootstrapped dataset
        """
        feat_per_tree = int(num_feat**0.5)  # sqrt(features) per tree
        feat_ixs = np.append(
            np.random.choice(
                np.arange(num_feat),
                size=feat_per_tree,
                replace=False,
            ),
            values=-1,
        )
        return bootstrap[:, feat_ixs], feat_ixs

    def build_dtree(self, feat_subset: np.ndarray) -> DecisionTreeClassifier:
        """
        Build and train a decision tree classifier on the sub-sampled training data
        """
        Xsub, ysub = feat_subset[:, :-1], feat_subset[:, -1:]  # (N, D), (D, 1)
        dtree = DecisionTreeClassifier()
        dtree.fit(Xsub, ysub)
        return dtree

    def build_ensemble(self, dataset):
        """
        Build an ensemble of decision trees, each trained on their own bootstrapped,
        feature-selected datasets
        """
        X = dataset[:, :-1]
        N, D = X.shape
        for _ in range(self.num_trees):
            # First randomly sample all of the data with replacement
            bootstrap = self.bootstrap(dataset)
            # Next get the features we want to keep
            feat_subset, feat_ixs = self.select_feats(bootstrap, num_feat=D)
            # Finally, build train and add a dtree classifier to the ensemble
            dtree = self.build_dtree(feat_subset)
            self.ensemble.append(
                RFCTree(
                    dtree=dtree,
                    feat_ixs=feat_ixs[:-1],  # do not include the label feature
                )
            )

    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.build_ensemble(dataset)

    def predict(self, X):
        """
        Make ensemble predictions where each DT in the ensemble
        has a vote of equal weight as to what the output label
        should be
        """
        ensemble_preds = []
        for ens_tree in self.ensemble:
            xf = X[:, ens_tree.feat_ixs]
            preds = ens_tree.dtree.predict(xf)  # (N,)
            ensemble_preds.append(preds)

        # Determine the most frequent prediction per sample
        ensemble_preds = np.array(ensemble_preds).T  # (N, num_trees)
        model_preds = []
        for row in ensemble_preds:
            values, counts = np.unique(row, return_counts=True)
            max_count_ix = np.argmax(counts)
            model_preds.append(values[max_count_ix])
        return model_preds

    def score(self, y_true, y_pred):
        """
        Accuracy score on predictions
        vs actual values
        """
        return np.mean((y_true == y_pred))


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
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
