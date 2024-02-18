from collections import namedtuple
import numpy as np


Node = namedtuple(
    typename="Node",
    field_names=["feat_ix", "threshold", "left", "right", "value"],
    defaults=[None] * 5,
)
BestSplit = namedtuple(
    typename="BestSplit",
    field_names=["left_ds", "right_ds", "feat_ix", "threshold", "score"],
    defaults=[None, None, None, None, None],
)


class DecisionTreeBase:
    def __init__(
        self,
        direction: str,
        max_depth: int = 5,
        min_samples_split: int = 3,
    ):
        """
        Build the decision tree base (not meant to be used as a model)
        direction: 'minimize' to minimize the value of the scoring function,
        'maximize' to maximize it
        max_depth: Do not split samples any further if the decision tree
        reaches this depth value
        min_samples_split: Do not split samples any further if the number
        of samples left to split becomes this number
        optim_dir
        """
        self.root: Node = None  # Initialize tree root
        # Optimization direction
        self.direction = direction

        # Stopping conditions
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, dataset, curr_depth: int = 0):
        """
        dataset: (N, D+1) array where
        N: Number of samples
        D: Number of features per sample
        +1: The label column
        curr_depth: Current depth within the tree
        """
        N, D = dataset.shape

        # Split samples until stopping conditions are met
        if N >= self.min_samples_split and curr_depth <= self.max_depth:
            # Get best split on the dataset
            best_split: BestSplit = self.get_best_split(dataset)

            # Was a good split actually found?
            if abs(best_split.score) != float("inf"):
                # Create left and right subtrees on that split of the data
                left_subtree = self.build_tree(best_split.left_ds, curr_depth + 1)
                right_subtree = self.build_tree(best_split.right_ds, curr_depth + 1)

                # Return a node holding the left and right subtrees
                return Node(
                    feat_ix=best_split.feat_ix,
                    threshold=best_split.threshold,
                    left=left_subtree,
                    right=right_subtree,
                )

        # Create leaf decision node over the labels
        # once no more subtrees can be made
        value = self.calc_feat_value(dataset[:, -1])
        return Node(value=value)

    def get_best_split(self, dataset) -> BestSplit:
        """
        This function finds the best split given the samples and their labels.
        By iterating over all features and all possible splits, we can find
        the best split by comparing the "score" function of each, whether that
        be the loss, information gain, etc.
        dataset: Same dimensions as in `build_tree`
        """
        best_split, optim_func = (
            (BestSplit(score=float("inf")), min)
            if self.direction == "minimize"
            else (BestSplit(score=float("-inf")), max)
        )
        X, y = dataset[:, :-1], dataset[:, -1]
        N, D = X.shape

        # Find the best feature to threshold and split the dataset on
        for feat_ix in range(D):
            thresholds = np.unique(X[:, feat_ix])
            for threshold in thresholds:
                left_ds, right_ds = self.split(dataset, feat_ix, threshold)
                if len(left_ds) > 0 and len(right_ds) > 0:
                    # Last column of left and right datasets is the labels column
                    score = self.calc_score(
                        root=y,
                        left=left_ds[:, -1],  # left labels
                        right=right_ds[:, -1],  # right labels
                    )
                    split = BestSplit(
                        left_ds=left_ds,
                        right_ds=right_ds,
                        feat_ix=feat_ix,
                        threshold=threshold,
                        score=score,
                    )
                    # Best split maximizes or minimizes the score
                    best_split = optim_func(best_split, split, key=lambda x: x.score)
        return best_split

    def split(self, dataset, feat_ix, threshold):
        """
        Create a split of the (N, D) dataset on
        the feature index given a threshold value
        """
        condition = dataset[:, feat_ix] <= threshold
        return dataset[condition], dataset[~condition]

    def calc_score(self, root, left, right) -> float:
        """
        Calculate the total score for this split of the data
        """
        raise NotImplementedError

    def calc_feat_value(self, labels) -> int | float:
        """
        This function calculates the feature value to be returned given
        a list of labels representing the labels for each sample at this
        leaf node
        """
        raise NotImplementedError

    def fit(self, X, y):
        """
        Training function for the decision tree
        X: (N, D) array of data
        y: (N,) array of labels
        """
        dataset = np.concatenate([X, y], axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        preds = [self.make_prediction(x, self.root) for x in X]
        return np.array(preds)

    def make_prediction(self, x, node: Node):
        # Base case, return the leaf node value if existing
        if node.value is not None:
            return node.value

        # Walk down the tree and recursively go through each node
        feat_val = x[node.feat_ix]
        if feat_val <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)


class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, max_depth: int = 5, min_samples_per_leaf: int = 3):
        super().__init__("minimize", max_depth, min_samples_per_leaf)

    def calc_score(self, root, left, right):
        """
        Calculate the SSE across both the left and right
        datasets
        """
        return self.calc_loss(left) + self.calc_loss(right)

    def calc_loss(self, labels):
        """
        SSE loss on the labels against
        the calculated feature value for this split
        """
        val = self.calc_feat_value(labels)
        return np.sum((labels - val) ** 2)

    def calc_feat_value(self, labels):
        """
        Feature value for each leaf is the average
        value of all "labels" where labels is a (N,) array
        """
        return np.mean(labels)

    def score(self, y_true, y_pred):
        """
        Sum of squared errors on predictions
        vs actual values
        """
        return np.sum((y_true - y_pred) ** 2)


class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, max_depth: int = 5, min_samples_per_leaf: int = 3):
        super().__init__("maximize", max_depth, min_samples_per_leaf)

    def calc_score(self, root, left, right) -> float:
        """
        Calculate the information gain of a particular split. The combined
        gini index of the child nodes will be subtracted from the gini index
        of the parent. The larger the information gain, the better the split.
        """
        weight_left = len(left) / len(root)
        weight_right = len(right) / len(root)
        gain = self.gini_index(root) - (
            weight_left * self.gini_index(left) + weight_right * self.gini_index(right)
        )
        return gain

    def gini_index(self, labels):
        """
        Compute average gini index over all unique labels
        gini = 1 - sum[(p_i)**2]
        where p_i is the probability of a label i belonging
        to a particular class
        """
        cls_labels = np.unique(labels)
        gini = 0
        for cls in cls_labels:
            p_cls = len(labels[labels == cls]) / len(labels)
            gini += p_cls**2
        return 1 - gini

    def calc_feat_value(self, labels) -> int | float:
        """
        Return the most frequent label in the leaf node
        """
        labels = list(labels)
        return max(labels, key=labels.count)

    def score(self, y_true, y_pred):
        """
        Accuracy score on predictions
        vs actual values
        """
        return np.mean((y_true == y_pred))


if __name__ == "__main__":
    np.random.seed(42)

    N, D, C = 50, 5, 2
    dataset = np.random.randn(N, D)

    """Decision Tree Regressor"""
    X, y = dataset[:, :-1], dataset[:, -1:]
    print("Creating and fitting a Decision Tree Regressor...")
    model = DecisionTreeRegressor()
    model.fit(X, y)
    print(
        "Regressor SSE Loss:",
        model.score(
            y_true=y.squeeze(),
            y_pred=model.predict(X),
        ),
    )
    print()

    """Decision Tree Classifier"""
    X, y = dataset, np.random.choice(C, size=(N, 1))
    print("Creating and fitting a Decision Tree Classifier...")
    model = DecisionTreeClassifier()
    model.fit(X, y)
    print(
        "Classifier Accuracy:",
        model.score(
            y_true=y.squeeze(),
            y_pred=model.predict(X),
        ),
    )
