import typing as t
import numpy as np

from .bag_of_words import StupidBOW


class C45Node:
    value: t.Optional[str]
    feature_index: t.Optional[int]
    feature_threshold: t.Optional[np.float64]
    left: t.Optional["C45Node"]
    right: t.Optional["C45Node"]

    def __init__(
        self,
        value=None,
        feature_index: t.Optional[int] = None,
        feature_threshold: t.Optional[np.float64] = None,
        left: t.Optional["C45Node"] = None,
        right: t.Optional["C45Node"] = None,
    ):
        """
        C45Node represents a node in the c4.5 tree.

        Attributes:
            value: Class label, in this case 0 for clean, 1 for prompt injection detected
            feature_index: Index of the feature
            feature_threshold: For continuous features, found by calculating the information gain
                for multiple possible splits and chooses one that maximizes the gain.
            left: Left node
            right: Right node
        """
        self.value = value
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left = left
        self.right = right


class C45Tree:
    root: C45Node

    def __init__(self, labeled_documents: list[t.Tuple[str, int]]):
        self.bow = StupidBOW([d[0] for d in labeled_documents])
        labels = np.array([d[1] for d in labeled_documents])

        vectors = self.bow.vectors
        self.root = self._build_tree(vectors, labels)

    def _build_tree(self, vectors: np.ndarray, labels: np.ndarray) -> C45Node:
        if len(np.unique(labels)) == 1:
            return C45Node(value=labels[0])

        if vectors.shape[1] == 0:
            # most common label if no features left
            return C45Node(value=np.bincount(labels).argmax())

        best_feature = None
        best_gain_ratio = -1
        for feature in range(vectors.shape[1]):
            gain = gain_ratio(vectors, labels, feature)
            if gain > best_gain_ratio:
                best_gain_ratio = gain
                best_feature = feature

        # For continuous attributes, find threshold using median
        threshold = np.median(vectors[:, best_feature])

        # Split the dataset on the best feature
        left_indices = vectors[:, best_feature] <= threshold
        right_indices = vectors[:, best_feature] > threshold
        left = self._build_tree(vectors[left_indices], labels[left_indices])
        right = self._build_tree(vectors[right_indices], labels[right_indices])
        return C45Node(
            feature_index=best_feature,
            feature_threshold=threshold,
            left=left,
            right=right,
        )

    def predict(self, message: str):
        vector = np.array(self.bow.vectorize(message))
        current = self.root
        while current and current.value is None:
            if vector[current.feature_index] <= current.feature_threshold:
                current = current.left
            else:
                current = current.right
        assert current, "Unable to determine a hit, tree was misconstructed"
        return current.value


def gain_ratio(vectors, labels: np.ndarray, feature):
    info_gain = information_gain(vectors, labels, feature)
    intrinsic_value = -np.sum(
        [
            len(vectors[vectors[:, feature] == v])
            / len(vectors)
            * np.log2(len(vectors[vectors[:, feature] == v]) / len(vectors))
            for v in np.unique(vectors[:, feature])
        ]
    )
    return info_gain / intrinsic_value if intrinsic_value != 0 else 0


def information_gain(vectors: np.ndarray, labels: np.ndarray, feature):
    entropy_before = entropy(labels)

    # Sort the data based on the feature
    values = vectors[:, feature]
    unique_values = np.unique(values)

    # Calculate the weighted entropy after the split
    weighted_entropy = 0
    for value in unique_values:
        labels_subset = labels[values == value]
        weighted_entropy += (len(labels_subset) / len(labels)) * entropy(labels_subset)

    # Information gain is the difference in entropy
    return entropy_before - weighted_entropy


def entropy(labels: np.ndarray):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = [x / len(labels) for x in counts]

    # H(s) = E -(pi x log2(pi))
    return -np.sum(probabilities * np.log2(probabilities))
