
from sklearn.tree import DecisionTreeClassifier


def build_model():
    return DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )
