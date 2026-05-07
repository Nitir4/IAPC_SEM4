from .linearregression import get_model as linear
from .ridge_model import get_model as ridge
from .randomforest import get_model as random_forest
from .gradientboost import get_model as gradient_boost
from .svm import get_model as svm
from .decision_tree import get_model as decision_tree

MODEL_MAP = {
    "linear": linear,
    "ridge": ridge,
    "random_forest": random_forest,
    "gradient_boost": gradient_boost,
    "svm": svm,
    "decision_tree": decision_tree
}
