from .linearregression import get_model as linear
from .ridge_model import get_model as ridge
from .randomforest import get_model as random_forest
from .gradientboost import get_model as gradient_boost

MODEL_MAP = {
    "linear": linear,
    "ridge": ridge,
    "random_forest": random_forest,
    "gradient_boost": gradient_boost,
}

# Experimental models (knn, lasso, svm, decision_tree)
# are available in models/_experimental/ but not
# included in the active pipeline.
