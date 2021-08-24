from .base import RAPABase 

from sklearn.feature_selection import f_regression, f_classif
from sklearn.model_selection import StratifiedKFold, KFold

import datarobot as dr

class RAPAClassif(RAPABase):
    """
        RAPA class meant for classification problems.
    """
    def __init__(self):
        pass

class RAPARegress(RAPABase):
    """
        RAPA class meant for regression problems
    """
    def __init__(self):
        pass