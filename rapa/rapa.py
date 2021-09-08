from .base import RAPABase 
from . import utils

import datarobot as dr

class RAPAClassif(RAPABase):
    """
        RAPA class meant for classification problems.
    """

    # set the problem type
    _classification = True 
    _regression = False


class RAPARegress(RAPABase):
    """
        RAPA class meant for regression problems
    """

     # set the problem type
    _classification = False 
    _regression = True
