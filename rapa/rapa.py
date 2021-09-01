from .base import RAPABase 
from .utils import initialize_dr_api, find_project

import datarobot as dr

class RAPAClassif(RAPABase):
    """
        RAPA class meant for classification problems.
    """

    # set the problem type
    classification = True 
    regression = False


class RAPARegress(RAPABase):
    """
        RAPA class meant for regression problems
    """

     # set the problem type
    classification = False 
    regression = True
