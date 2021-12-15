from .base import RAPABase 
from . import utils
"""from . import base
base.RAPABase"""

import datarobot as dr

class RAPAClassif(RAPABase):
    """
        RAPA class meant for classification problems.
    """

    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = True 
        self._regression = False


class RAPARegress(RAPABase):
    """
        RAPA class meant for regression problems.
    """
    
    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = False 
        self._regression = True
