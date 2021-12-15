from . import base

import datarobot as dr

class RAPAClassif(base.RAPABase):
    """
        RAPA class meant for classification problems.
    """

    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = True 
        self._regression = False


class RAPARegress(base.RAPABase):
    """
        RAPA class meant for regression problems.
    """
    
    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = False 
        self._regression = True
