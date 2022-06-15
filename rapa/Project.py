from . import base

import datarobot as dr

class Classification(base.RAPABase):
    """
        RAPA class meant for classification problems.
    """

    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = True 
        self._regression = False


class Regression(base.RAPABase):
    """
        RAPA class meant for regression problems.
    """
    
    def __init__(self, project: dr.Project = None):
        self.project = project
        self._classification = False 
        self._regression = True
