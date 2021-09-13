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
    project = None

    def __init__(self, project: dr.Project = None):
        self.project = project


class RAPARegress(RAPABase):
    """
        RAPA class meant for regression problems
    """

     # set the problem type
    _classification = False 
    _regression = True
    project = None
    
    def __init__(self, project: dr.Project = None):
        self.project = project
