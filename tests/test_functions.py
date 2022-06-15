import pytest
import rapa


### create_submittable_dataframe
class TestCreateSubmittableDataframe:
    # the number of samples matches the number provided
    projects = [rapa.Project.Classification(), rapa.Project.Regression()]
