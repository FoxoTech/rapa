import pytest

import rapa

# test rapa.Project initialization
def test_Project_initialization():
    '''Checks that the rapa.Project objects can be initialized with each instance of arguments
    '''
    rapa_test_classification_project = rapa.Project.Classification()
    rapa_test_regression_project = rapa.Project.Regression()

    assert rapa_test_classification_project._classification == True and rapa_test_classification_project._regression == False
    assert rapa_test_regression_project._regression == True and rapa_test_regression_project._classification == False