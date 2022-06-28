import pytest

import rapa
import os
import pickle

import datarobot as dr


# test api initialization
@pytest.mark.order(1)
def test_api_initialization():
    '''Checks that the api can be connected with `utils.initialize_dr_api`.

    Currently only checks the default endpoint: https://app.datarobot.com/api/v2.
    '''
    print('test_api_inizialization test called')

    pkl_file_name = 'dr-tokens.pkl'
    dr_test_api_key = os.environ.get('DR_TEST_RAPA') # get the api key

    ## create the pickle file
    pickle.dump({'test':dr_test_api_key}, open(pkl_file_name, 'wb'))

    ## try connecting
    try:
        retval = rapa.utils.initialize_dr_api('test', pkl_file_name)
    except ValueError:
        ## delete the pickle file
        os.remove(pkl_file_name) 
        raise ValueError("API Key is Incorrect, check that the key is still valid in DataRobot.")

    ## delete the pickle file
    os.remove(pkl_file_name) 

    assert retval == None

# test retrieval of projects
@pytest.mark.order(2)
def test_datarobot_project_retrieval():
    '''Checks that `rapa` retrieves the correct project from DataRobot.
    
    Tests different situations:
        1. Name provided matches project name exactly, and only one project exists with that name.
        2. Project ID is provided, and there exists a project with that ID.
        3. Name provided fetches one project, but not exact.
        4. Name provided fetches more than one project.
        5. Name provided fetches no project.
    '''

    # 1. Name provided matches project name exactly, and only one project exists with that name
    bc_project = rapa.utils.find_project('TUTORIAL_breast_cancer')
    assert bc_project is not None, 'Project using exact name not found (check that `TUTORIAL_breast_cancer` still exists)'
    assert type(bc_project) is dr.Project
