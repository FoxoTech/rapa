from ..conf import * # datarobot project names/ids etc

import pytest

import rapa
import os
import pickle

import datarobot as dr


"""project_name = 'TUTORIAL_breast_cancer'
project_id = '62b5dc8249ed6b10669ab468'

featurelist_prefix = 'TEST_0.0.9'
featurelist_name = featurelist_prefix + ' (25)'
featurelist_id = '62b5de67a92c8927b1fd710b'

starred_model_id = '62b5e1aeddc5c75c4d91cf7a'

best_AUC_model_id = '62b5e1aeddc5c75c4d91cf84'"""


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
        1. Name provided matches project name exactly, and only one project exists with that name
        2. Project ID is provided, and there exists a project with that ID
        3. Name provided fetches one project, but not exact
        4. Name provided fetches more than one project
        5. Name provided fetches no project
    '''

    # 1. Name provided matches project name exactly, and only one project exists with that name
    bc_project = rapa.utils.find_project(project_name)
    assert bc_project is not None, f'1. Project using exact name not found (check that `{project_name}` still exists)'
    assert project_id == bc_project.id
    assert type(bc_project) is dr.Project

    # 2. Project ID is provided, and there exists a project with that ID
    bc_project = rapa.utils.find_project(project_id)
    assert bc_project is not None, f'2. Project using {project_id} not found (check that `{project_name}` still exists)'
    assert project_id == bc_project.id
    assert type(bc_project) is dr.Project

    # 3. Name provided fetches one project, but not exact
    substring = project_name[:12]
    bc_project = rapa.utils.find_project(substring)
    assert bc_project is not None, f'3. Project using {substring} not found (check that `{project_name}` still exists)'
    assert project_id == bc_project.id
    assert type(bc_project) is dr.Project

    # 4. Name provided fetches more than one project
    substring = project_name[:5]
    bc_project = rapa.utils.find_project(substring)
    assert bc_project is not None, f'4. Project using {substring} not found (check that `{project_name}` still exists)'
    assert project_id == bc_project.id
    assert type(bc_project) is dr.Project

    # 5. Name provided fetches no project
    wrong_project_name = project_name + 'string'
    bc_project = rapa.utils.find_project(wrong_project_name)
    assert bc_project is None, f'5. Project using {wrong_project_name} was found (it should not have found a project...?)'

# test retrieval of featurelists
@pytest.mark.order(3)
def test_datarobot_featurelist_retrieval():
    '''Checks that `rapa` retrieves the correct featurelist from DataRobot.

    Tests different situations:
        1. Name is provided exactly
        2. Featurelist id is provided
        3. Name is provided inexactly
        4. Name is provided and fetches no featurelist
    '''
    bc_project = rapa.utils.find_project(project_name)

    # 1. Name is provided exactly
    bc_featurelist = rapa.utils.get_featurelist(featurelist_name, bc_project)
    assert bc_featurelist is not None, f'1. The name provided: `{featurelist_name}` did not yield any results (check that `{project_name}` still exists)'
    assert featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 2. Featurelist id is provided
    
    bc_featurelist = rapa.utils.get_featurelist(featurelist_id, bc_project)
    assert bc_featurelist is not None, f'2. The id provided: `{featurelist_id}` did not yield any results (check that `{project_name}` still exists)'
    assert featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 3. Name is provided inexactly
    substring = featurelist_name[:-1]
    bc_featurelist = rapa.utils.get_featurelist(substring, bc_project)
    assert bc_featurelist is not None, f'3. The name provided: `{featurelist_name}` did not yield any results (check that `{project_name}` still exists)'
    assert featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 4. Name is provided and fetches no featurelist
    wrong_featurelist_name = featurelist_name + 'j923ifnoguhe'
    bc_featurelist = rapa.utils.get_featurelist(wrong_featurelist_name, bc_project)
    assert bc_featurelist is None, f'4. Featurelist using `{wrong_featurelist_name}` was found (it should not have found a featurelist...?)'

# test getting the starred model
@pytest.mark.order(4)
def test_datarobot_starred_model_retrieval():
    '''Tests that `rapa` can retrieve a starred model from DataRobot
    '''
    bc_project = rapa.utils.find_project(project_name)
    
    bc_starred_model = rapa.utils.get_starred_model(bc_project)
    assert bc_starred_model is not None, f'No starred model found for `{project_name}` when the Python `Logistic Regression` model should be starred.'
    assert starred_model_id == bc_starred_model.id
    assert type(bc_starred_model) is dr.Model

# test getting the best model for AUC
@pytest.mark.order(5)
def test_datarobot_best_model_retrieval():
    '''Tests that `rapa` retrieves the best model from Datarobot
    '''
    bc_project = rapa.utils.find_project(project_name)

    bc_best_model = rapa.utils.get_best_model(bc_project, metric='AUC')
    assert bc_best_model is not None, f'No best model found for `{project_name}`. Check the project still exists.'
    assert best_AUC_model_id == bc_best_model.id
    assert type(bc_best_model) is dr.Model
