from .. import conf # datarobot project names/ids etc

import pytest

import rapa
import os
import pickle

import datarobot as dr


"""
Tests the utility functions of rapa.

Tests are performend in the order they appear in the file, from the 
top down (using pytest-order)
"""


# test api initialization
@pytest.mark.order(1)
def test_api_initialization():
    '''Checks that the api can be connected with `utils.initialize_dr_api`.

    Currently only checks the default endpoint: https://app.datarobot.com/api/v2.

    Tests different situations:
        1. try connecting without a file
        2. try connecting with the wrong dictionary key
        3. try connecting with a wrong api key
        4. try connecting correctly
    '''
    pkl_file_name = 'dr-tokens.pkl'

    # 1. try connecting without a file
    try:
        retval = rapa.utils.initialize_dr_api('test', pkl_file_name)
    except ValueError:
        ## delete the pickle file
        os.remove(pkl_file_name) 
        raise ValueError("API Key is Incorrect, check that the key is still valid in DataRobot.")
    except FileNotFoundError:
        # this is expected
        pass
    else:
        raise Exception("DataRobot connected without an API key...")

    
    dr_test_api_key = os.environ.get('DR_TEST_RAPA') # get the api key

    ## create the pickle file
    pickle.dump({'test':dr_test_api_key, 'wrong':'1234'}, open(pkl_file_name, 'wb'))

    del dr_test_api_key # delete the api key for security reasons ..?

    # 2. try connecting with the wrong dictionary key
    try:
        wrong_api_key = 'abcdefghijklmnopqrstuvwxyz'
        retval = rapa.utils.initialize_dr_api(wrong_api_key, pkl_file_name)
    except KeyError:
        # this is expected
        pass
    else:
        raise Exception(f"`{wrong_api_key}` somehow initialized the DataRobot API")

    # 3. try connecting with a wrong api key
    try:
        wrong_dict_key = 'wrong'
        retval = rapa.utils.initialize_dr_api(wrong_dict_key, pkl_file_name)
    except Exception:
        # this is expected
        pass
    else:
        raise Exception(f'`{wrong_dict_key}` somehow pulled a key from the pickled dictionary')

    # 4. try connecting correctly
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
    bc_classification_project = rapa.utils.find_project(conf.classification_project_name)
    assert bc_classification_project is not None, f'1. Project using exact name not found (check that `{conf.classification_project_name}` still exists)'
    assert conf.classification_project_id == bc_classification_project.id
    assert type(bc_classification_project) is dr.Project

    # 2. Project ID is provided, and there exists a project with that ID
    bc_classification_project = rapa.utils.find_project(conf.classification_project_id)
    assert bc_classification_project is not None, f'2. Project using {conf.classification_project_id} not found (check that `{conf.classification_project_name}` still exists)'
    assert conf.classification_project_id == bc_classification_project.id
    assert type(bc_classification_project) is dr.Project

    # 3. Name provided fetches one project, but not exact
    substring = conf.classification_project_name[:12]
    bc_classification_project = rapa.utils.find_project(substring)
    assert bc_classification_project is not None, f'3. Project using {substring} not found (check that `{conf.classification_project_name}` still exists)'
    assert conf.classification_project_id == bc_classification_project.id
    assert type(bc_classification_project) is dr.Project

    # 4. Name provided fetches more than one project
    substring = conf.classification_project_name[:5]
    bc_classification_project = rapa.utils.find_project(substring)
    assert bc_classification_project is not None, f'4. Project using {substring} not found (check that `{conf.classification_project_name}` still exists)'
    assert conf.classification_project_id == bc_classification_project.id
    assert type(bc_classification_project) is dr.Project

    # 5. Name provided fetches no project
    wrong_project_name = conf.classification_project_name + 'string'
    try:
        bc_classification_project = rapa.utils.find_project(wrong_project_name)
    except Exception:
        # excpected
        pass
    else:
        raise Exception(f'5. Project using {wrong_project_name} was found (it should not have found a project...?)')

# test retrieval of featurelists
@pytest.mark.order(3)
def test_datarobot_featurelist_retrieval():
    '''Checks that `rapa` retrieves the correct featurelist from DataRobot.

    Tests different situations:
        1. Name is provided exactly
        2. Featurelist id is provided
        3. Name is provided inexactly
        4. Name is provided and fetches no featurelist
        5. Name is provided and fetches more than one featurelist
    '''
    bc_classification_project = rapa.utils.find_project(conf.classification_project_name)

    # 1. Name is provided exactly
    bc_featurelist = rapa.utils.get_featurelist(conf.featurelist_name, bc_classification_project)
    assert bc_featurelist is not None, f'1. The name provided: `{conf.featurelist_name}` did not yield any results (check that `{conf.classification_project_name}` still exists)'
    assert conf.featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 2. Featurelist id is provided
    
    bc_featurelist = rapa.utils.get_featurelist(conf.featurelist_id, bc_classification_project)
    assert bc_featurelist is not None, f'2. The id provided: `{conf.featurelist_id}` did not yield any results (check that `{conf.classification_project_name}` still exists)'
    assert conf.featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 3. Name is provided inexactly
    substring = conf.featurelist_name[:-1]
    bc_featurelist = rapa.utils.get_featurelist(substring, bc_classification_project)
    assert bc_featurelist is not None, f'3. The name provided: `{conf.featurelist_name}` did not yield any results (check that `{conf.classification_project_name}` still exists)'
    assert conf.featurelist_id == bc_featurelist.id
    assert type(bc_featurelist) is dr.Featurelist

    # 4. Name is provided and fetches no featurelist
    wrong_featurelist_name = conf.featurelist_name + 'j923ifnoguhe'
    try:
        bc_featurelist = rapa.utils.get_featurelist(wrong_featurelist_name, bc_classification_project)
    except Exception:
        # expected
        pass
    else:
        raise Exception(f'4. Featurelist using `{wrong_featurelist_name}` to search was found (it should not have found a featurelist...?)')

    # 5. Name is provided and fetches more than one featurelist
    substring = conf.featurelist_name[:-4]
    bc_featurelist = rapa.utils.get_featurelist(substring, bc_classification_project)
    assert bc_featurelist is not None, f'5. Search for featurelist using `{substring}` was not found (meant to find multiple featurelists)'

# test getting the starred model
@pytest.mark.order(4)
def test_datarobot_starred_model_retrieval():
    '''Tests that `rapa` can retrieve a starred model from DataRobot
    '''
    # classification
    bc_classification_project = rapa.utils.find_project(conf.classification_project_name)
    
    bc_classification_starred_model = rapa.utils.get_starred_model(bc_classification_project)
    assert bc_classification_starred_model is not None, f'No starred model found for `{conf.classification_project_name}` when the Python `Logistic Regression` model should be starred.'
    assert conf.classification_starred_model_id == bc_classification_starred_model.id
    assert type(bc_classification_starred_model) is dr.Model

    # regression
    bc_regression_project = rapa.utils.find_project(conf.regression_project_name)
    
    bc_regression_starred_model = rapa.utils.get_starred_model(bc_regression_project)
    assert bc_regression_starred_model is not None, f'No starred model found for `{conf.regression_project_name}` when the Python `Logistic Regression` model should be starred.'
    assert conf.regression_starred_model_id == bc_regression_starred_model.id
    assert type(bc_regression_starred_model) is dr.Model

# test getting the best model for AUC
@pytest.mark.order(5)
def test_datarobot_best_model_retrieval():
    '''Tests that `rapa` retrieves the best model from Datarobot

    Tests different situations:
        1. it gets the best model
        2. gets the best of two starred models
        3. gets best model with prefix
    '''

    # classification
    bc_classification_project = rapa.utils.find_project(conf.classification_project_name)

    # 1. gets best model
    bc_best_classification_model = rapa.utils.get_best_model(bc_classification_project, metric='AUC')
    assert bc_best_classification_model is not None, f'No best model found for `{conf.classification_project_name}`. Check the project still exists.'
    assert conf.best_AUC_model_id == bc_best_classification_model.id
    assert type(bc_best_classification_model) is dr.Model

    # 2. gets the best of two starred models
    temporary_starred_model = dr.models.Model.get(bc_classification_project.id, '62b5e1aeddc5c75c4d91cf84')
    # temporarily star the best model
    temporary_starred_model.star_model()
    bc_best_classification_model = rapa.utils.get_best_model(bc_classification_project, starred=True)
    temporary_starred_model.unstar_model()

    assert bc_best_classification_model is not None, f'No best model found for `{conf.classification_project_name}`. Check the project still exists.'
    assert conf.best_AUC_model_id == bc_best_classification_model.id
    assert type(bc_best_classification_model) is dr.Model

    # 3. gets best model with prefix
    bc_best_classification_model = rapa.utils.get_best_model(bc_classification_project, metric='AUC', featurelist_prefix='TEST')
    assert bc_best_classification_model is not None, f'No best model found for `{conf.classification_project_name}`. Check the project still exists.'
    assert conf.best_AUC_model_id == bc_best_classification_model.id
    assert type(bc_best_classification_model) is dr.Model

    # regression
    bc_regression_project = rapa.utils.find_project(conf.regression_project_name)

    # 1. gets best model
    bc_best_regression_model = rapa.utils.get_best_model(bc_regression_project, metric='RMSE', highest=False)
    assert bc_best_regression_model is not None, f'No best model found for `{conf.regression_project_name}`. Check the project still exists.'
    assert conf.best_RMSE_model_id == bc_best_regression_model.id
    assert type(bc_best_regression_model) is dr.Model


    # 3. gets best model with prefix
    bc_best_regression_model = rapa.utils.get_best_model(bc_regression_project, metric='RMSE', featurelist_prefix='Informative Features - Leakage Removed')
    assert bc_best_regression_model is not None, f'No best model found for `{conf.regression_project_name}`. Check the project still exists.'
    assert conf.best_RMSE_model_id == bc_best_regression_model.id
    assert type(bc_best_regression_model) is dr.Model
