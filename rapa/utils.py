import datarobot as dr
from datarobot.errors import ClientError

import pickle

import logging
from warnings import warn
from warnings import catch_warnings

from statistics import mean

LOGGER = logging.getLogger(__name__)


def find_project(project: str) -> dr.models.project.Project:
    """Uses the DataRobot api to find a current project.

    Uses datarobot.Project.get() and dr.Project.list() to test if 'project' is either an id
    or possibly a name of a project in DataRobot, then returns the project found.

    ## Parameters:
    ----------
    project: str
        Either a project id or a search term for project name

    ## Returns: 
    ----------
    datarobot.models.Project
        A datarobot project object that is either the project with the id provided, or the 
        first/only project returned by searching by project name. Returns None if the list is 
        empty.
    """

    project = str(project) # make sure the project id/name provided is a string
    return_project = None

    try: # try finding project with id
        return_project = dr.Project.get(project_id=project)
        return return_project
    except ClientError: # id was not provided, most likely a name
        project_list = dr.Project.list(search_params={'project_name': project})
        if len(project_list) == 0: # probably wrong id, check id?
            warn(f"No projects found with id or search for \'{project}\'")
            return None
        elif len(project_list) == 1: # found one project with search, good
            return project_list[0]
        else: # more than one project was found
            warn(f"Returning the first of multiple projects with \'{project}\': {project_list=}")
            return project_list[0]
    

def get_best_model(project: dr.models.project.Project, starred: bool = False, cv_score: str = 'AUC') -> dr.models.model.Model:
    """Attempts to find the 'best' model in a datarobot by averaging cross validation scores of all the
    models in a supplied project.

    CURRENTLY SUPPORTS METRICS WHERE HIGHER = BETTER

    <mark>WARNING</mark>: Actually finding the 'best' model takes more than averageing cross validation
    scores, and it is suggested that the 'best' model is decided and starred in DataRobot.
    (Make sure 'starred = True' if starring the 'best' model) 

    Note: Some models may not have cross validation scores because they were not run. These
    models are ignored by this function. Cross validate all models if each model should be 
    considered.

    ## Parameters 
    ----------
    project: datarobot.models.project.Project
        The project object that will be searched for the 'best' model

    starred: bool, optional (default = False)
        If True, return the starred model. If there are more than one starred models,
        then warn the user and return the 'best' one

    cv_score: str, optional (default = 'AUC')
        What model cross validation metric to use when averaging scores
    
    ## Returns
    ----------
    datarobot.models.Model
        A datarobot model that is either the 'best', starred, or the 'best' of the starred models
        from the provided datarobot project
    """

    all_models = project.get_models() # Retrieve all models from the supplied project

    if starred: # if the model is starred logic
        starred_models = []
        for model in all_models: # find each starred model
            if model.is_starred:
                starred_models.append(model)
        if len(starred_models) == 0:
            warn(f'There are no starred models in \'{project}\'. Will try to return the \'best\' model.')
        elif len(starred_models) == 1: # if there is a starred model, return it regardless of whether or not it has been cross-validated
            return starred_models[0]
        else: # more than one model is starred
            warn(f'More than one model is starred: {starred_models}. Will try to return the \'best\' of these models.')
            averages = {} # keys are average scores and values are the models
            num_no_cv = 0
            for starred_model in starred_models:
                try:
                    averages[mean(starred_model.get_cross_validation_scores()['cvScores'][cv_score].values())] = starred_model
                except ClientError: # the model wasn't cross-validated
                    num_no_cv += 1
            if len(averages) == 0:
                warn(f'The starred models were not cross-validated!')
                return None
            else:
                return averages[sorted(averages.keys())[-1]] # highest metric is 'best' TODO: support the other metrics
    else: # starred == False
        if len(all_models) > 20: # arbitrarily chosen
            warn(f'There are \'{len(all_models)}\'. Keep in mind it may take a while to obtain each CV score (~1 second per model).') # TODO: make it faster
        averages = {} # keys are average scores and values are the models
        num_no_cv = 0
        for model in all_models:
            try:
                averages[mean(model.get_cross_validation_scores()['cvScores'][cv_score].values())] = model
            except ClientError: # the model wasn't cross-validated
                num_no_cv += 1
            if len(averages) == 0:
                warn(f'There were no cross-validated models in \'{project=}\'')
                return None
            else:
                return averages[sorted(averages.keys())[-1]] # highest metric is 'best' TODO: support the other metrics


def initialize_dr_api(token_key, file_path: str = 'data/dr-tokens.pkl', endpoint: str = 'https://app.datarobot.com/api/v2') -> None:
    """Initializes the DataRobot API with a pickled dictionary created by the user.

    <mark>WARNING</mark>: It is advised that the user keeps the pickled dictionary in an ignored 
    directory if using GitHub (put the file in the .gitignore)

    Accesses a file that should be a pickled dictionary. This dictionary has the API token
    as the value to the provided token_key. Ex: {token_key: 'API_TOKEN'}

    ## Parameters
    ----------
    token_key: str | int | etc...
        The API token's key in the pickled dictionary located in file_path

    file_path: str, optional (default = 'data/dr-tokens.pkl')
        Path to the pickled dictionary containing the API token

    endpoint: str, optional (default = 'https://app.datarobot.com/api/v2')
        The endpoint is usually the URL you would use to log into the DataRobot Web User Interface

    """
    # load pickled dictionary and initialize api, catching FileNotFound, KeyError, and failed authentication warning
    try:
        datarobot_tokens = pickle.load(open(file_path, 'rb'))
        with catch_warnings(record=True) as w: # appends warning to w if warning occurs
            dr.Client(endpoint=endpoint, token=datarobot_tokens[token_key])
            if not not w: # check to see if w is not None or empty (has a warning)
                raise Exception(w[0].message)
            else:
                pass
    except FileNotFoundError:
        raise FileNotFoundError(f'The file {file_path} does not exist.') # TODO: Make a tutorial on how to create the pickled dictionary with api tokens and link here
    except KeyError:
        raise KeyError(f'\'{token_key}\' is not in the dictionary at \'{file_path}\'')
    
    # TODO: I probably didn't catch all errors, make tests for this
    
    print(f'DataRobot API initiated with endpoint \'{endpoint}\'')