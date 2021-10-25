from . import _config

import datarobot as dr
from datarobot.errors import ClientError

import pickle

import logging
from warnings import warn
from warnings import catch_warnings
from datarobot.models import featurelist

import pandas as pd
import numpy as np
from statistics import mean
from statistics import median

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sb

LOGGER = logging.getLogger(__name__)


def find_project(project: str) -> dr.Project:
    """Uses the DataRobot api to find a current project.

    Uses datarobot.Project.get() and dr.Project.list() to test if 'project' is either an id
    or possibly a name of a project in DataRobot, then returns the project found.

    ## Parameters:
    ----------
    project: str
        Either a project id or a search term for project name

    ## Returns: 
    ----------
    datarobot.Project
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
    

# if changing get_best_model, check if it's alias get_starred_model needs changing
def get_best_model(project: dr.Project, 
                featurelist_prefix: str = None, 
                starred: bool = False, 
                metric: str = 'AUC') -> dr.Model:
    """Attempts to find the 'best' model in a datarobot by searching cross validation scores of all the
    models in a supplied project. # TODO make dictionary for minimize/maximize 

    CURRENTLY SUPPORTS METRICS WHERE HIGHER = BETTER

    WARNING: Actually finding the 'best' model takes more than averageing cross validation
    scores, and it is suggested that the 'best' model is decided and starred in DataRobot.
    (Make sure 'starred = True' if starring the 'best' model) 

    Note: Some models may not have cross validation scores because they were not run. These
    models are ignored by this function. Cross validate all models if each model should be 
    considered.

    ## Parameters 
    ----------
    project: datarobot.Project
        The project object that will be searched for the 'best' model

    featurelist_prefix: str, optional (default = 'RAPA Reduced to')
        The desired featurelist prefix used to search in for models using specific
        rapa featurelists

    starred: bool, optional (default = False)
        If True, return the starred model. If there are more than one starred models,
        then warn the user and return the 'best' one

    metric: str, optional (default = 'AUC')
        What model cross validation metric to use when averaging scores
    
    ## Returns
    ----------
    datarobot.Model
        A datarobot model that is either the 'best', starred, or the 'best' of the starred models
        from the provided datarobot project
    """

    all_models = []
    if featurelist_prefix: # if featurelist_prefix is not none or empty
        for model in project.get_models():
            if model.featurelist_name != None:
                if model.featurelist_name.lower().startswith(featurelist_prefix.lower()):
                    all_models.append(model)
    else:
        all_models = project.get_models() # Retrieve all models from the supplied project
    
    if len(all_models) == 0:
        return None

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
            averages = {} # keys are average scores and values are the models
            num_no_cv = 0
            for starred_model in starred_models:
                try:
                    averages[mean(starred_model.get_cross_validation_scores()['cvScores'][metric].values())] = starred_model
                except ClientError: # the model wasn't cross-validated
                    num_no_cv += 1
            if len(averages) == 0:
                warn(f'The starred models were not cross-validated!')
                return None
            else:
                return averages[sorted(averages.keys())[-1]] # highest metric is 'best' TODO: support the other metrics
    else: # starred == False
        averages = {} # keys are average scores and values are the models
        num_no_cv = 0
        for model in all_models:
            try:
                averages[mean(model.get_cross_validation_scores()['cvScores'][metric].values())] = model
            except ClientError: # the model wasn't cross-validated
                num_no_cv += 1

        if len(averages) == 0:
            warn(f'There were no cross-validated models in \'{project=}\'')
            return None
        else:
            return averages[sorted(averages.keys())[-1]] # highest metric is 'best' TODO: support the other metrics

# alias for get_best_model
def get_starred_model(project: dr.Project, 
                    metric: str = 'AUC',
                    featurelist_prefix: str = None) -> dr.Model:
    """Alias for rapa.utils.get_best_model() but makes starred = True
    """
    return get_best_model(project, starred = True, metric = metric, featurelist_prefix = featurelist_prefix)


def initialize_dr_api(token_key, 
                    file_path: str = 'data/dr-tokens.pkl', 
                    endpoint: str = 'https://app.datarobot.com/api/v2'):
    """Initializes the DataRobot API with a pickled dictionary created by the user.

    <mark>WARNING</mark>: It is advised that the user keeps the pickled dictionary in an ignored 
    directory if using GitHub (put the file in the .gitignore)

    Accesses a file that should be a pickled dictionary. This dictionary has the API token
    as the value to the provided token_key. Ex: {token_key: 'API_TOKEN'}

    ## Parameters
    ----------
    token_key: str
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


def get_featurelist(featurelist: str, 
                    project: dr.Project) -> dr.Featurelist:
    """Uses the DataRobot api to search for a desired featurelist.

    Uses datarobot.Project.get_featurelists() to retrieve all the featurelists in
    the project. Then, it searches the list for id's, and if it doesn't find any,
    it searches the list again for names. Returns the first project it finds.

    ## Parameters
    ----------
    featurelist: str
        Either a featurelist id or a search term for featurelist name
    
    project: datarobot.Project
        The project that is being searched for the featurelist

    ## Returns
    ----------
    datarobot.Featurelist
        The featurelist that was found. Returns None if no featurelist is found
    """
    featurelist = str(featurelist) # cast to string just in case id is an int or something
    featurelists = project.get_featurelists()
    dr_featurelist = [x for x in featurelists if featurelist == x.id] # loop over all the featurelists and get all that match featurelist (assuming it is an id)
    if dr_featurelist: # if dr_featurelist is not empty 
        return dr_featurelist[0] # there should only be one id
    else: # if dr_featurelist is empty
        dr_featurelist = [x for x in featurelists if featurelist.lower() in str(x.name).lower()] # use python's `in` to search strings
        if not dr_featurelist: # if dr_featurelist is empty
            warn(f'No featurelists were found with either the id or name of \'{featurelist}\'')
            return None
        elif len(dr_featurelist) > 1: # if dr_featurelist has more than 1
            warn(f'More than one featurelist were found: \'{dr_featurelist}\', returning the first.')
            return dr_featurelist[0]
        else: # dr_Featurelist has 1
            return dr_featurelist[0]
    
    
def parsimony_performance_boxplot(project: dr.Project, 
                                featurelist_prefix: str = 'RAPA Reduced to',
                                metric: str = 'AUC',
                                split: str = 'crossValidation'):
    """Uses `seaborn`'s `boxplot` function to plot featurelist size vs performance
    for all models that use that featurelist. # TODO warn about multiple prefixes, try to use new prefixes

    ## Paremeters
    ----------
    project: datarobot.Project
        Either a datarobot project, or a string of it's id or name

    featurelist_prefix: str, optional (default = 'RAPA Reduced to')
        The desired prefix for the featurelists that will be used for plotting parsimony performance. Each featurelist
        will start with the prefix, include a space, and then end with the number of features in that featurelist

    metric: str, optional (default = 'AUC')
        The metric used for plotting accuracy of models

    split: str, optional (default = 'crossValidation')
        What split's performance to take from. 
        Can be: ['crossValidation', 'holdout'] TODO: i think it can be more, double check

    ## Returns
    ----------
    TODO: what does a matplotlib.pyplot plot return? should i even return a plot?
    """
    # if `project` is a string, find the project
    if type(project) is str:
        project = find_project(project)

    datarobot_project_models = project.get_models() # get all the models in the provided project
    RAPA_model_featurelists = []
    featurelist_performances = defaultdict(list)
    for model in datarobot_project_models: # for every model, if the model has the prefix, then add it's performance
        if model.featurelist_name != None and featurelist_prefix in model.featurelist_name:
            RAPA_model_featurelists.append(model.featurelist_name)
            num_features = int(model.featurelist_name.split(' ')[-1]) # parse the number of features from the featurelist name
            featurelist_performances[num_features].append(model.metrics[metric][split])

    featurelist_performances_df = pd.DataFrame(featurelist_performances)[sorted(featurelist_performances.keys())]
    featurelist_performances_df = featurelist_performances_df.dropna(how="all", axis=1).dropna()
    featurelist_performances_df.columns = [str(x) for x in featurelist_performances_df.columns][::-1]

    with plt.style.context('tableau-colorblind10'):
        plt.ylabel(f'{split} {metric}')
        plt.xlabel('Number of Features')
        return(sb.boxplot(data=featurelist_performances_df))
    
def feature_performance_stackplot(project: dr.Project, 
                                featurelist_prefix: str = 'RAPA Reduced to',
                                starting_featurelist: str = None,
                                feature_importance_metric: str = 'median',
                                metric: str = 'AUC'):
    """Utilizes `matplotlib.pyplot.stackplot` to show feature performance during 
    parsimony analysis.

    ## Parameters
    ----------
    project: datarobot.Project
        Either a datarobot project, or a string of it's id or name

    featurelist_prefix: str, optional (default = 'RAPA Reduced to')
        The desired prefix for the featurelists that will be used for plotting feature performance. Each featurelist
        will start with the prefix, include a space, and then end with the number of features in that featurelist

    starting_featurelist: str, optional (default = None)
        The starting featurelist used for parsimony analysis. If None, only
        the featurelists with the desired prefix in `featurelist_prefix` will be plotted
    
    feature_importance_metric: str, optional (default = mean)
        Which metric to use when finding the  most representative feature importance of all models in the featurelist
            Options: 'median', 'mean', or 'cumulative'

    metric: str, optional (default = 'AUC')
        Which metric to use when finding feature importance of each model

    ## Returns
    ----------
    TODO: what does a matplotlib.pyplot plot return? should i even return a plot?
    """
    # if `project` is a string, find the project
    if type(project) is str:
        project = find_project(project)
    
    if type(starting_featurelist) == str:
        starting_featurelist = get_featurelist(starting_featurelist, project)
    
    datarobot_project_models = project.get_models() # get all the models in the provided project

    if starting_featurelist != None: # have the starting featurelist as well
        all_feature_importances = {}
        for model in datarobot_project_models:
            if model.featurelist_name != None and (model.featurelist_name.startswith(featurelist_prefix) or model.featurelist_id == starting_featurelist.id): # if the model uses the starting featurelist/featurelist prefix
                if model.metrics[metric]['crossValidation'] != None:
                    if model.featurelist_name in all_feature_importances.keys():
                        for x in model.get_feature_impact():
                            if x['featureName'] in all_feature_importances[model.featurelist_name].keys():
                                all_feature_importances[model.featurelist_name][x['featureName']].append(x['impactNormalized'])
                            else:
                                all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
                    else:
                        all_feature_importances[model.featurelist_name] = {} 
                        for x in model.get_feature_impact():
                            all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
    else: # same as if, but without starting featurelist
        all_feature_importances = {}
        for model in datarobot_project_models:
            if model.featurelist_name.startswith(featurelist_prefix): # if the model's featurelist starts with the featurelist prefix
                if model.metrics[metric]['crossValidation'] != None:
                    if model.featurelist_name in all_feature_importances.keys():
                        for x in model.get_feature_impact():
                            if x['featureName'] in all_feature_importances[model.featurelist_name].keys():
                                all_feature_importances[model.featurelist_name][x['featureName']].append(x['impactNormalized'])
                            else:
                                all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
                    else:
                        all_feature_importances[model.featurelist_name] = {} 
                        for x in model.get_feature_impact():
                            all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]

    for featurelist_name in all_feature_importances.keys():
        for feature in all_feature_importances[featurelist_name].keys():
            if feature_importance_metric.lower() == 'median':
                all_feature_importances[featurelist_name][feature] = median(all_feature_importances[featurelist_name][feature])
            elif feature_importance_metric.lower() == 'mean':
                all_feature_importances[featurelist_name][feature] = mean(all_feature_importances[featurelist_name][feature])
            elif feature_importance_metric.lower() == 'cumulative':
                all_feature_importances[featurelist_name][feature] = sum(all_feature_importances[featurelist_name][feature])
            else:
                raise Exception(f'`feature_importance_metric` provided ({feature_importance_metric}) not accepted.\nOptions: \'median\', \'mean\', or \'cumulative\'')

    # create 1d array of dimension N (x), and 2d array of dimension MxN (y) for stackplot
    df = pd.DataFrame(all_feature_importances).replace({np.nan: 0})
    if starting_featurelist != None: # rename starting_featurelist column to have the number of features
        df = df.rename(columns={starting_featurelist.name: f'{starting_featurelist.name} {len(starting_featurelist.features)}'})
    df = df/df.sum()
    cols = [(int(x.split(' ')[-1]), x) for x in list(df.columns)] # get a list of tuples where (# of features, column name)
    cols = sorted(cols)[::-1] # sorted descending by first object in tuple (featurelist size)
    x = []
    y = []
    for col in cols:
        x.append(str(col[0]))
        y.append(list(df[col[1]]))
    y = np.array(y)
    y = y.T
    labels = {}

    # unreadable list comprehension really means: get a dictionary with keys that are the old column names (features), and values with new column names (starting with underscore)
    # this is so that the underscored names are not shown in the legend.
    labels = [{x:'_' + str(x)} if i > _config.num_features_to_graph or i >= int(list(df.columns)[-1].split(' ')[-1]) else {x:x} for i, x in enumerate(df.iloc[:,-1].sort_values(ascending=False).index)]
    l = {}
    for label in labels:
        l.update(label)
    df = df.rename(index=l)
    _, ax = plt.subplots(figsize=(_config.fig_size[0], _config.fig_size[1]/2))
    ax.stackplot(x, y, labels=list(df.index))
    ax.legend(loc='upper left')
    return x, y