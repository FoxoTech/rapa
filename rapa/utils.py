from . import config

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

    :Parameters:
    ----------
        project: str
            Either a project id or a search term for project name

    :Returns: 
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
            raise Exception(f"No projects found with id/string of \'{project}\'")
        elif len(project_list) == 1: # found one project with search, good
            return project_list[0]
        else: # more than one project was found
            warn(f"Returning the first of multiple projects with \'{project}\': {project_list}")
            return project_list[0]
    

def get_best_model(project: dr.Project, 
                    featurelist_prefix: str = None, 
                    starred: bool = False, 
                    metric: str = None,
                    fold: str = 'crossValidation',
                    highest: bool = None) -> dr.Model:
    """Attempts to find the 'best' model in a datarobot by searching cross validation scores of all the
    models in a supplied project. # TODO make dictionary for minimize/maximize 

    CURRENTLY SUPPORTS METRICS WHERE HIGHER = BETTER

    .. warning:: 
        Actually finding the 'best' model takes more than averageing cross validation
        scores, and it is suggested that the 'best' model is decided and starred in DataRobot.
        (Make sure 'starred = True' if starring the 'best' model) 

    .. note::
        Some models may not have scores for the supplied fold because they were not run. These
        models are ignored by this function. Make sure all models of interest have scores for
        the fold being provided if those models should be considered.

    :Parameters:
    ----------
        project: datarobot.Project
            The project object that will be searched for the 'best' model

        featurelist_prefix: str, optional (default = None)
            The desired featurelist prefix used to search in for models using specific
            rapa featurelists

        starred: bool, optional (default = False)
            If True, return the starred model. If there are more than one starred models,
            then warn the user and return the 'best' one

        metric: str, optional (default = 'AUC' or 'RMSE') [classification and regression]
            What model metric to use when finding the 'best'
        
        fold: str, optional (default = 'crossValidation')
            The fold of data used in DataRobot. Options are as follows:
                ['validation', 
                'crossValidation', 
                'holdout', 
                'training', 
                'backtestingScores', 
                'backtesting']
        
        highest: bool, optional (default for classification = True, default for regression = False)
            Whether to take the highest value (highest = True), or the lowest
            value (highest = False). Change this when assumed switch is 
    
    :Returns:
    ----------
        datarobot.Model
            A datarobot model that is either the 'best', starred, or the 'best' of the starred models
            from the provided datarobot project
    """
    
    # if metric is missing, assume a metric
    if metric == None:
        if project.target_type == dr.TARGET_TYPE.BINARY or project.target_type == dr.TARGET_TYPE.MULTICLASS:
            # classification
            metric = 'AUC'
        elif project.target_type == dr.TARGET_TYPE.REGRESSION:
            # regression
            metric = 'RMSE'

    # if highest is missing, assume a direction
    if highest == None:
        if project.target_type == dr.TARGET_TYPE.BINARY or project.target_type == dr.TARGET_TYPE.MULTICLASS:
            # classification
            highest = True
        elif project.target_type == dr.TARGET_TYPE.REGRESSION:
            highest = False

    scores = []

    #### get scores

    # set featurelist_prefix to '' for ease of use in code
    if not starred: # the model(s) is/are not starred
        if featurelist_prefix == None: 
            featurelist_prefix = ''

        for model in project.get_models():
            current_model_score = model.metrics[metric][fold] # get the score for the metric and fold

            if model.featurelist_name.startswith(featurelist_prefix) and current_model_score: # if the model is scored in this fold, and it was created with the featurelist we are looking at
                scores.append((current_model_score, model)) # add the model score and model object to a list
    else: # the model(s) is/are starred
        if featurelist_prefix == None: 
            featurelist_prefix = ''

        for model in project.get_models():
            current_model_score = model.metrics[metric][fold] # get the score for the metric and fold

            if model.is_starred and model.featurelist_name.startswith(featurelist_prefix) and current_model_score: # if the model is scored in this fold, and it was created with the featurelist we are looking at
                scores.append((current_model_score, model)) # add the model score and model object to a list


    #### find the best scores

    # check that there are any models
    if len(scores) > 1: # multiple models
        return sorted(scores, key=lambda tup: tup[0], reverse=highest)[0][1] # sort by first item in the tuples
    elif len(scores) == 1: # one model
        return scores[0][1]
    else: # no models
        raise Exception(f"No models found. \n Parameters: project=`{project}`, metric=`{metric}`, fold=`{fold}`, featurelist_prefix=`{featurelist_prefix}`, starred=`{starred}`, highest=`{highest}`")


# alias for get_best_model
def get_starred_model(project: dr.Project, 
                    metric: str = None,
                    featurelist_prefix: str = None) -> dr.Model:
    """Alias for rapa.utils.get_best_model() but makes starred = True
    """
    return get_best_model(project, starred = True, metric = metric, featurelist_prefix = featurelist_prefix)


def initialize_dr_api(token_key: str = None, 
                    file_path: str = 'data/dr-tokens.pkl', 
                    endpoint: str = 'https://app.datarobot.com/api/v2'):
    """Initializes the DataRobot API with a pickled dictionary created by the user.

    .. warning:
        It is advised that the user keeps the pickled dictionary in an ignored 
        directory if using GitHub (put the file in the .gitignore)

    Accesses a file that should be a pickled dictionary. This dictionary has the API token
    as the value to the provided token_key. Ex: {token_key: 'API_TOKEN'}

    :Parameters:
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

    :Parameters:
    ----------
        featurelist: str
            Either a featurelist id or a search term for featurelist name
        
        project: datarobot.Project
            The project that is being searched for the featurelist

    :Returns:
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
            raise Exception(f"No featurelists were found with the id/name of \'{featurelist}\'")
        elif len(dr_featurelist) > 1: # if dr_featurelist has more than 1
            warn(f'More than one featurelist was found: \'{dr_featurelist}\', returning the first.')
            return dr_featurelist[0]
        else: # dr_Featurelist has 1
            return dr_featurelist[0]
    
    
def parsimony_performance_boxplot(project: dr.Project, 
                                featurelist_prefix: str = 'RAPA Reduced to',
                                starting_featurelist: str = None,
                                metric: str = None,
                                split: str = 'crossValidation',
                                featurelist_lengths: list = None):
    """Uses `seaborn`'s `boxplot` function to plot featurelist size vs performance
    for all models that use that featurelist prefix. There is a different boxplot for
    each featurelist length. # TODO warn about multiple prefixes, try to use new prefixes

    :Paremeters:
    ----------
        project: datarobot.Project
            Either a datarobot project, or a string of it's id or name

        featurelist_prefix: str, optional (default = 'RAPA Reduced to')
            The desired prefix for the featurelists that will be used for plotting parsimony performance. Each featurelist
            will start with the prefix, include a space, and then end with the number of features in that featurelist

        starting_featurelist: str, optional (default = None)
            The starting featurelist used for parsimony analysis. If None, only
            the featurelists with the desired prefix in `featurelist_prefix` will be plotted

        metric: str, optional (default = 'AUC' or 'RMSE') [classification and regression]
            The metric used for plotting accuracy of models

        split: str, optional (default = 'crossValidation')
            What split's performance to take from. 
            Can be: ['crossValidation', 'holdout'] TODO: i think it can be more, double check
        
        featurelist_lengths: list, optional (default = None)
            A list of featurelist lengths to plot

    :Returns:
    ----------
        None TODO: return plot?
    """
    # if `project` is a string, find the project
    if type(project) is str:
        project = find_project(project)

    # if metric is missing, assume a metric
    if metric == None:
        if project.target_type == dr.TARGET_TYPE.BINARY or project.target_type == dr.TARGET_TYPE.MULTICLASS:
            # classification
            metric = 'AUC'
        elif project.target_type == dr.TARGET_TYPE.REGRESSION:
            # regression
            metric = 'RMSE'


    datarobot_project_models = project.get_models() # get all the models in the provided project

    if starting_featurelist:
        if type(starting_featurelist) == str:
            starting_featurelist = get_featurelist(starting_featurelist, project)
        num_starting_featurelist_features = len(starting_featurelist.features)


    featurelist_performances = defaultdict(list)
    for model in datarobot_project_models: # for every model, if the model has the prefix, then add it's performance
        if model.featurelist_name != None and featurelist_prefix in model.featurelist_name:
            num_features = int(model.featurelist_name.split(' ')[-1].strip('()')) # parse the number of features from the featurelist name
            if model.metrics[metric][split] != None: # if there is no feature impact for the model/split, don't add the metric
                if featurelist_lengths and num_features in featurelist_lengths:
                    featurelist_performances[num_features].append(model.metrics[metric][split])
                elif not featurelist_lengths:
                    featurelist_performances[num_features].append(model.metrics[metric][split])
        elif starting_featurelist and model.featurelist_id == starting_featurelist.id: # starting featurelist
            if model.metrics[metric][split] != None: # if there is no feature impact for the model/split, don't add the metric
                if featurelist_lengths and num_starting_featurelist_features in featurelist_lengths:
                    featurelist_performances[num_starting_featurelist_features].append(model.metrics[metric][split])
                elif not featurelist_lengths:
                    featurelist_performances[num_starting_featurelist_features].append(model.metrics[metric][split])
    
    # Add Nones so that the arrays are the same length
    last = 0
    for key in featurelist_performances:
        m = max(last, len(featurelist_performances[key]))
        last = m
    for key in featurelist_performances:
        temp_len = len(featurelist_performances[key])
        for _ in range(m-temp_len):
            featurelist_performances[key].append(None)
    
    featurelist_performances_df = pd.DataFrame(featurelist_performances)[sorted(featurelist_performances.keys())[::-1]]
    
    with plt.style.context('tableau-colorblind10'):
        plt.ylabel(f'{split} {metric}')
        plt.xlabel('Number of Features')
        plt.title(f'{project.project_name} - {featurelist_prefix}\nParsimonious Model Performance')
        sb.boxplot(data=featurelist_performances_df)
    return featurelist_performances_df
    
def feature_performance_stackplot(project: dr.Project, 
                                featurelist_prefix: str = 'RAPA Reduced to',
                                starting_featurelist: str = None,
                                feature_impact_metric: str = 'median',
                                metric: str = None,
                                vlines: bool = False):
    """Utilizes `matplotlib.pyplot.stackplot` to show feature performance during 
    parsimony analysis.

    :Parameters:
    ----------
        project: datarobot.Project
            Either a datarobot project, or a string of it's id or name

        featurelist_prefix: str, optional (default = 'RAPA Reduced to')
            The desired prefix for the featurelists that will be used for plotting feature performance. Each featurelist
            will start with the prefix, include a space, and then end with the number of features in that featurelist

        starting_featurelist: str, optional (default = None)
            The starting featurelist used for parsimony analysis. If None, only
            the featurelists with the desired prefix in `featurelist_prefix` will be plotted
        
        feature_impact_metric: str, optional (default = mean)
            Which metric to use when finding the  most representative feature importance of all models in the featurelist

            Options:
                * median
                * mean
                * cumulative

        metric: str, optional (default = 'AUC' or 'RMSE') [classification and regression]
            Which metric to use when finding feature importance of each model
        
        vlines: bool, optional (default = False)
            Whether to add vertical lines at the featurelist lengths or not, False by default

    :Returns:
    ----------
        None TODO: return plot?
    """
    # if `project` is a string, find the project
    if type(project) is str:
        project = find_project(project)

    # if metric is missing, assume a metric
    if metric == None:
        if project.target_type == dr.TARGET_TYPE.BINARY or project.target_type == dr.TARGET_TYPE.MULTICLASS:
            # classification
            metric = 'AUC'
        elif project.target_type == dr.TARGET_TYPE.REGRESSION:
            # regression
            metric = 'RMSE'
    
    if starting_featurelist:
        if type(starting_featurelist) == str:
            starting_featurelist = get_featurelist(starting_featurelist, project)
    
    datarobot_project_models = project.get_models() # get all the models in the provided project

    if starting_featurelist != None: # have the starting featurelist as well
        all_feature_importances = {}
        for model in datarobot_project_models:
            if model.featurelist_name != None and (model.featurelist_name.startswith(featurelist_prefix) or model.featurelist_id == starting_featurelist.id): # if the model uses the starting featurelist/featurelist prefix
                if model.metrics[metric]['crossValidation'] != None:
                    if model.featurelist_name in all_feature_importances.keys():
                        for x in model.get_or_request_feature_impact():
                            if x['featureName'] in all_feature_importances[model.featurelist_name].keys():
                                all_feature_importances[model.featurelist_name][x['featureName']].append(x['impactNormalized'])
                            else:
                                all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
                    else:
                        all_feature_importances[model.featurelist_name] = {} 
                        for x in model.get_or_request_feature_impact():
                            all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
    else: # same as if, but without starting featurelist 
        all_feature_importances = {}
        for model in datarobot_project_models:
            if model.featurelist_name.startswith(featurelist_prefix): # if the model's featurelist starts with the featurelist prefix
                if model.metrics[metric]['crossValidation'] != None:
                    if model.featurelist_name in all_feature_importances.keys():
                        for x in model.get_or_request_feature_impact():
                            if x['featureName'] in all_feature_importances[model.featurelist_name].keys():
                                all_feature_importances[model.featurelist_name][x['featureName']].append(x['impactNormalized'])
                            else:
                                all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]
                    else:
                        all_feature_importances[model.featurelist_name] = {} 
                        for x in model.get_or_request_feature_impact():
                            all_feature_importances[model.featurelist_name][x['featureName']] = [x['impactNormalized']]

    for featurelist_name in all_feature_importances.keys():
        for feature in all_feature_importances[featurelist_name].keys():
            if feature_impact_metric.lower() == 'median':
                all_feature_importances[featurelist_name][feature] = median(all_feature_importances[featurelist_name][feature])
            elif feature_impact_metric.lower() == 'mean':
                all_feature_importances[featurelist_name][feature] = mean(all_feature_importances[featurelist_name][feature])
            elif feature_impact_metric.lower() == 'cumulative':
                all_feature_importances[featurelist_name][feature] = sum(all_feature_importances[featurelist_name][feature])
            else:
                raise Exception(f'`feature_impact_metric` provided ({feature_impact_metric}) not accepted.\nOptions: \'median\', \'mean\', or \'cumulative\'')

    # create 1d array of dimension N (x), and 2d array of dimension MxN (y) for stackplot
    df = pd.DataFrame(all_feature_importances).replace({np.nan: 0})
    if starting_featurelist != None: # rename starting_featurelist column to have the number of features
        df = df.rename(columns={starting_featurelist.name: f'{starting_featurelist.name} {len(starting_featurelist.features)}'})
    df = df/df.sum()
    cols = [(int(x.split(' ')[-1].strip('()')), x) for x in list(df.columns)] # get a list of tuples where (# of features, column name)
    cols = sorted(cols)[::-1] # sorted descending by first object in tuple (featurelist size)
    x = []
    y = []
    for col in cols:
        x.append(str(col[0]))
        y.append(list(df[col[1]]))
    y = np.array(y)
    y = y.T
    
    featurelist_lengths = sorted([int(x.split(' ')[-1].strip('()')) for x in df.columns])[::-1] # descending list of featurelist lengths

    len_smallest_featurelist = min(featurelist_lengths)
    smallest_featurelist = featurelist_prefix + ' (' + str(len_smallest_featurelist) + ')'

    # if the length of the smallest featurelist is less than the number of features to label
    # get a featurelist that has a length higher than the minimum features to label for labeling purposes
    if len_smallest_featurelist < config.MIN_FEATURES_TO_LABEL:
        len_smallest_featurelist = config.MIN_FEATURES_TO_LABEL
        last_length = np.inf
        for length in featurelist_lengths:
            if length < len_smallest_featurelist:
                break
            last_length = length
            smallest_featurelist = featurelist_prefix + ' (' + str(last_length) + ')'

    # unreadable list comprehension really means: get a dictionary with keys that are the old column names (features), and values with new column names (starting with underscore)
    # at least show config.MIN_FEATURES_TO_GRAPH
    # this is so that the underscored names are not shown in the legend.
    labels = [{x:'_' + str(x)} if i > config.MAX_FEATURES_TO_LABEL or i >= len_smallest_featurelist else {x:x} for i, x in enumerate(df.loc[:,smallest_featurelist].sort_values(ascending=False).index)]
    l = {}
    for label in labels:
        l.update(label)

    df = df.rename(index=l)
    _, ax = plt.subplots(figsize=(config.FIG_SIZE[0], config.FIG_SIZE[1]/2))
    plt.xlabel('Feature List Length')
    plt.ylabel('Normalized Feature Impact\n(Normalized Impact Normalized)')
    plt.title(f'{project.project_name} - {featurelist_prefix}\nFeature Impact Stackplot')
    if vlines:
        plt.vlines([z for z in range(1,len(x)-1)], ymin=0, ymax=1, linestyles='dashed')
    ax.stackplot(x, y, labels=list(df.index), colors=plt.cm.tab20.colors)
    ax.legend(loc='upper left')
    return None

