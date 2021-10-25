from logging import debug
from re import VERBOSE
import warnings
from datarobot.models import feature, featurelist
from . import utils
from . import _config

import time

try: # check if in jupyter notebook
    get_ipython
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

from sklearn.feature_selection import f_regression, f_classif
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_array

from typing import List
from typing import Callable
from typing import Tuple
from typing import Union

import pandas as pd
import numpy as np
from statistics import mean
from statistics import stdev
from math import ceil

import matplotlib.pyplot as plt

import datarobot as dr

class RAPABase():
    """
        The base of regression and classification RAPA analysis
    """

    POSSIBLE_TARGET_TYPES = [x for x in dir(dr.enums.TARGET_TYPE) if not x.startswith('__')] # List of DR TARGET_TYPES

    """_classification = None # Set by child classes
    target_type = None # Set at initialization
    # target_name = None # Set with 'create_submittable_dataframe()'
    project = None # Set at initialization or with 'perform_parsimony()'"""

    def __init__(self, project: Union[dr.Project, str] = None):
        if self.__class__.__name__ == "RAPABase":
            raise RuntimeError("Do not instantiate the RAPABase class directly; use RAPAClassif or RAPARegress")
        self._classification = None
        self.target_type = None
        self.project = None

    @staticmethod
    def _wait_for_jobs(project: dr.Project, progress_bar: bool = True, sleep_time: int = 5):
        """Gets all the jobs for a project, and if there are more than 0 current jobs, 
        sleeps for 5 seconds and checks again.

        ## Parameters
        ----------
        project: datarobot.Project
            The datarobot.Project that will be probed for current jobs
        
        progress_bar: bool, optional (default = True)
            If True, a print statement and a progress bar will appear TODO: a print statement and a progress bar will appear

        sleep_time: int, optional (default = 5)
            The time to sleep between datarobot.Project.get_all_jobs() 
            (avoid sending too many api requests) TODO: warning or check for max api requests
        """
        # first time, no end='\r' test
        if len(project.get_all_jobs()) > 0:
            if progress_bar:
                tqdm.write(f'\rThere are {len(project.get_all_jobs())} jobs remaining...'.ljust(33), end='') 
            time.sleep(sleep_time)
        while len(project.get_all_jobs()) > 0:
            if progress_bar: # PROGRESS BAR
                tqdm.write(f'\rThere are {len(project.get_all_jobs())} jobs remaining...'.ljust(33), end='') 
            time.sleep(sleep_time)
        return None 

    @staticmethod
    def _check_lives(lives: int, 
                    project: dr.Project, 
                    previous_best_model: dr.Model,
                    featurelist_prefix: str = None, 
                    starred: bool = False, 
                    metric: str = 'AUC',
                    verbose: bool = True) -> Tuple[int, dr.Model]:
        """Finds the 'best' model of a project/featurelist of a project and returns the new
        `lives` count (decreased by 1 if the model doesn't change) and the 'best' model

        Uses `rapa.utils.get_best_model` to find the current best model, and decides
        if the model has changed by equating `datarobot.Model.id`. Returns a tuple with
        the number of 'lives' left in the first position, and the current 'best' model
        in the second position.

        ## Parameters
        ----------
        lives: int
            The current number of 'lives' remaining in parsimony analysis

        project: datarobot.Project
            The datarobot.Project parsimony analysis is being performed in
        
        previous_best_model: datarobot.Model
            The previously 'best' model in the datarobot.Project before
            a round of parsimony analysis

        featurelist_prefix: str, optional (default = None)
            The desired prefix for the featurelists that will be used for searching
            for the 'best' model. If None, will search the entire datarobot.Project
        
        starred: bool, optional (default = False)
            If True, searching the project's starred models. If False, searches
            all of the project's models
        
        metric: str, optional (default = 'AUC')
            What model cross validation metric to use when averaging scores to
            find the 'best' model
        
        verbose: bool, optional (default = True)
            If True, prints previous and current best model information when 
            before returning

        ## Returns
        ----------
        Tuple(int, datarobot.Model)
            A tuple with the new `lives` in the first position, and the new
            'best' model after one round of persimony analysis
        """

        # check for the best model (supplied metric of cv)
        current_best_model = utils.get_best_model(project, metric=metric, featurelist_prefix=featurelist_prefix, starred=starred)
        if current_best_model.id == previous_best_model.id:
            lives -= 1
            current_best_model_score = mean(current_best_model.get_cross_validation_scores()['cvScores'][metric].values())
            last_best_model_score = mean(previous_best_model.get_cross_validation_scores()['cvScores'][metric].values())
            if verbose:
                tqdm.write(f'Current model performance: \'{current_best_model_score}\'. Last best model performance: \'{last_best_model_score}\'\nNo change in the best model, so a life was lost.\nLives remaining: \'{lives}\'')
            else:
                tqdm.write(f'Lives left: \'{lives}\'')
        return (lives, current_best_model)
    

    def create_submittable_dataframe(self, 
                                    input_data_df: pd.DataFrame, 
                                    target_name: str, 
                                    n_features: int = 19990,
                                    n_splits: int = 6, 
                                    filter_function: Callable[[pd.DataFrame, np.ndarray], List[np.ndarray]] = None,
                                    random_state: int = None) -> pd.DataFrame: #TODO: change return type
        """Prepares the input data for submission as either a regression or classification problem on DataRobot.

        Creates pre-determined k-fold cross-validation splits and filters the feature
        set down to a size that DataRobot can receive as input, if necessary. TODO: private function submit_datarobot_project explanation

        ## Parameters
        ----------
        input_data_df: pandas.DataFrame
            pandas DataFrame containing the feature set and prediction target.

        target_name: str
            Name of the prediction target column in `input_data_df`.

        n_features: int, optional (default: 19990)
            The number of features to reduce the feature set in `input_data_df`
            down to. DataRobot's maximum feature set size is 20,000.
            If `n_features` has the same number of features as the `input_data_df`,
            NaN values

        n_splits: int, optional (default: 6)
            The number of cross-validation splits to create. One of the splits
            will be retained as a holdout split, so by default this function
            sets up the dataset for 5-fold cross-validation with a holdout.

        filter_function: callable, optional (default: None)
            The function used to calculate the importance of each feature in
            the initial filtering step that reduces the feature set down to
            `max_features`.

            This filter function must take a feature matrix as the first input
            and the target array as the second input, then return two separate
            arrays containing the feature importance of each feature and the
            P-value for that correlation, in that order.

            When None, the filter function is determined by child class.
            If an instance of `RAPAClassif()`, sklearn.feature_selection.f_classif is used.
            If `RAPARegress()`, sklearn.feature_selection.f_regression is used.
            See scikit-learn's f_classif function for an example:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html

        random_state: int, optional (default: None)
            The random number generator seed for RAPA. Use this parameter to make sure
            that RAPA will give you the same results each time you run it on the
            same input data set with that seed.

        Returns
        ----------
        pandas.DataFrame
            DataFrame holds original values from the input Dataframe, but with 
            pre-determined k-fold cross-validation splits, and was 
            filtered down to 'max_features' size using the 'filter_function'
        """
        #TODO: make private function? 
        # Check dataframe has 'target_name' columns
        if target_name not in input_data_df.columns:
            raise KeyError(f'{target_name} is not a column in the input DataFrame')

        # Check that the dataframe can be copied and remove target_name column
        input_data_df = input_data_df.copy()
        only_features_df = input_data_df.drop(columns=[target_name])

        # Check if the requested number of features is equal to the number of features provided
        # If True, skip feature filtering and allow NaNs
        if n_features == only_features_df.shape[1]:
            feature_filter = False
        else:
            feature_filter = True

        # Set target_type, kfold_type, and filter_function based on type of classification/regression problem
        if self._classification:
            # Check if binary or multi classification problem
            if len(np.unique(input_data_df[target_name].values)) == 2:
                self.target_type = dr.enums.TARGET_TYPE.BINARY
            else:
                self.target_type = dr.enums.TARGET_TYPE.MULTICLASS
            kfold_type = StratifiedKFold
            filter_function = f_classif
        else:
            # Check array for infinite values/NaNs
            if feature_filter:
                check_array(input_data_df)
            kfold_type = KFold
            self.target_type = dr.enums.TARGET_TYPE.REGRESSION
            filter_function = f_regression

        # Create 'partition' column and set all values to 'train'
        input_data_df['partition'] = 'train'
        train_feature_importances = []

        # Make cross validation folds
        fold_name_prefix = 'CV Fold'
        for fold_num, (_, fold_indices) in enumerate(kfold_type(n_splits=n_splits, random_state=random_state, shuffle=True).split(only_features_df.values,
                                                    input_data_df[target_name].values)):

            # Replace the values in the partition column with their true CV fold, removing all 'train' entries
            input_data_df.iloc[fold_indices, input_data_df.columns.get_loc('partition')] = f'{fold_name_prefix} {fold_num}'

            # Fold 0 is the holdout set, so don't calculate feature importances using that fold
            if feature_filter and fold_num > 0:
                feature_importances, _ = filter_function(only_features_df.iloc[fold_indices].values, input_data_df[target_name].iloc[fold_indices].values)
                train_feature_importances.append(feature_importances)

        if feature_filter:
            # We calculate the overall feature importance scores by averaging the feature importance scores across all of the training folds
            avg_train_feature_importances = np.mean(train_feature_importances, axis=0)

            # Change parition 0 name to 'Holdout'
            input_data_df.loc[input_data_df['partition'] == f'{fold_name_prefix} 0', 'partition'] = 'Holdout'

            # Gets the top `n_features` correlated features as a list
            most_correlated_features = only_features_df.columns.values[np.argsort(avg_train_feature_importances)[::-1][:n_features]].tolist()

            # put target_name, partition, and most correlated features columns in dr_upload_df
            datarobot_upload_df = input_data_df[[target_name, 'partition'] + most_correlated_features]
        
        else:
            # put target_name, partition, and most correlated features columns in dr_upload_df
            datarobot_upload_df = input_data_df[[target_name, 'partition'] + only_features_df.columns.values.tolist()]

        return datarobot_upload_df


    def submit_datarobot_project(self, 
                                input_data_df: pd.DataFrame, 
                                target_name: str, 
                                project_name: str, 
                                target_type: str = None, 
                                worker_count: int = -1, 
                                metric: str = None,
                                mode: str = dr.AUTOPILOT_MODE.FULL_AUTO,
                                random_state: int = None) -> dr.Project: #TODO check input df for partition, target_name (data-robotified df), logger.warning
        """Submits the input data to DataRobot as a new modeling project.

        It is suggested to prepare the `input_data_df` using the
        'create_submittable_dataframe' function first with an instance of
        either RAPAClassif or RAPARegress.

        Parameters
        ----------
        input_data_df: pandas.DataFrame
            pandas DataFrame containing the feature set and prediction target.

        target_name: str
            Name of the prediction target column in `input_data_df`.

        project_name: str
            Name of the project in DataRobot.

        target_type: str (enum)
            Indicator to DataRobot of whether the new modeling project should be
            a binary classification, multiclass classification, or regression project.

            Options:
                datarobot.TARGET_TYPE.BINARY
                datarobot.TARGET_TYPE.REGRESSION
                datarobot.TARGET_TYPE.MULTICLASS

        worker_count: int, optional (default: -1)
            The number of worker engines to assign to the DataRobot project.
            By default, -1 tells DataRobot to use all available worker engines.
        
        metric: str, optional (default: None)
            Name of the metric to use for evaluating models. You can query the metrics 
            available for the target by way of Project.get_metrics. If none is specified, 
            then the default recommended by DataRobot is used.

        mode: str (enum), optional (default: datarobot.AUTOPILOT_MODE.FULL_AUTO)
            The modeling mode to start the DataRobot project in.

            Options:
                datarobot.AUTOPILOT_MODE.FULL_AUTO
                datarobot.AUTOPILOT_MODE.QUICK
                datarobot.AUTOPILOT_MODE.MANUAL
                datarobot.AUTOPILOT_MODE.COMPREHENSIVE: Runs all blueprints in
                the repository (warning: this may be extremely slow).

        random_state: int, optional (default: None)
            The random number generator seed for DataRobot. Use this parameter to make sure
            that DataRobot will give you the same results each time you run it on the
            same input data set with that seed.

        """
        # TODO: provide an option for columns to disregard

        # Check for a target_type
        if target_type == None or target_type not in self.POSSIBLE_TARGET_TYPES:
            target_type = self.target_type
            if target_type == None:
                raise Exception(f'No target type.')

        project = dr.Project.create(sourcedata=input_data_df, project_name=project_name)

        project.set_target(target=target_name, target_type=target_type,
                        worker_count=worker_count, mode=mode, metric=metric,
                        advanced_options=dr.AdvancedOptions(seed=random_state, accuracy_optimized_mb=False,
                                                            prepare_model_for_deployment=False, blend_best_models=False),
                        partitioning_method=dr.UserCV(user_partition_col='partition', cv_holdout_level='Holdout'))

        return project


    def perform_parsimony(self, feature_range: List[Union[int, float]], 
                        project: Union[dr.Project, str] = None,
                        starting_featurelist_name: str = 'Informative Features',
                        featurelist_prefix: str = 'RAPA Reduced to', 
                        mode: str = dr.AUTOPILOT_MODE.FULL_AUTO,
                        lives: int = None,
                        cv_average_mean_error_limit: float = None,
                        feature_importance_metric: str = 'median',
                        progress_bar: bool = True, 
                        to_graph: List[str] = None, 
                        metric: str = None,
                        verbose: bool = True):
        """Performs parsimony analysis by repetatively extracting feature importance from 
        DataRobot models and creating new models with reduced features (smaller feature lists). # TODO take a look at featurelist_prefix for running multiple RAPA

        Parameters:
        ----------
        feature_range: list[int] | list[float]
            Either a list containing integers representing desired featurelist lengths,
            or a list containing floats representing desired featurelist percentages (of the original featurelist size)

        project: datarobot.Project | str, optional (default = None)
            Either a datarobot project, or a string of it's id or name. If None,
            uses the project that was provided to create the rapa class
        
        starting_featurelist: str, optional (default = 'Informative Features')
            The name or id of the featurelist that rapa will start pasimony analysis with

        featurelist_prefix: str, optional (default = 'RAPA Reduced to')
            The desired prefix for the featurelists that rapa creates in datarobot. Each featurelist
            will start with the prefix, include a space, and then end with the number of features in that featurelist

        mode: str (enum), optional (default: datarobot.AUTOPILOT_MODE.FULL_AUTO)
            The modeling mode to start the DataRobot project in.
            Options:
                datarobot.AUTOPILOT_MODE.FULL_AUTO
                datarobot.AUTOPILOT_MODE.QUICK
                datarobot.AUTOPILOT_MODE.MANUAL
                datarobot.AUTOPILOT_MODE.COMPREHENSIVE: Runs all blueprints in
                the repository (warning: this may be extremely slow).
        
        lives: int, optional (default = None)
            The number of times allowed for reducing the featurelist and obtaining a worse model. By default,
            'lives' are off, and the entire 'feature_range' will be ran, but if supplied a number >= 0, then 
            that is the number of 'lives' there are. 

            Ex: lives = 0, feature_range = [100, 90, 80, 50]
            RAPA finds that after making all the models for the length 80 featurelist, the 'best' model was created with the length
            90 featurelist, so it stops and doesn't make a featurelist of length 50.

            Similar to datarobot's Feature Importance Rank Ensembling for advanced feature selection (FIRE) package's 'lifes' 
            https://www.datarobot.com/blog/using-feature-importance-rank-ensembling-fire-for-advanced-feature-selection/ 
        
        cv_mean_error_limit: float, optional (default = None)
            The limit of cross validation mean error to help avoid overfitting. By default, the limit is off, 
            and the each 'feature_range' will be ran. Limit exists only if supplied a number >= 0.0

            Ex: 'feature_range' = 2.5, feature_range = [100, 90, 80, 50]
            RAPA finds that the average AUC for each CV fold is [.8, .6, .9, .5] respectfully,
            the mean of these is 0.7. The average error is += 0.15. If 0.15 >= cv_mean_error_limit,
            the training stops.
        
        feature_importance_metric: str, optional (default = 'median')
            How RAPA will decide each feature's importance over every model in a feature list
                Options: 'median', 'mean', or 'cumulative'

        progress_bar: bool, optional (default = True)
            If True, a simple progres bar displaying complete and incomplete featurelists. 
            If False, provides updates in stdout Ex: current worker count, current featurelist, etc.

        to_graph: List[str], optional (default = None)
            A list of keys choosing which graphs to produce. Possible Keys:
                'models': `seaborn` boxplot with model performances with provided metric
                'feature_performance': `matplotlib.pyplot` stackplot of feature performances

        metric: str, optional (default = None)
            The metric used for scoring models, when finding the 'best' model, and when
            plotting model performance

            When None, the metric is determined by what class inherits from base. For instance,
            a `RAPAClassif` instance's default is 'AUC', and `RAPARegress` is 'R Squared'
        """ 
        # TODO: return a dictionary of values? {"time_taken": 2123, "cv_mean_error": list[floats], ""}

        # check project
        if project == None:
            project = self.project
            if project == None:
                raise Exception('No provided datarobot.Project()')

        # check scoring metric TODO: support more scoring metrics
        if metric == None:
            if self._classification: # classification
                metric = 'AUC'
            else: # regression
                metric = 'R Squared'
    
        # check if project is a string, and if it is, find it
        if type(project) == str:
            project = utils.find_project(project)
            if project == None:
                raise Exception(f'Could not find the project.')
        
        # get starting featurelist
        try:
            starting_featurelist = utils.get_featurelist(starting_featurelist_name, project)
        except: # TODO: flesh out exceptions
            tqdm.write("Something went wrong getting the starting featurelist...")

        # check feature_range size
        if len(feature_range) == 0:
            raise Exception('The provided feature_range is empty.')

        # feature_range logic for sizes (ints) / ratios (floats)
        if np.array(feature_range).dtype.kind in np.typecodes['AllInteger']: 
            feature_range_check = [x for x in feature_range if x < len(starting_featurelist.features)-2 and x > 0] # -2 because of target feature and partitions TODO: CHECK FOR FEATURE/PARTITIONS INSTEAD OF JUST SUBTRACTING 2
            if len(feature_range_check) != len(feature_range): # check to see if values are < 0 or > the length of the original featurelist
                raise Exception('The provided feature_range integer values have to be: 0 < feature_range < original featurelist length')
        elif np.array(feature_range).dtype.kind in np.typecodes['AllFloat']:
            feature_range_check = [x for x in feature_range if x > 0 and x < 1]
            if len(feature_range_check) != len(feature_range):
                raise Exception(f'The provided feature_range ratio values have to be: 0 < feature_range < 1')
            # convert ratios to featurelist sizes
            original_featurelist_size = len(starting_featurelist.features)-2 # -2 because of target feature and partitions
            feature_range = [ceil(original_featurelist_size * feature_pct) for feature_pct in feature_range] # multiply by feature_pct and take ceil
            feature_range = pd.Series(feature_range).drop_duplicates() # drop duplicates
            feature_range = list(feature_range[feature_range < original_featurelist_size]) # take all values that less than the original featurelist size
            feature_range.sort(reverse=True) # sort descending
        else:
            raise TypeError('Provided \'feature_range\' is not all Int or all Float.')

        # ----------------------------------------------------------------------------------
        if _config.debug_statements:
            tqdm.write(f'{project=} {starting_featurelist=} {metric=} {feature_range=}')
        # ----------------------------------------------------------------------------------

        # get the models from starting featurelist
        datarobot_project_models = project.get_models()

        for model in datarobot_project_models: # for each model
            if model.featurelist_id == starting_featurelist.id: # if the model uses the starting featurelist, request the feature impact
                if model.metrics[metric]['crossValidation'] != None:
                    try:
                        model.request_feature_impact()
                    except dr.errors.JobAlreadyRequested:
                        continue
        
        # waiting for DataRobot projects TODO tqdm/multithreading/print tqdm function for printing things w/o messing things up 
        # TODO check to see if datarobot made a function
        # TODO request_featureimpact returns a job indicator?
        tqdm.write("waiting")
        self._wait_for_jobs(project)
        tqdm.write("done waiting")

        # get feature impact/importances of original featurelist
        all_feature_importances = []
        for model in datarobot_project_models:
            if model.featurelist_id == starting_featurelist.id: # if the model uses the starting featurelist, request the feature impact
                if model.metrics[metric]['crossValidation'] != None:
                    all_feature_importances.extend(model.get_feature_impact())
        
        # sort by features by feature importance statistic TODO: better way to do this, dictionary w/ [median:pd.DataFrame.median()] ?
        stat_feature_importances = pd.DataFrame(all_feature_importances).groupby('featureName')['impactNormalized']
        if feature_importance_metric.lower() == 'median':
            stat_feature_importances = stat_feature_importances.median().sort_values(ascending=False)
        elif feature_importance_metric.lower() == 'mean':
            stat_feature_importances = stat_feature_importances.mean().sort_values(ascending=False)
        elif feature_importance_metric.lower() == 'cumulative':
            stat_feature_importances = stat_feature_importances.sum().sort_values(ascending=False)
        else: # feature_importance_metric isn't one of the provided statistics
            raise ValueError(f'The provided feature_importance_metric:{feature_importance_metric} is not one of the provided:{_config.feature_importance_metrics}')

        # retain feature performance for each round, and plot stacked bar plot of original feature performances
        # TODO decide on coloring scheme
        if 'feature_performance' in to_graph:
            feature_performances = pd.DataFrame(stat_feature_importances.rename(len(stat_feature_importances)))
            warnings.filterwarnings('ignore', message='The handle <BarContainer object of 1 artists>')
            feature_performances.iloc[:_config.num_features_to_graph].T.set_axis(list(feature_performances.iloc[:_config.num_features_to_graph].T.columns), axis=1, inplace=False).plot(kind='bar',
                                                                                                                                            stacked=False,
                                                                                                                                            figsize=(_config.fig_size[0]/2, _config.fig_size[1]),
                                                                                                                                            title=f'Top {_config.num_features_to_graph} Impact Normalized Feature Performances\nFeaturelist: {starting_featurelist_name}',
                                                                                                                                            xlabel='Original Featurelist Length',
                                                                                                                                            ylabel='Normalized Impact of Features',
                                                                                                                                            colormap='tab20')
            warnings.filterwarnings('default')
            plt.show()
            plt.close()

        # waiting for DataRobot projects
        self._wait_for_jobs(project)
        
        # get the best performing model of this iteration
        previous_best_model = utils.get_best_model(project, metric=metric, featurelist_prefix=starting_featurelist_name)

        
        tqdm.write(f'Project: {project.project_name} | Featurelist Prefix: {featurelist_prefix} | Feature Range: {feature_range}')
        if verbose:
            tqdm.write(f'Feature Importance Metric: {feature_importance_metric} | Model Performance Metric: {metric}')
            if lives:
                tqdm.write(f'Lives: {lives}')
            if cv_average_mean_error_limit:
                tqdm.write(f'CV Mean Error Limit: {cv_average_mean_error_limit}')

        # perform parsimony
        for featurelist_length in tqdm(feature_range, disable= not progress_bar):
            # ---------------------------------------------------------------------------------------------
            if verbose:
                tqdm.write(f"---------- {featurelist_prefix} {featurelist_length} ----------")
            # ---------------------------------------------------------------------------------------------
            try:
                # get shortened featurelist
                desired_reduced_featurelist_size = featurelist_length
                reduced_features = stat_feature_importances.head(desired_reduced_featurelist_size).index.values.tolist()

                # ----- create new featurelist in datarobot -----
                new_featurelist_name = '{} {}'.format(featurelist_prefix, len(reduced_features)) # TODO have some suffix added, move try except
                reduced_featurelist = project.create_featurelist(name=new_featurelist_name, features=reduced_features)
                
                # ----- submit new featurelist and create models -----
                tqdm.write('starting autopilot...')
                project.start_autopilot(featurelist_id=reduced_featurelist.id, mode=mode, blend_best_models=False, prepare_model_for_deployment=False)
                tqdm.write('waiting for autopilot...')
                project.wait_for_autopilot(verbosity=dr.VERBOSITY_LEVEL.SILENT)

                datarobot_project_models = project.get_models()
                for model in datarobot_project_models:
                    if model.featurelist_id == reduced_featurelist.id and model.metrics[metric]['crossValidation'] != None:
                        try:
                            model.request_feature_impact()
                        except dr.errors.JobAlreadyRequested:
                            pass

                # API note: Is there a project-level wait function for all jobs, regardless of AutoPilot status?
                tqdm.write("waiting")
                self._wait_for_jobs(project)
                tqdm.write("done waiting")

                while(len(all_feature_importances) == 0):
                    all_feature_importances = []
                    for model in datarobot_project_models:
                        if model.featurelist_id == reduced_featurelist.id and model.metrics[metric]['crossValidation'] != None:
                            all_feature_importances.extend(model.get_feature_impact())
                    time.sleep(5)

                # sort by features by feature importance statistic TODO: better way to do this, dictionary w/ [median:pd.DataFrame.median()] ?
                stat_feature_importances = pd.DataFrame(all_feature_importances).groupby('featureName')['impactNormalized']
                if feature_importance_metric.lower() == 'median':
                    stat_feature_importances = stat_feature_importances.median().sort_values(ascending=False)
                elif feature_importance_metric.lower() == 'mean':
                    stat_feature_importances = stat_feature_importances.mean().sort_values(ascending=False)
                elif feature_importance_metric.lower() == 'cumulative':
                    stat_feature_importances = stat_feature_importances.sum().sort_values(ascending=False)
                
                # ----- Graphing Feature Performance -----
                if 'feature_performance' in to_graph:
                    utils.feature_performance_stackplot(project=project,
                                                        featurelist_prefix=featurelist_prefix,
                                                        starting_featurelist=starting_featurelist,
                                                        feature_importance_metric=feature_importance_metric,
                                                        metric=metric)
                    plt.show()
                    plt.close()
                

                # ----- LIVES -----
                # check for the best model (supplied metric of cv)
                if lives != None:
                    if featurelist_length == feature_range[0]: # for the first time, check model scores instead of making sure the model id doesn't change (what _check_lives does)
                        current_best_model = utils.get_best_model(project, metric=metric, featurelist_prefix=featurelist_prefix)
                        previous_best_model_score = mean(previous_best_model.get_cross_validation_scores()['cvScores'][metric].values())
                        current_best_model_score = mean(current_best_model.get_cross_validation_scores()['cvScores'][metric].values())
                        if previous_best_model_score > current_best_model_score:
                            lives -= 1
                            tqdm.write(f'Current model performance: \'{current_best_model_score}\'. Last best model performance: \'{previous_best_model_score}\'\nNo change in the best model, so a life was lost.\nLives remaining: \'{lives}\'')
                            previous_best_model = current_best_model
                    else: # get the best model and check their id
                        lives, previous_best_model = self._check_lives(lives=lives, 
                                                                        project=project, 
                                                                        previous_best_model=previous_best_model, 
                                                                        featurelist_prefix=featurelist_prefix, 
                                                                        metric=metric,
                                                                        verbose=True)
                    if lives < 0:
                        current_best_model_score = mean(previous_best_model.get_cross_validation_scores()['cvScores'][metric].values())
                        tqdm.write(f'Ran out of lives.\nBest model: \'{previous_best_model}\'\nAccuracy ({metric}):\'{current_best_model_score}\'')
                        break

                    # ----------------------------------------------------------------------------------
                    if verbose:
                        tqdm.write(f'Lives left: {lives} | Previous Model Best Score: {previous_best_model_score} | Current Best Model Score: {current_best_model_score=}')
                    # ----------------------------------------------------------------------------------
                    
                # ----- cv_average_mean_error_limit -----
                # for the current featurelist, check the cv metric for all models and get the standard deviation of the metric among the cv folds for each model.
                # Then, take the average of those standard deviation values and check that it is below the cv_average_mean_error_limit
                if cv_average_mean_error_limit != None:
                    cv_metrics_dict = {}
                    for model in datarobot_project_models:
                        if model.featurelist_id == reduced_featurelist.id and model.metrics[metric]['crossValidation'] != None:
                            cv_metrics_dict[model] = stdev(model.get_cross_validation_scores()['cvScores'][metric].values())
                    error_from_mean = mean(cv_metrics_dict.values())
                    if error_from_mean > cv_average_mean_error_limit:
                        tqdm.write(f'Error from the mean over the limit! Stopping parsimony analysis.\nError from the mean: \'{error_from_mean}\'\nLimit set: \'{cv_average_mean_error_limit}\'')
                        break
                    # ----------------------------------------------------------------------------------
                    if verbose:
                        tqdm.write(f'CV Error From the Mean: {error_from_mean} | CV Mean Error Limit: {cv_average_mean_error_limit} | CV Model Performance Metric: {metric}')
                    # ----------------------------------------------------------------------------------

            except dr.errors.ClientError as e: # TODO flesh out exceptions logger option/verbose
                if 'Feature list named' in str(e) and 'already exists' in str(e):
                    pass
                else:
                    raise e