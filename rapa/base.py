from sklearn.feature_selection import f_regression, f_classif
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_array

from typing import List
from typing import Callable
from typing import Tuple
from typing import Union

import pandas as pd
import numpy as np
from statistics import variance

import datarobot as dr

class RAPABase():
    """
        The base of regression and classification RAPA analysis
    """

    POSSIBLE_TARGET_TYPES = [x for x in dir(dr.enums.TARGET_TYPE) if not x.startswith('__')] # List of DR TARGET_TYPES

    _classification = None # Set by child classes
    target_type = None # Set at initialization
    # target_name = None # Set with 'create_submittable_dataframe()'
    project = None # Set at initialization or with 'perform_parsimony()'



    def __init__(self, project: Union[dr.Project, str] = None):
        if self.__class__.__name__ == "RAPABase":
            raise RuntimeError("Do not instantiate the RAPABase class directly; use RAPAClassif or RAPARegress")
        


    def create_submittable_dataframe(self, input_data_df: pd.DataFrame, target_name: str, max_features: int = 19990,
                                    n_splits: int = 6, filter_function: Callable[[pd.DataFrame, np.ndarray], List[np.ndarray]] = f_classif,
                                    random_state: int = None) -> Tuple[pd.DataFrame, str]:
        """Prepares the input data for submission as either a regression or classification problem on DataRobot.

        Creates pre-determined k-fold cross-validation splits and filters the feature
        set down to a size that DataRobot can receive as input, if necessary.

        ## Parameters
        ----------
        input_data_df: pandas.DataFrame
            pandas DataFrame containing the feature set and prediction target.

        target_name: str
            Name of the prediction target column in `input_data_df`.

        max_features: int, optional (default: 19990)
            The number of features to reduce the feature set in `input_data_df`
            down to. DataRobot's maximum feature set size is 20,000.

        n_splits: int, optional (default: 6)
            The number of cross-validation splits to create. One of the splits
            will be retained as a holdout split, so by default this function
            sets up the dataset for 5-fold cross-validation with a holdout.

        filter_function: callable, optional (default: sklearn.feature_selection.f_regression)
            The function used to calculate the importance of each feature in
            the initial filtering step that reduces the feature set down to
            `max_features`.

            This filter function must take a feature matrix as the first input
            and the target array as the second input, then return two separate
            arrays containing the feature importance of each feature and the
            P-value for that correlation, in that order.

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

        # Check dataframe has 'target_name' columns
        if target_name not in input_data_df.columns:
            raise AssertionError(f'{target_name} is not a column in the input DataFrame')
        # self.target_name = target_name # set self.target_name

        # Check that the dataframe can be copied and remove target_name column
        input_data_df = input_data_df.copy()
        only_features_df = input_data_df.drop(columns=[target_name])

        # Set target_type and kfold_type based on type of classification/regression problem
        if self._classification:
            # Check if binary or multi classification problem
            if len(np.unique(input_data_df[target_name].values)) == 2:
                self.target_type = dr.enums.TARGET_TYPE.BINARY
            else:
                self.target_type = dr.enums.TARGET_TYPE.MULTICLASS
            kfold_type = StratifiedKFold
        else:
            # Check array for infinite values/NaNs
            check_array(input_data_df)
            kfold_type = KFold
            self.target_type = dr.enums.TARGET_TYPE.REGRESSION

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
            if fold_num > 0:
                feature_importances, _ = filter_function(only_features_df.iloc[fold_indices].values, input_data_df[target_name].iloc[fold_indices].values)
                train_feature_importances.append(feature_importances)

        # We calculate the overall feature importance scores by averaging the feature importance scores across all of the training folds
        avg_train_feature_importances = np.mean(train_feature_importances, axis=0)

        # Change parition 0 name to 'Holdout'
        input_data_df.loc[input_data_df['partition'] == f'{fold_name_prefix} 0', 'partition'] = 'Holdout'

        most_correlated_features = only_features_df.columns.values[np.argsort(avg_train_feature_importances)[::-1][:max_features]].tolist()
  
        datarobot_upload_df = input_data_df[[target_name, 'partition'] + most_correlated_features]

        return datarobot_upload_df


    def submit_datarobot_project(self, input_data_df: pd.DataFrame, target_name: str, project_name: str, 
                                target_type: str = None, worker_count: int = -1, mode: str = dr.AUTOPILOT_MODE.FULL_AUTO,
                                random_state: int = None) -> dr.Project:
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

        # Check for a target_type
        if target_type == None or target_type not in self.POSSIBLE_TARGET_TYPES:
            target_type = self.target_type
            if target_type == None:
                raise Exception(f'No target type.')
        
        """if target_name == None: # TODO think about this
            target_name = self.target_name
            if target_name == None:
                raise Exception(f'No target name.')"""

        project = dr.Project.create(sourcedata=input_data_df, project_name=project_name)

        project.set_target(target=target_name, target_type=target_type,
                        worker_count=worker_count, mode=mode,
                        advanced_options=dr.AdvancedOptions(seed=random_state, accuracy_optimized_mb=False,
                                                            prepare_model_for_deployment=False, blend_best_models=False),
                        partitioning_method=dr.UserCV(user_partition_col='partition', cv_holdout_level='Holdout'))

        return project


    def perform_parsimony(self, project: Union[dr.Project, str], 
                        feature_range: List[Union[int, float]], 
                        starting_featurelist: str = 'Informative Features',
                        featurelist_prefix: str = 'RAPA Reduced to', 
                        lives: int = None, 
                        cv_variance_limit: float = None, 
                        feature_importance_statistic: str = 'median',
                        progress_bar: bool = True, to_graph: List[str] = [], 
                        scoring_metric: str = None):
        """Performs parsimony analysis by repetatively extracting feature importance from 
        DataRobot models and creating new models with reduced features (smaller feature lists).

        Parameters:
        ----------
        project: datarobot.Project | str
            Either a datarobot project, or a string of it's id or name
        
        feature_range: list[int] | list[float]
            Either a list containing integers representing desired featurelist lengths in descending order,
            or a list containing floats representing desired featurelist percentages (of the original featurelist size)
        
        starting_featurelist: str
            The name of the featurelist that rapa will start pasimony analysis with 

        featurelist_prefix: str, optional (default = 'RAPA Reduced to')
            The desired prefix for the featurelists that rapa creates in datarobot. Each featurelist
            will start with the prefix, and end with the number of features in that featurelist
        
        lives: int, optional (default = None)
            The number of times allowed for reducing the featurelist and obtaining a worse model. By default,
            'lives' are off, and the entire 'feature_range' will be ran, but if supplied a number >= 0, then 
            that is the number of 'lives' there are. 

            Ex: lives = 0, feature_range = [100, 90, 80, 50]
            RAPA finds that after making all the models for the length 80 featurelist, the 'best' model was created with the length
            90 featurelist, so it stops and doesn't make a featurelist of length 50.

            Similar to datarobot's Feature Importance Rank Ensembling for advanced feature selection (FIRE) package's 'lifes' 
            https://www.datarobot.com/blog/using-feature-importance-rank-ensembling-fire-for-advanced-feature-selection/ 
        
        cv_variance_limit: float, optional (default = None)
            The limit of cross validation variance to avoid overfitting. By default, the limit is off, 
            and the each 'feature_range' will be ran. Limit exists only if supplied a number >= 0.0

            Ex: 'feature_range' = 2.5, feature_range = [100, 90, 80, 50]
            RAPA finds that after making all the models for the length 80 featurelist, the 'best' model was created with the length
            90 featurelist, so it stops and doesn't make a featurelist of length 50.

        """
        dr.Project()