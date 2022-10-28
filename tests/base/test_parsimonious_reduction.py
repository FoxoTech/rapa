from tkinter import E
from tkinter.ttk import Progressbar
import pytest
import rapa
import os
import datarobot as dr
from sklearn import datasets
import pandas as pd
from .. import conf

created_project_name = conf.created_project_name

n_splits = 6
n_features = 20
target = "benign"
regression_target = "worst area"

# loads the dataset (as a dictionary)
breast_cancer_dataset = datasets.load_breast_cancer()
# puts features and targets from the dataset into a dataframe
breast_cancer_df = pd.DataFrame(
    data=breast_cancer_dataset['data'], columns=breast_cancer_dataset['feature_names'])
breast_cancer_df[target] = breast_cancer_dataset['target']


@pytest.mark.order(12)
def test_perform_parsimony():
    """Tests performing parsimony for a classification project.

        NOTE: this takes a while!

        1. empty feature list
        2. wrong starting featurelist
        3. provide the wrong string for the project
    """
    bc_classification = rapa.Project.Classification()  # rapa classification project
    bc_regression = rapa.Project.Regression()  # rapa classification project

    sub_df = bc_classification.create_submittable_dataframe(breast_cancer_df,
                                                            target_name=target,
                                                            n_splits=6,
                                                            random_state=conf.random_state)
    project = bc_classification.submit_datarobot_project(sub_df,
                                                        target,
                                                        created_project_name+'_classification',
                                                        mode=dr.AUTOPILOT_MODE.QUICK,
                                                        random_state=conf.random_state)

    bc_classification._wait_for_jobs(project=project, progress_bar=False)

    options = {
        'feature_range': [30, 20, 10],
        'project': project,
        'starting_featurelist': 'Informative Features',
        'featurelist_prefix': f'rapa-{rapa.version.__version__}-Github Actions Test:',
        'mode': dr.AUTOPILOT_MODE.QUICK,
        'lives': 3,
        'cv_average_mean_error_limit': 1,
        'feature_impact_metric': 'mean',
        'progress_bar': False,
        'verbose': True
    }

    
    # 1. empty feature list
    try:
        bc_classification.perform_parsimony(
            feature_range=[], # wrong feature range ! (empty)
            project=options['project']
        )
    except Exception: 
        # expected
        pass
    else:
        raise Exception('Empty feature range accepted...')
    
    # 2. wrong starting featurelist
    try: 
        bc_classification.perform_parsimony(
            feature_range=options['feature_range'],
            project=options['project'],
            starting_featurelist='does_not_exist'
        )
    except Exception:
        # expected
        pass
    else:
        raise Exception('Wrong starting featurelist accepted...')

    # 3. provide the wrong string for the project
    try:
        bc_classification.perform_parsimony(
            feature_range=options['feature_range'],
            project='does_not_exist'
        )  
    except Exception:
        # expected
        pass
    else:
        raise Exception('Unknown project provided and accepted...')

    # 4. Full classification parsimony
    bc_classification.perform_parsimony(
        **options
    )
