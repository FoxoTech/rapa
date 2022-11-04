from .. import config

import pytest

import rapa
import os
import pickle

import datarobot as dr

# test feature performance stackplot
@pytest.mark.order(6)
def test_plot_feature_performance_stackplot():
    '''Tests that the feature performance stackplot is created without errors (however... definitely will not catch bugs)

    Tests different situations:
        1. Mean metric, with vlines, and with a starting featurelist
        2. Cumulative metric, without vlines, and without a starting featurelist
        3. wrong feature_impact_metric
    '''
    # 1. Mean metric, with vlines, and with a starting featurelist
    for project, metric, vlines in [(config.classification_project_name, 'mean', True)]:
        rapa.utils.feature_performance_stackplot(project, featurelist_prefix=config.featurelist_prefix,\
                                                feature_impact_metric=metric,\
                                                metric='AUC', vlines=vlines,\
                                                starting_featurelist=config.starting_featurelist_name)
    
    # 2. Cumulative metric, without vlines, and without a starting featurelist
    for project, metric, vlines in [(config.classification_project_name, 'cumulative', False)]:
        rapa.utils.feature_performance_stackplot(project, featurelist_prefix=config.featurelist_prefix,\
                                                feature_impact_metric=metric,\
                                                metric='AUC', vlines=vlines)

    # 3. wrong feature_impact_metric
    try:
        for project, metric, vlines in [(config.classification_project_name, 'wrong', False)]:
            rapa.utils.feature_performance_stackplot(project, featurelist_prefix=config.featurelist_prefix,\
                                                    feature_impact_metric=metric,\
                                                    metric='AUC', vlines=vlines)
    except Exception:
        # this is expected
        pass
    else:
        raise Exception("The plot was somehow created when given a wrong metric")
    '''
    [(project_name, 'mean', False),\
    (rapa.utils.find_project(project_name), 'mean', False),\
    (project_name, 'cumulative', False),\
    (project_name, 'median', False),\
    (project_name, 'mean', True)]'''

@pytest.mark.order(7)
def test_plot_parsimony_model_performance():
    '''Tests that the model performance boxplots are created without errors (however... definitely will not catch bugs)
    '''
    rapa.utils.parsimony_performance_boxplot(config.classification_project_name,\
                                            featurelist_prefix=config.featurelist_prefix,\
                                            starting_featurelist=config.starting_featurelist_name,)
