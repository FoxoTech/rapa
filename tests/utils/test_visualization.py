from .. import conf

import pytest

import rapa
import os
import pickle

import datarobot as dr

# test feature performance stackplot
@pytest.mark.order(6)
def test_plot_feature_performance_stackplot():
    '''Tests that the feature performance stackplot is created without errors (however... definitely will not catch bugs)
    '''
    for project, metric, vlines in [(conf.project_name, 'mean', True)]:
        rapa.utils.feature_performance_stackplot(project, featurelist_prefix=conf.featurelist_prefix,\
                                                starting_featurelist=None, feature_impact_metric=metric,\
                                                metric='AUC', vlines=vlines, starting_featurelist=conf.starting_featurelist_name)
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
    rapa.utils.parsimony_performance_boxplot(conf.project_name,\
                                            featurelist_prefix=conf.featurelist_prefix,\
                                            starting_featurelist=conf.starting_featurelist_name,)
