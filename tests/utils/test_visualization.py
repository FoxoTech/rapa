from ..conf import * # datarobot project names/ids etc

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
    for project, metric, vlines in [(project_name, 'mean', True)]:
        rapa.utils.feature_performance_stackplot(project, featurelist_prefix=featurelist_prefix,\
                                                starting_featurelist=None, feature_impact_metric=metric,\
                                                metric='AUC', vlines=vlines)
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
    