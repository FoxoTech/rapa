import pytest
import rapa
import os
import datarobot as dr
from sklearn import datasets
import pandas as pd
from .. import conf


"""
Tests the main bread and butter of rapa, with testing the a majority of the 
automated parsimony analysis.

Tests are performend in the order they appear in the file, from the 
top down (using pytest-order)
"""

created_project_name = f"GA_TESTS rapa=={rapa.version.__version__}"

# initialize dr api


@pytest.mark.order(1)
def test_datarobot_api_initialization():
    """Tests that the datarobot api can be initialized with the test api key.
    Also serves to initialize the api so tests can run...
    """
    dr_test_api_key = os.environ.get('DR_TEST_RAPA')  # get the api key

    dr.Client(endpoint='https://app.datarobot.com/api/v2',
              token=dr_test_api_key)

    del(dr_test_api_key)  # delete api key (for security?)


# test rapa.Project initialization
@pytest.mark.order(2)
def test_Project_initialization():
    """Checks that the rapa.Project objects can be initialized with each instance of arguments
    """
    rapa_test_classification_project = rapa.Project.Classification()
    rapa_test_regression_project = rapa.Project.Regression()

    assert rapa_test_classification_project._classification == True and rapa_test_classification_project._regression == False
    assert rapa_test_regression_project._regression == True and rapa_test_regression_project._classification == False

# test creating a submittable dataframe


@pytest.mark.order(3)
def test_creating_submittable_dataframe():
    """Checks that rapa correctly makes a submittable dataframe.

        1. Submittable dataframe with no initial feature reduction
        2. Submittable df with feature reduction
        3. Checks submittable df cannot be created with a target not in the original df
    """
    target = "benign"

    # loads the dataset (as a dictionary)
    breast_cancer_dataset = datasets.load_breast_cancer()
    # puts features and targets from the dataset into a dataframe
    breast_cancer_df = pd.DataFrame(
        data=breast_cancer_dataset['data'], columns=breast_cancer_dataset['feature_names'])
    breast_cancer_df[target] = breast_cancer_dataset['target']

    bc_classification = rapa.Project.Classification()  # rapa classification project
    n_splits = 6

    # 1. Submittable dataframe with no initial feature reduction
    sub_df = bc_classification.create_submittable_dataframe(breast_cancer_df,
                                                            target_name=target,
                                                            n_splits=6,
                                                            random_state=conf.random_state)
    # checking that no columns are lost
    assert sub_df.columns.isin(breast_cancer_df.columns).sum() == len(
        breast_cancer_df.columns)
    # check the number of splits
    assert len(sub_df["partition"].value_counts()) == n_splits

    n_features = 20

    # 2. Submittalbe dataframe with initial feature reduction
    sub_df = bc_classification.create_submittable_dataframe(breast_cancer_df,
                                                            target_name=target,
                                                            n_splits=6,
                                                            random_state=conf.random_state,
                                                            n_features=n_features)
    # checking that the correct number of featres exists
    assert sub_df.columns.isin(breast_cancer_df.columns).sum() == n_features
    # check the number of splits
    assert len(sub_df["partition"].value_counts()) == n_splits

    # 3. Checks submittable df cannot be created with a target not in the original df
    absent_target = "not_there"
    try:
        sub_df = bc_classification.create_submittable_dataframe(breast_cancer_df,
                                                                target_name=absent_target,
                                                                n_splits=6,
                                                                random_state=conf.random_state,
                                                                n_features=n_features)
    except KeyError:
        # this is expected
        pass
    else: 
        raise Exception(f"A submittable df was created with the target {absent_target}")
