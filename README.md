# Robust Automated Parsimony Analysis (RAPA)

`rapa` provides a robust, freely usable and shareable tool for creating and analyzing more accurate machine learning (ML) models with fewer features in Python. View documentation on [ReadTheDocs](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/).

[![ReadTheDocs](https://readthedocs.com/projects/life-epigenetics-rapa/badge/?version=latest)](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/) [![pypi](https://img.shields.io/pypi/v/rapa.svg)](https://pypi.org/project/rapa/#data)

`rapa` is currently developed on top of DataRobot's Python API to use DataRobot as a "model-running engine", with plans to include open source software such as `scikit-learn`, `tensorflow`, or `pytorch` in the future. [Install using pip!](#installation)

## Getting Started

### Initializing the DataRobot API
Majority of `rapa`'s utility comes from the DataRobot auto-ML platform. To utilize DataRobot through Python, an API key is required. Acquire an API key from [app.datarobot.com](app.datarobot.com) after logging into an account. [(More information about DataRobot's API keys)](https://docs.datarobot.com/en/docs/api/api-quickstart/api-qs.html)
<p align="center">
  <div>
    First, log in and find the developer tools tab.<br/>
    <img src="https://github.com/FoxoTech/rapa/blob/main/docs/profile_pull_down.png" alt="profile_pulldown" width="200"/>
  </div>
  <div>
    Then create an API key for access to the API with Python.<br/>
    <img src="https://github.com/FoxoTech/rapa/blob/main/docs/create_api_key.png" alt="api_key" width="300"/>
  </div>
</p>

Currently, `rapa` provides two primary functions:

  [1. Initial feature filtering](#initial_feature_filtering) to reduce a feature list down to a size that DataRobot can receive as input.

  [2. Automated parsimony analysis](#automated_parsimony_analysis) using feature importance metrics directly tied to the feature's impact on accurate models (permutation importance). 

<a name='initial_feature_filtering'></a>
## Initial Feature Filtering
Automated machine learning is easily applicable to samples with fewer features, as the time and resources required reduces significantly as the number of features decreases. Additionally, DataRobot's automated ML platform only accepts projects that have up to 20,000 features per sample. 

For feature selection, `rapa` uses `sklearn`'s ```f_classif``` or ```f_regression``` to reduce the number of features. This provides an ANOVA F-statistic for each sample, which is then used to select the features with the highest F-statistics.

```python
# first, create a rapa classification object
rapa_classif = rapa.rapa.RAPAClassif()

# then provide the original data for feature selection
sdf = rapa_classif.create_submittable_dataframe(input_data_df=input, 
                                                target_name='target_column', 
                                                n_features=2000)
```

---
**NOTE**

When calling ```create_submittable_dataframe```, the provided ```input_data_df``` should have all of the features as well as the target as columns, and samples as the index.

If the number of features is reduced, then there should be no missing values.

---

<a name='automated_parsimony_analysis'></a>
## Automated Parsimony Analysis

To present to the user the trade-off between the size of Feature List and the model performance for each Feature List, a series of boxplots can be plotted `rapa.utils.parsimony_performance_boxplot`.

Although the current implementation of these features will be based on basic techniques such as linear feature filters and recursive feature elimination, we plan to rapidly improve these features by integrating state-of-the-art techniques from the academic literature.



<a name='installation'></a>
## Installation
```pip install rapa```


A tutorial for using `rapa` with DataRobot is currently being demonstrated in [general_tutorial.ipynb](https://github.com/FoxoTech/rapa/blob/main/tutorials/general_tutorial.ipynb). 
