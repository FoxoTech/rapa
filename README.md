# Robust Automated Parsimony Analysis (RAPA)

`rapa` provides a robust, freely usable and shareable tool for creating and analyzing more accurate machine learning (ML) models with fewer features in Python. View documentation on [ReadTheDocs](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/).

[![ReadTheDocs](https://readthedocs.com/projects/life-epigenetics-rapa/badge/?version=latest)](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/) [![pypi](https://img.shields.io/pypi/v/rapa.svg)](https://pypi.org/project/rapa/#data)

`rapa` is currently developed on top of DataRobot's Python API to use DataRobot as a "model-running engine", with plans to include open source software such as `scikit-learn`, `tensorflow`, or `pytorch` in the future. [Install using pip!](#installation)

<details open>
 <summary>Contents</summary>
<br>
 
 
* [Getting Started](#getting_started)
* [Primary Features](#primary_features)
  * [Initial Feature Filtering](#initial_feature_filtering)
  * [Automated Parsimony Analysis](#automated_parsimony_analysis)
 
 
</details>

<a name='getting_started'></a>
## Getting Started

<a name='installation'></a>
### Installation
```
pip install rapa
```

### Initializing the DataRobot API
Majority of `rapa`'s utility comes from the DataRobot auto-ML platform. To utilize DataRobot through Python, an API key is required. Acquire an API key from [app.datarobot.com](app.datarobot.com) after logging into an account. [(More information about DataRobot's API keys)](https://docs.datarobot.com/en/docs/api/api-quickstart/api-qs.html)

<div align="center">
  <p>First, log in and find the developer tools tab.</p>
  <img src="https://github.com/FoxoTech/rapa/blob/main/docs/profile_pull_down.png" alt="profile_pulldown" width="200"/>
  <br/>
</div>
  <div align="center">
  <p>Then create an API key for access to the API with Python.</p>
  <img src="https://github.com/FoxoTech/rapa/blob/main/docs/create_api_key.png" alt="api_key" width="300"/>
  <br/>
</div>


---

#### **NOTE**

**This API key lets anyone who has it access your DataRobot projects, so never share it with anyone.**

To avoid sharing your API accidentally by uploading a notebook to github, it is suggested to use the `rapa` function to read in a pickled dictionary for the API key or using [`datarobot`'s configuration for authentication.](https://docs.datarobot.com/en/docs/api/api-quickstart/api-qs.html#configure-api-authentication)

---

Once having obtained an API key, use `rapa` or `datarobot` to initialize the API connection. 

Using `rapa`, first create the pickled dictionary containting an API key.
```python
# DO NOT UPLOAD THIS CODE WITH THE API KEY FILLED OUT 
# save a pickled dictionary for datarobot api initialization in a new folder named 'data'
import os
import pickle

api_dict = {'tutorial':'APIKEYHERE'}
if 'data' in os.listdir('.'):
    print('data folder already exists, skipping folder creation...')
else:
    print('Creating data folder in the current directory.')
    os.mkdir('data')

if 'dr-tokens.pkl' in os.listdir('data'):
    print('dr-tokens.pkl already exists.')
else:
    with open('data/dr-tokens.pkl', 'wb') as handle:
        pickle.dump(api_dict, handle)
```

Then use `rapa` to initialize the API connection!
```python
# Use the pickled dictionary to initialize the DataRobot API
import rapa

rapa.utils.initialize_dr_api('tutorial')
```

[`rapa.utils.initialize_dr_api`](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/_modules/rapa/utils.html#initialize_dr_api) takes 3 arguments: token_key - the dictionary key used to store the API key as a value, file_path - the pickled dataframe file path (default: data/dr-tokens.pkl), endpoint - and the endpoint (default:https://app.datarobot.com/api/v2).


<a name='primary_features'></a>
## Primary Features

Currently, `rapa` provides two primary features:

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
To start automated parsimonious analysis using Datarobot, a DataRobot project with a target and uploaded data must already be created.

[1. Use an existing project](#existing_project)
[2. Create a new project using `rapa`](#new_project_rapa)

<a name='existing_project'></a>
### 1 - Use a previously created DataRobot project:
...

<a name='new_project_rapa'></a>
### 2 - Create and submit data for a new DataRobot project using `rapa`:
...

Then, ...

To present to the user the trade-off between the size of Feature List and the model performance for each Feature List, a series of boxplots can be plotted `rapa.utils.parsimony_performance_boxplot`.

Although the current implementation of these features will be based on basic techniques such as linear feature filters and recursive feature elimination, we plan to rapidly improve these features by integrating state-of-the-art techniques from the academic literature.



A tutorial for using `rapa` with DataRobot is currently being demonstrated in [general_tutorial.ipynb](https://github.com/FoxoTech/rapa/blob/main/tutorials/general_tutorial.ipynb). 
