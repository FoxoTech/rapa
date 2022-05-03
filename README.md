# Robust Automated Parsimony Analysis (RAPA)

`RAPA` provides a robust, freely usable and shareable tool for creating and analyzing more accurate machine learning (ML) models with fewer features in Python. View documentation on [ReadTheDocs](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/).

[![ReadTheDocs](https://readthedocs.com/projects/life-epigenetics-rapa/badge/?version=latest)](https://life-epigenetics-rapa.readthedocs-hosted.com/en/latest/) [![pypi](https://img.shields.io/pypi/v/rapa.svg)](https://pypi.org/project/rapa/#data)

RAPA is currently developed on top of DataRobot's Python API to use DataRobot as a "model-running engine", with plans to include open source software such as `scikit-learn`, `tensorflow`, or `pytorch` in the future. 

Currently, RAPA provides two primary functions:

1. Initial feature filtering to reduce a feature list down to a size that DataRobot can receive as input.

2. Automated parsimony analysis using feature importance metrics directly tied to the feature's impact on accurate models (permutation importance). 


To present to the user the trade-off between the size of Feature List and the model performance for each Feature List, a series of boxplots can be plotted `rapa.utils.parsimony_performance_boxplot`.

Although the current implementation of these features will be based on basic techniques such as linear feature filters and recursive feature elimination, we plan to rapidly improve these features by integrating state-of-the-art techniques from the academic literature.


A tutorial for using `RAPA` with DataRobot is currently being demonstrated in [general_tutorial.ipynb](https://github.com/FoxoTech/rapa/blob/main/tutorials/general_tutorial.ipynb). 
