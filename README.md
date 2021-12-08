# Robust Automated Parsimony Analysis (RAPA)

With RAPA, we plan to provide a robust, freely usable, and shareable tool for automated parsimony analysis in Python.

RAPA will initially be developed on top of DataRobot's Python API to use DataRobot as a "model-running engine." In the RAPA MVP, we will provide two primary features:

* Initial feature filtering to reduce a feature list down to a size that DataRobot can receive as input.

* Automated parsimony analysis to present to the user the trade-off between the size of Feature List and the best model performance on each Feature List, presented as a Pareto front.

Although the MVP implementation of these features will be based on basic techniques such as linear feature filters and recursive feature elimination, we plan to rapidly improve these features by integrating state-of-the-art techniques from the academic literature.


The RAPA MVP is currently being demonstrated in [demo-rapa.ipynb](https://github.com/FoxoTech/rapa/blob/main/tutorials/02-tutorial.ipynb). In the near future, we will transition this project from a demo to a functioning Python module.