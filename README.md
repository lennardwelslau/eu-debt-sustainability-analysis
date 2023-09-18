# EU-Debt-Sustainability-Analysis

This repository contains Python code for the reproduction of the results in the Bruegel working paper "A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal" by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see Annex 2, "Methodology and code for implementing the European Commission´s DSA in the context of the economic governance review".

Code files are in the "scripts" folder. The Jupyter Notebook "ec_dsa_main.ipynb" reproduces the results and introduces various model functionalities. The Python files "EcDsaModelClass.py" and "EcStochasticModelClass.py" contain the Python class EcDsaModelClass and its subclass EcStochasticModel, which facilitate the deterministic and stochastic analysis. Results are saved in the "output" folder. Publicly available input data are in the "data/InputData" folder. For further details on data sources, see the "data_sources.xlsx" file in the data folder.

External packages used include numpy, pandas, datetime, time, os, matplotlib, seaborn, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org.

Author: Lennard Welslau

Date: 01/09/2023