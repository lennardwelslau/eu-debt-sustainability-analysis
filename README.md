# EU-Debt-Sustainability-Analysis

This repository contains Python code for the replication of the results in the Bruegel working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see Annex 2, "Methodology and code for implementing the European Commission´s DSA in the context of the economic governance review".

All code files can be found in the "scripts" folder. The Jupyter Notebook "ec_dsa_main.ipynb" reproduces the results and introduces various model functionalities. The Python file "ec_dsa_functions.py" contains functions that organize the anlysis in "ec_dsa_main.ipynb". The Python files "EcDsaModelClass.py" and "EcStochasticModelClass.py" contain the Python class EcDsaModelClass and its subclass EcStochasticModel, which facilitate the deterministic and stochastic analysis. Results are saved in the "output" folder. 

Publicly available input data are saved in the "data/InputData" folder. Note that non-public data has to be added manually before use. For further details on data sources, see the "data_sources.xlsx" file in the data folder.

External packages used include numpy, pandas, datetime, time, os, matplotlib, seaborn, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org.

Author: Lennard Welslau

Date: 01/09/2023
