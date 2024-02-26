# EU-Debt-Sustainability-Analysis

This repository contains Python code for the replication of the European Commission's Debt Sustainability Analysis. The replication was first introduced in the Bruegel working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see Annex 2, "Methodology and code for implementing the European Commission´s DSA in the context of the economic governance review". 

For the latest version of the model, including an application of rules agreed on by the Council, the European Parliament, and the Commission, refer to the "latest_version" folder. The Jupyter Notebook "dsa_main.ipynb" produces the results and introduces various model functionalities. The Python file "dsa_functions.py" contains functions that organize the anlysis in "dsa_main.ipynb". The Python files "DsaModelClass.py" and "StochasticDsaModelClass.py" contain the Python class DsaModelClass and its subclass StochasticDsaModel, which facilitate the deterministic and stochastic analysis. Results are saved in the "output" folder. 

Publicly available input data are saved in the "latest_version/data/InputData" folder. Note that non-public data have to be added manually before use. THis concerns interest rate and inflation expectations from Bloomberg, and growth and ageing cost data from the European Commission. For further details on data sources, see the "data_sources.xlsx" file in the data folder. Publically avaible alternatives to confidential data, as well as a BBG file with tickers for market interest rate and inflation expecations, can be found in the "latest_version/data/RawData" folder.

A previous version of the code that can be used to reproduce the results in our original working paper can be found in the "sep23_working_paper_replication_files" folder. The structure of the folder is analogous to the one described above.

External packages used include numpy, pandas, matplotlib, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org.

Author: Lennard Welslau

Date: 26/02/2024
