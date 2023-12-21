# EU-Debt-Sustainability-Analysis

This repository contains Python code for the replication of the results in the Bruegel working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see Annex 2, "Methodology and code for implementing the European Commission´s DSA in the context of the economic governance review".

All code files needed for the replication of the working paper results can be found in the "sep23_working_paper_replication_files/scripts" folder. The Jupyter Notebook "ec_dsa_main.ipynb" reproduces the results and introduces various model functionalities. The Python file "ec_dsa_functions.py" contains functions that organize the anlysis in "ec_dsa_main.ipynb". The Python files "EcDsaModelClass.py" and "EcStochasticModelClass.py" contain the Python class EcDsaModelClass and its subclass EcStochasticModel, which facilitate the deterministic and stochastic analysis. Results are saved in the "output" folder. 

Publicly available input data are saved in the "sep23_working_paper_replication_files/data/InputData" folder. Note that non-public data have to be added manually before use. For further details on data sources, see the "data_sources.xlsx" file in the data folder. Publically avaible alternatives to confidential data, as well as a BBG file with tickers for market interest rate and inflation expecations, can be found in the "sep23_working_paper_replication_files/data/RawData" folder.

The "latest_version" folder contains an up to date version of the code used to produce the [assessment of the December 21 Council compromise](https://www.bruegel.org/first-glance/assessing-ecofin-compromise-fiscal-rules-reform). The structure of the folder is the same as described above.

External packages used include numpy, pandas, datetime, time, os, matplotlib, seaborn, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org.

Author: Lennard Welslau

Date: 21/12/2023
