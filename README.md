# European-Commission-Debt-Sustainability-Analysis
Python code for reproduction of results of Bruegel working paper "A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal" by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see Annex II, "Methodology and code for implementing the European Commission´s DSA in the context of the economic governance review".

Code files are in the "scripts" folder. The jupyter notebook "ec_dsa_analysis.ipynb" reproduces the results. The python files "EcDsaModelClass.py" and "EcStochasticModelClass.py" contain the python class EcDsaModelClass and its subclass EcStochasticModel feature the methods that facilitate deterministic and stochastic analyses. Results are saved in the "output" folder. Input data are in the "data/InputData" folder.

External packages used: numpy, pandas, datetime, time, os, matplotlib, seaborn, scipy, numba

For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org.

Author: Lennard Welslau
Date: 31/08/2023