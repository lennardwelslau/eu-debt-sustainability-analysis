# EU-Debt-Sustainability-Analysis

## Introduction 

This repository contains Python code for the replication of the European Commission's Debt Sustainability Analysis. The replication was first introduced in the Bruegel working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology section below, as well as Annex A3 of the [European Commission's Debt Sustainability Monitor 2023](https://economy-finance.ec.europa.eu/publications/debt-sustainability-monitor-2023_en).

For the latest version of the model, including an application of rules agreed on by the Council, the European Parliament, and the Commission, refer to the "latest_version" folder. Code files can be found in the "code" folder. The Jupyter Notebook "main.ipynb" produces the results and introduces various model functionalities. The "functions" folder contains functions that organize the anlysis in "main.ipynb". The "classes" folder contains the Python files "DsaModelClass.py" and "StochasticDsaModelClass.py", which contain the class DsaModel and its subclass StochasticDsaModel that facilitate the deterministic and stochastic analysis. Results are saved in the "output" folder. 

All input data are saved in the "data/InputData" folder. For details on data sources, see the "SOURCES.xlsx" file in the data folder.

A previous version of the code that can be used to reproduce the results in our original working paper can be found in the "sep23_working_paper_replication_files" folder.

External packages used include numpy, pandas, matplotlib, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]gmail[dot]com.

Author: Lennard Welslau
Last update: 01 June 2024

## Methodology

### Deterministic Debt Projections

The starting point for the DSA methodology applied in this paper is the European Commission’s Debt Sustainability Monitor (DSM) 2023 (European Commission, 2024). Annex A3 of the DSM describes debt dynamics and the projection of implicit interest rate on government debt. The debt ratio in a given year, $d_t$, is calculated as:

$$
d_t = \alpha^n \cdot d_{t-1} \cdot \frac{(1+\text{iir}_t)}{(1+g_t)} + \alpha^f \cdot d_{t-1} \cdot \frac{(1+\text{iir}_t)}{(1+g_t)} \cdot \frac{e_t}{e_{t-1}} - pb_t + f_t, 
$$

where:
- $\alpha^n$ represents the share of total government debt denominated in domestic currency,
- $\alpha^f$ represents the share of total government debt denominated in other currencies,
- $\text{iir}_t$ represents the implicit interest rate on government debt (total interest payment during a year divided by the stock of debt at the end of the previous year),
- $g_t$ represents the nominal growth rate of GDP (in national currency),
- $e_t$ represents the nominal exchange rate (expressed as national currency per foreign currency),
- $pb_t$ represents the primary balance ratio,
- $f_t$ represents stock-flow adjustments over GDP.

#### Adverse Deterministic Stress Tests

In addition to the baseline deterministic scenario, three alternative deterministic scenarios, or stress tests, are also calculated by the Commission:

- **‘Lower SPB’ scenario**: the SPB is assumed to be reduced by 0.5 pp. of GDP in total, with a reduction of 0.25 pp. each year over the first two years, and to remain at that level afterwards (apart from changes in the cost of ageing – see below).
- **‘Adverse r-g’ scenario**: the interest/growth-rate differential is assumed to be permanently increased by 1 percentage point.
- **‘Financial stress’ scenario**: market interest rates are assumed to temporarily increase for one year by 1 pp., plus a risk premium for high-debt countries.

These adverse scenarios are assumed for ten years after the end of the adjustment period. The DSA criterion requires the public debt to GDP ratio to decline under these adverse scenarios.

#### Data Sources

- Shares of euro-denominated debt are calculated based on ECB data.
- Exchange rates are taken from Eurostat. Both variables are assumed to remain constant over the projection horizon.
- Stock-flow adjustments are taken from the AMECO database and based on projections by the European Commission’s DG ECFIN, available up to 2025. From 2026, stock-flow adjustments are assumed to be zero for all but three EU countries (Finland, Luxembourg, and Greece, see [DSM 2023 Section II.2](https://economy-finance.ec.europa.eu/publications/debt-sustainability-monitor-2023_en) for details).
- Nominal GDP growth, the primary balance, and the implicit interest rate on government debt are endogenous model variables. They build on medium-term real growth, output gap, and GDP-deflator projections by the European Commission’s Output Gap Working Group, long-term growth and ageing-cost projections based on the European Commission’s 2024 Ageing report, long-term market expectations for inflation from Bloomberg, structural primary balance projections from the AMECO database, fiscal multiplier data based on Carnot and de Castro (2015), and budget balance semi-elasticities based on Mourre et al. (2019).

The projection of the implicit interest rate on government debt further relies on ECB data on government debt stocks, shares of short- and long-term debt issuance, and average annual debt redemption, as well as market expectations for interest rates from Bloomberg. All data sources are described in detail in the [SOURCES.xlsx](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/data/SOURCES.xlsx) file found in the data folder of the accompanying GitHub repository.

#### Projecting Nominal Growth

The effect of fiscal stimulus and the cyclical dependence of the budget balance makes growth and primary balance projections mutually dependent. These dependencies affect the variables from the beginning of the adjustment period in 2025. Prior to the adjustment period, i.e. up to 2024, the model relies directly on projections for the primary balance and nominal growth taken from the AMECO database. From 2025, real growth is affected by annual adjustments of the structural primary balance. Specifically, in a given year, the effect of the fiscal multiplier effect is proportional to annual adjustments in the structural primary balance relative to its baseline trajectory:

$$ 
m_t = 0.75 \times (\Delta \text{spb}_t - \Delta \text{spb}_t^{BL}) 
$$

Here, 0.75 is the fiscal multiplier of Carnot and de Castro (2015) and $\Delta \text{spb}_t^{BL}$ denotes the annual change in baseline structural primary balance, which is based on the DG ECFIN projections up to 2024 and held constant thereafter. The multiplier $m_t$ affects real growth via its persistent effect on the output gap, narrowing the output gap by $m_t$ in the year of the adjustment $t$, and reducing its impact by one-third of its initial effect in the two consecutive periods. Thus, the total impact on the output gap in a particular year is the sum of the impact in that year plus 2/3 of the impact from the previous year plus 1/3 of the impact from two years before.

For euro area countries, Bulgaria, Czechia, Denmark, and Sweden, inflation numbers used to compute nominal growth rates are based on the European Commission’s forecast up to 2025 (GDP deflator), which are linearly interpolated with market expectations for 2033 implied by euro area inflation swaps (HICP), before converging to the 2 percent HICP inflation targets of these countries by 2053, in line with the Commission’s methodology. For Hungary, Poland, and Romania, where the central banks have higher than 2 percent inflation target, the Commission’s methodology assumes that half of the spread vis-à-vis euro area inflation expected in 2025 remains by 2033, which in turn gradually converges to the national inflation targets by 2053.

#### Projecting the Primary Balance

The primary balance ratio is the sum of the structural primary balance ratio, a cyclical component, a property income component, and an ageing cost component. Importantly, the latter component, ageing costs net of pension tax revenues, is not separated out during the adjustment period. After the end of the adjustment period, it is assumed that the structural primary balance without the change in ageing costs remains the same, thus, the change in ageing costs changes the structural primary balance after the end of the adjustment period. Ageing costs and pension tax revenues are based on the European Commission’s 2024 Ageing report. The cyclical component is defined as the product of country-specific budget balance elasticities and the output gap.

#### Projecting the Implicit (Average) Interest Rate

The implicit (average) interest rate on the public debt stock, $\text{iir}_t$, is projected as the weighted average of the short-term market interest rate $i_t^{ST}$ and the long-term implicit interest rate $\text{iir}_t^{LT}$:

$$ 
\text{iir}_t = \alpha_{t-1} \cdot i_t^{ST} + (1 - \alpha_{t-1}) \cdot \text{iir}_t^{LT} 
$$

Here, $\alpha_{t-1}$ is the share of short-term debt in the total debt stock in $t-1$ and $\text{iir}_t^{LT}$ is projected as the weighted average of the long-term market rate $i_t^{LT}$ and the long-term implicit market interest rate in $t-1$:

$$ 
\text{iir}_t^{LT} = \beta_{t-1} \cdot i_t^{LT} + (1 - \beta_{t-1}) \cdot \text{iir}_{t-1}^{LT} 
$$

where $\beta_{t-1}$ is the share of new long-term debt issuance in total long-term debt stock in $t-1$. Long-term market rates are projected by linearly interpolating from Bloomberg 10-year government bond benchmark rates to 10Y10Y forward rates. Between T+10 and T+30, long-term market rates converge linearly to 2 percent plus national inflation targets, which yields 4.5 percent for Poland and Romania, 5 percent for Hungary, and 4 percent for all other countries. Short-term market rates are calculated using 3 months benchmark rates, 3M10Y forward rates, and 0.5 times the country-specific values for the long-term rate in T+30.

To project the implicit interest rate forward, we calculate the new issuance and total stock of short-term and long-term debt in each period $t$. Gross financing needs, i.e. the size of new issuance, are the sum of all interest and amortization payments, and the primary balance. Here, interest on short-term debt is the product of short-term market rates and the stock of short-term debt in $t-1$. Interest on long-term debt is the product
