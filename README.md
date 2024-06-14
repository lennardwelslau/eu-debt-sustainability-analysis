# EU-Debt-Sustainability-Analysis

## Introduction 

This repository contains Python code for the replication of the European Commission's Debt Sustainability Analysis. The replication was first introduced in the Bruegel working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). For details on the methodology see the sections below, as well as Annex A3 of the [European Commission's Debt Sustainability Monitor 2023](https://economy-finance.ec.europa.eu/publications/debt-sustainability-monitor-2023_en).

For the latest version of the model, including an application of rules agreed on by the Council, the European Parliament, and the Commission, refer to the [latest_version](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/) folder. Code files can be found in the [code](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/code/) folder. The Jupyter Notebook [tutorial.ipynb](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/tutorial.ipynb) introduces various mdoel functionalities. [main.ipynb](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/main.ipynb) produces the results. The [functions](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/code/functions) folder contains functions that organize the anlysis. The [classes](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/code/classes/) folder contains the Python classes [DsaModel](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/code/classes/DsaModelClass.py) and [StochasticDsaModel](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/code/classes/StochasticDsaModelClass.py) that facilitate the deterministic and stochastic analysis. Results are saved in the [output](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/output) folder. 

All input data are saved in the [InputData](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/data/InputData) folder. For details on data sources, see the [SOURCES.xlsx](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/data/SOURCES.xlsx) file in the [data](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/data) folder.

The [sep23_working_paper](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/sep23_working_paper/) folder hosts a previous version of the code that can be used to reproduce the results in our original working paper.

External packages used include numpy, pandas, matplotlib, scipy, and numba.

For comments and suggestions please contact lennard.welslau[at]gmail[dot]com.

Author: Lennard Welslau
Last update: 01 June 2024

## Methodology

### Deterministic Debt Projections

The starting point for the DSA methodology is the European Commission’s Debt Sustainability Monitor (DSM) 2023 (European Commission, 2024). Annex A3 of the DSM describes debt dynamics and the projection of implicit interest rate on government debt. The debt ratio in a given year, $`d_t`$, is calculated as:

```math
d_t = \alpha^n \cdot d_{t-1} \cdot \frac{(1+\text{iir}_t)}{(1+g_t)} + \alpha^f \cdot d_{t-1} \cdot \frac{(1+\text{iir}_t)}{(1+g_t)} \cdot \frac{e_t}{e_{t-1}} - pb_t + f_t,
```

where:
- $`\alpha^n`$ represents the share of total government debt denominated in domestic currency,
- $`\alpha^f`$ represents the share of total government debt denominated in other currencies,
- $`\text{iir}_t`$ represents the implicit interest rate on government debt (total interest payment during a year divided by the stock of debt at the end of the previous year),
- $`g_t`$ represents the nominal growth rate of GDP (in national currency),
- $`e_t`$ represents the nominal exchange rate (expressed as national currency per foreign currency),
- $`pb_t`$ represents the primary balance ratio,
- $`f_t`$ represents stock-flow adjustments over GDP.

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

```math
m_t = 0.75 \times (\Delta \text{spb}_t - \Delta \text{spb}_t^{BL}) 
```

Here, 0.75 is the fiscal multiplier of Carnot and de Castro (2015) and $`\Delta \text{spb}_t^{BL}`$ denotes the annual change in baseline structural primary balance, which is based on the DG ECFIN projections up to 2024 and held constant thereafter. The multiplier $`m_t`$ affects real growth via its persistent effect on the output gap, narrowing the output gap by $`m_t`$ in the year of the adjustment $`t`$, and reducing its impact by one-third of its initial effect in the two consecutive periods. Thus, the total impact on the output gap in a particular year is the sum of the impact in that year plus 2/3 of the impact from the previous year plus 1/3 of the impact from two years before.

For euro area countries, Bulgaria, Czechia, Denmark, and Sweden, inflation numbers used to compute nominal growth rates are based on the European Commission’s forecast up to 2025 (GDP deflator), which are linearly interpolated with market expectations for 2033 implied by euro area inflation swaps (HICP), before converging to the 2 percent HICP inflation targets of these countries by 2053, in line with the Commission’s methodology. For Hungary, Poland, and Romania, where the central banks have higher than 2 percent inflation target, the Commission’s methodology assumes that half of the spread vis-à-vis euro area inflation expected in 2025 remains by 2033, which in turn gradually converges to the national inflation targets by 2053.

#### Projecting the Primary Balance

The primary balance ratio is the sum of the structural primary balance ratio, a cyclical component, a property income component, and an ageing cost component. Importantly, the latter component, ageing costs net of pension tax revenues, is not separated out during the adjustment period. After the end of the adjustment period, it is assumed that the structural primary balance without the change in ageing costs remains the same, thus, the change in ageing costs changes the structural primary balance after the end of the adjustment period. Ageing costs and pension tax revenues are based on the European Commission’s 2024 Ageing report. The cyclical component is defined as the product of country-specific budget balance elasticities and the output gap.

#### Projecting the Implicit (Average) Interest Rate

The implicit (average) interest rate on the public debt stock, $`\text{iir}_t`$, is projected as the weighted average of the short-term market interest rate $`i_t^{ST}`$ and the long-term implicit interest rate $`\text{iir}_t^{LT}`$:

```math
\text{iir}_t = \alpha_{t-1} \cdot i_t^{ST} + (1 - \alpha_{t-1}) \cdot \text{iir}_t^{LT} 
```

Here, $`\alpha_{t-1}`$ is the share of short-term debt in the total debt stock in $`t-1`$ and $`\text{iir}_t^{LT}`$ is projected as the weighted average of the long-term market rate $`i_t^{LT}`$ and the long-term implicit market interest rate in $`t-1`$:

```math
\text{iir}_t^{LT} = \beta_{t-1} \cdot i_t^{LT} + (1 - \beta_{t-1}) \cdot \text{iir}_{t-1}^{LT} 
```

where $`\beta_{t-1}`$ is the share of new long-term debt issuance in total long-term debt stock in $`t-1`$. Long-term market rates are projected by linearly interpolating from Bloomberg 10-year government bond benchmark rates to 10Y10Y forward rates. Between $`T+10`$ and $`T+30`$, long-term market rates converge linearly to 2 percent plus national inflation targets, which yields 4.5 percent for Poland and Romania, 5 percent for Hungary, and 4 percent for all other countries. Short-term market rates are calculated using 3 months benchmark rates, 3M10Y forward rates, and 0.5 times the country-specific values for the long-term rate in $`T+30`$.

To project the implicit interest rate forward, we calculate the new issuance and total stock of short-term and long-term debt in each period $`t`$. Gross financing needs, i.e. the size of new issuance, are the sum of all interest and amortization payments, and the primary balance. Here, interest on short-term debt is the product of short-term market rates and the stock of short-term debt in $`t-1`$. Interest on long-term debt is the product of the implied interest rate on long-term debt $`iir_t^{LT}`$ and the long-term debt stock in $`t-1`$. Short-term debt is redeemed entirely each period. The share of long-term debt maturing each year is assumed to be equal to the share of long-term debt with maturity below one year in total long-term debt in 2023. By 2033, this share is assumed to converge to the 2017-2022 historical average of redemption shares. Data for the share of maturing debt in 2023 is from the ECB. Given gross financing needs, the share of newly issued short- and long-term debt is calculated such that the share of short-term debt in total debt is held constant. The resulting debt issuances and stocks in period $`t`$ are then used to calculate the implicit interest rate in $`t+1`$

### Deterministic Debt Projections

Stochastic projections of the debt ratio are based on Annex A4 of the DSM. This approach involves drawing multiple shock series from a joint normal distribution of historical quarterly shocks for the primary balance, nominal short- and long-term interest rates, nominal GDP growth, and the exchange rate. After transforming these shocks to annual frequency and constructing the shocks to the implicit interest rate, each series is combined with the projected deterministic path of the respective variable. By recalculating the debt ratio path for each draw using the equation in Section A.2.1, we obtain the probability distribution of debt ratio projections. Unlike the Commission’s practice, which is based on 10,000 draws, we calculate the distribution based on one million draws to increase precision.

The Commission’s methodology assumes no shocks during the adjustment period. Stochastic shocks are simulated for 5 years after the end of the adjustment period, and the DSA criterion requires the public debt to GDP ratio to decline with a 70 percent probability over these five years.

#### Definition of Historical Shocks

Quarterly shocks are defined as the first differences in the historical quarterly time series. We correct for outliers by replacing observations that fall outside the 5th and 95th percentiles with the respective thresholds. Historical series are collected from the same sources listed in Table A4.1 of the DSM. Quarterly series for exchange rates, nominal GDP growth, short- and long-term interest rates, and the primary balance are all sourced from Eurostat. These data sources are described in detail in the [SOURCES.xlsx](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/latest_version/data/SOURCES.xlsx) file referenced above.

#### Aggregation of Shocks

Quarterly shocks for nominal GDP growth, the primary balance, the nominal exchange rate, and the short-term interest rate are transformed to annual frequency by summing the historical shocks in each year. In the first projection year, shocks to the long-term interest rate are transformed similarly. However, because a change in the long-term interest rate in a given quarter affects the overall interest on government debt until the debt issued in that quarter matures, aggregating quarterly long-term interest rate shocks must account for such persistence. A shock in year $`t`$ is assumed to carry over to subsequent years, proportionally to the share of maturing debt that is progressively rolled over. Thus, shocks to the implicit long-term interest rate $`\epsilon_t^{i^{LT}}`$, from the second projection year onward, are defined as:

```math
\epsilon_t^{i^{LT}} = \frac{t}{T} \sum_{q=-4t}^{4} \epsilon_q^{i^{LT}},
```

where $`T`$ denotes the average maturity of long-term debt in years, calculated as one over the historical average share of long-term debt maturing, and $`q`$ denotes the quarters of historical shocks being aggregated. Finally, shocks to the implicit interest rate on government debt are calculated as a weighted average of annualized shocks to the short- and long-term interest rates:

```math
\epsilon_t^{iir} = \alpha^{ST} \epsilon_t^{i^{ST}} + (1 - \alpha^{ST}) \epsilon_t^{i^{LT}},
```

Here, $\alpha^{ST}$ is the share of short-term debt in total government debt, calculated based on ECB data. The variance-covariance matrix of the resulting annual shock series is then used in a joint normal distribution with zero mean from which the shocks used in the stochastic projection are drawn.