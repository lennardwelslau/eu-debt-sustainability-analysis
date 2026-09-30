# EU-Debt-Sustainability-Analysis

## Introduction 

This repository contains Python code for the replication of the European Commission's Debt Sustainability Analysis featured in the Bruegel working paper [The implications of the European Union’s new fiscal rules](https://www.bruegel.org/policy-brief/implications-european-unions-new-fiscal-rules) by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2024). The replication was first introduced in the 2023 working paper [A Quantitative Evaluation of the European Commission´s Fiscal Governance Proposal](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal). For details on the methodology, refer to the sections below, as well as Annex A3 of the [European Commission's Debt Sustainability Monitor 2023](https://economy-finance.ec.europa.eu/publications/debt-sustainability-monitor-2023_en).

Repository structure:

- `code/` notebooks:
  - [main.ipynb](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/main.ipynb) produces the results for all EU countries (Excel workbook and country charts in `output/`).
  - [tutorial.ipynb](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/tutorial.ipynb) introduces the model: input data (updating data, data you have to provide yourself), deterministic and stochastic projections, `find_spb_binding` and its rule options, several countries, and additional functions and use cases.
  - [commission_replication.ipynb](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/commission_replication.ipynb) replicates the Commission's reference trajectories from its prior guidance calculation sheets, compares them with the default rules and discusses the drivers of the differences.
  - [exercises](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/exercises/): additional analyses (EU fiscal rules targets for Hungary from 2026 with Bloomberg market data and an optional own 2026 projection, EU rules vs. German debt brake, green golden rule, potential growth sensitivity).
- `code/classes/`: the model.
  - [DsaModel](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/classes/DsaModelClass.py): deterministic projections and optimization.
  - [StochasticDsaModel](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/classes/StochasticDsaModelClass.py): stochastic projections; inherits the EU fiscal rules from `FiscalRules`.
  - [FiscalRules](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/classes/FiscalRules.py): the integrated optimizer `find_spb_binding`, the EDP and the safeguards, with the default and Commission rules.
  - [GroupDsaModel](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/classes/GroupDsaModelClass.py): several countries in parallel, results tables and Excel output.
  - [ResultsTables](https://github.com/lennardwelslau/eu-debt-sustainability-analysis/blob/main/code/classes/ResultsTables.py): labelled results tables and the results workbook.
- `code/data_pipeline/`: builds the Excel input workbook read by the model (see [Input data](#input-data) below).
- `code/functions/`: country charts (`plot_annex_charts`), consecutive adjustment periods (`run_consecutive_dsa`) and the green golden rule scenario.
- `data/InputData/`: input workbooks (`dsa_inputs_<vintage>.xlsx`); `data/RawData/`: downloaded source files (Commission prior guidance sheets, Ageing Report, Debt Sustainability Monitor).
- `output/`: results (`example_folder` from `main.ipynb`, `validation` from the Commission comparison).

External packages used include matplotlib, scipy, numba, openpyxl, requests and dbnomics. For comments and suggestions please contact lennard.welslau[at]gmail[dot]com.

Author: Lennard Welslau
Last update: September 2026

## Installation

The code runs with Python 3.11 (Anaconda or any Python distribution):

```
pip install -r requirements.txt
```

Start Jupyter and run the notebooks from the `code` folder (the notebooks import `classes`, `functions` and `data_pipeline` from there, and `code/matplotlibrc` sets the chart style). The exercises in `code/exercises` add the `code` folder to the path themselves. Building new input workbooks (`build_inputs`) needs an internet connection; source files are downloaded to `data/RawData` (the Ageing Report and Debt Sustainability Monitor files are not part of the repository and are downloaded on first use).

## Input data

The model reads all country data from one Excel workbook in `data/InputData`:

| Sheet | Content |
|---|---|
| `README`, `Sources` | Vintage, build mode, colour legend, source of every variable |
| `Overview` | Key starting values for all countries |
| `Overrides` | User assumptions (`COUNTRY \| CODE \| YEAR \| VALUE \| COMMENT`, YEAR blank for parameters), applied when the model reads the file |
| `<ISO3>` | One sheet per country: **PARAMETERS** (debt structure, market anchors, EDP status, elasticities, multiplier) and **TIME SERIES** (forecasts, ageing costs, stock-flow, repayments), colour-coded by source |
| `Shocks_Q`, `Shocks_A` | Historical shocks (first differences) for the stochastic projections |
| `History_Q`, `History_A` | Underlying historical levels, for transparency |

Blank cells are filled by the model following the Commission methodology (interpolation to T+10 and T+30 anchors). Workbooks shipped in `data/InputData`:

| Workbook | Content |
|---|---|
| `dsa_inputs_<yyyy_mm>.xlsx` (default: most recent) | Up-to-date public data (API mode), market anchors from the latest Commission guidance, fiscal multiplier 0.75 with persistent effect on the output gap |
| `dsa_inputs_commission_2024.xlsx` | All inputs from the Commission prior guidance calculation sheets (2024 guidance), reproducing the Commission reference trajectories (Commission output gap rule) |
| `dsa_inputs_commission_latest.xlsx` | As above, latest guidance per country (e.g. IE and NL 2025, CZ 2026, which use a fiscal multiplier of 0.6) |
| `dsa_inputs_2025_10.xlsx` | Legacy October 2025 input data, for replication of earlier results |

```python
from classes import StochasticDsaModel as DSA
model = DSA('FRA')                                                  # latest public data, start year = reference year T
model.find_spb_binding()                                            # Darvas, Welslau and Zettelmeyer (2024) rules, results as tables
model.binding_tables['Adjustment path']                             # pandas tables: 'Summary', 'SPB targets', 'Adjustment path'

model = DSA('FRA', input_file='dsa_inputs_commission_2024.xlsx')    # Commission 2024 prior guidance data
model.find_spb_binding(rules='commission')                          # Commission prior guidance rules (reference trajectory)
model.find_spb_binding(edp='commission', frontloading=False)        # individual rules can be switched, see below
```

Inputs that are not publicly available default to the values in the latest Commission prior guidance calculation sheet and can be updated by the user:

| Input | Workbook code | How to update |
|---|---|---|
| Bloomberg forward rates (market expectations T+10) | `INTEREST_RATE_ST_T10`, `INTEREST_RATE_LT_T10` | `Overrides` sheet or `set_overrides` |
| Bloomberg inflation swaps (T+10) | `INFLATION_T10` | `Overrides` sheet or `set_overrides` |
| Long-run anchors (T+30) | `INTEREST_RATE_*_T30`, `INFLATION_T30` | `Overrides` sheet or `set_overrides` |
| Output Gaps Working Group projections T+3 to T+5 (CIRCABC) | `REAL_GDP`, `POTENTIAL_GDP` | `build_inputs(mode='api', ogwg_file=...)`; otherwise the output gap closes by T+5 |
| EDP status, budget semi-elasticity | `EXCESSIVE_DEFICIT_PROCEDURE`, `BUDGET_BALANCE_ELASTICITY` | `Overrides` sheet or `set_overrides`; EDP decisions after the guidance (e.g. Austria 2025) are listed in `EDP_STATUS_UPDATES` in `data_pipeline/sources.py` |

```python
from data_pipeline import set_overrides, latest_input_file
set_overrides(latest_input_file(), [{'COUNTRY': 'FRA', 'CODE': 'INTEREST_RATE_LT_T10', 'VALUE': 4.0, 'COMMENT': 'Bloomberg, Sep 2026'}],
              out_file='dsa_inputs_user.xlsx')   # omit out_file to edit the workbook in place

# Or without changing any file: overrides applied in memory for one model
model = DSA('FRA', overrides=[{'CODE': 'INTEREST_RATE_LT_T10', 'VALUE': 4.0}, {'CODE': 'FISCAL_BALANCE', 'YEAR': 2026, 'VALUE': -5.0}])
```

Workbooks are built with the `data_pipeline` package. Downloaded source files are stored in `data/RawData`.

```python
from data_pipeline.build import build_inputs

# Up-to-date public data: AMECO (via DBnomics), ECB Data Portal, ESM repayments, Ageing Report 2024, DSM 2025 fiches,
# historical shocks from Eurostat, OECD and ECB. Market anchors (Bloomberg forward rates, inflation swaps), EDP status
# and semi-elasticities default to the Commission prior guidance sheets. Output Gaps Working Group data are optional.
build_inputs(mode='api', ogwg_file=None)

# All inputs from the Commission prior guidance calculation sheets
build_inputs(mode='commission', prior_guidance='2024')   # or 'latest' (default)
```

## Rules and results

`find_spb_binding` implements the rules in two versions. The default follows Darvas, Welslau and Zettelmeyer (2024) and translates the legal text consistently across countries, allowing front-loading of the adjustment. `rules='commission'` switches all rules to the version of the Commission prior guidance calculation sheets; each rule can also be switched individually:

| Option | Default | `'commission'` |
|---|---|---|
| `frontloading` | EDP and deficit resilience steps front-load the adjustment; later steps are reduced to keep the SPB target | Constant annual adjustment; minimum steps are added on top where they bind |
| `dsa_criteria` | Per scenario, debt declines or is below 60%; deficit below 3% | Debt declines in all scenarios (and stochastically) or is below 60% in all scenarios; technical information for countries with debt < 60% and deficit < 3% |
| `stochastic_criteria` | Debt declines or is below 60% with 70% probability | Debt declines with 70% probability |
| `edp_status` | EDP status in T from the input file; `'infer'`: predicted by the model from a projected deficit above 3% | EDP status in T from the input file |
| `edp` | Minimum steps of 0.5 pp. while the deficit exceeds 3%, front-loaded | Min. 0.5 pp. step after a year with a deficit above 3%, added on top; abrogated after two years with deficit below 3% |
| `debt_safeguard` | Average decline from the year the EDP is projected to be abrogated (as in the Commission sheets), or T without EDP, by debt in T | By start-of-year debt band, averaged over adjustment years outside the EDP |
| `deficit_resilience` | Steps raised until the structural deficit is below 1.5% in the same year | Step after a year with a structural deficit above 1.55% |
| `grid` | Exact SPB target | Annual adjustment rounded up to 0.01 pp.; floor of 0 pp. (reference trajectory) or -1 pp. per year (technical information) |

In both versions, the EDP benchmark applies to the SPB until 2027 and to the structural balance from 2028 (Regulation (EU) 2024/1264, recital 23). Both versions also use the same implementation of each deterministic DSA criterion (`find_spb_deterministic`): over the 10 years after the adjustment period, the debt ratio declines, the debt ratio is at or below 60% at the end, and the deficit is at or below 3%. The versions differ only in how the criteria are combined. The Commission sheets appear to allow a small tolerance on the deficit (up to about 3.05%); the model applies the 3% threshold of the regulation in both versions, which changes the Commission replication by at most 0.03 pp. per year where the deficit criterion binds.

Results are displayed as pandas tables in notebooks (`print_results=True`) and stored in `model.binding_tables`; `model.key_results()` returns key variables with readable labels. For several countries, `GroupDsaModel.summary()` combines the results in one table and `GroupDsaModel.save_results(folder)` writes one Excel workbook (README, Summary, SPB targets, Adjustment paths, debt by scenario, and one sheet per country with key variables). `save_dfs` still exports all raw model variables.

`data_pipeline.validate.compare_with_commission(input_file)` runs the model for all countries and compares debt paths and required SPB adjustments with the Commission prior guidance sheets (see `output/validation`); `data_pipeline.validate.compare_rules(input_file)` decomposes the differences between the two rule sets. `data_pipeline.convert_legacy_csv(csv_file)` converts CSV input files of earlier versions of this repository to the workbook format.

## Data sources and licence

The code is published under the MIT licence (see `LICENSE`). The licence does not cover the data. Input workbooks and raw files contain data from:

- European Commission (DG ECFIN): AMECO database and forecasts, prior guidance calculation sheets, Debt Sustainability Monitor 2025 country fiches, 2024 Ageing Report. Reuse is authorised with acknowledgement of the source ([Commission reuse policy](https://commission.europa.eu/legal-notice_en#copyright-notice)). Market expectations in the prior guidance sheets (forward rates, inflation swaps) are Bloomberg data as published by the Commission.
- Eurostat, ECB Data Portal, OECD (historical series for the stochastic projections, debt structure, benchmark rates) and the ESM repayment database, used with acknowledgement of the source.
- Bond-level repayment data (`BOND_REPAYMENT`, legacy workbook only) from Refinitiv/Eikon; bond-level data from Bloomberg or Refinitiv can be supplied by the user.

The `Sources` sheet and the country sheets of each input workbook document the source of every value.

## Methodology

### Deterministic Debt Projections

The starting point for the DSA methodology is the European Commission’s Debt Sustainability Monitor (DSM). Annex A3 of the DSM describes debt dynamics and the projection of implicit interest rate on government debt. The debt ratio in a given year, $`d_t`$, is calculated as:

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

- **‘Lower SPB’ scenario**: after the adjustment period, the SPB is assumed to be reduced by 0.5 pp. of GDP in total, in equal steps over two years (three years for a 7-year adjustment period), and to remain at that level afterwards (apart from changes in the cost of ageing – see below).
- **‘Adverse r-g’ scenario**: the interest/growth-rate differential is assumed to be permanently increased by 1 percentage point (interest rates 0.5 pp. higher, growth 0.5 pp. lower).
- **‘Financial stress’ scenario**: market interest rates are assumed to temporarily increase for one year by 1 pp., plus a risk premium for high-debt countries.

These adverse scenarios are assumed for ten years after the end of the adjustment period. The DSA criterion requires the public debt to GDP ratio to decline under these adverse scenarios.

#### Data Sources

The reference year $`T`$ is the year of the Commission forecast vintage, as in the Commission prior guidance. Forecast data (AMECO) are available up to $`T+1`$ (spring forecast) or $`T+2`$ (autumn forecast); later years are projected.

- Shares of debt by currency, of short-term debt and of maturing debt are based on ECB data.
- Exchange rates are taken from the Commission forecast and assumed to remain constant afterwards.
- Stock-flow adjustments are taken from the Commission forecast. After the forecast, they are zero except for country-specific paths from the Debt Sustainability Monitor 2025 country fiches (e.g. pension fund balances in Finland and Luxembourg).
- Nominal GDP growth, the primary balance, and the implicit interest rate on government debt are endogenous model variables. They build on the Commission forecast, medium-term real and potential growth projections of the Output Gaps Working Group (if supplied; otherwise the output gap closes by $`T+5`$), long-term growth and ageing-cost projections from the 2024 Ageing Report, market expectations for inflation and interest rates (Bloomberg, as used in the Commission prior guidance), a fiscal multiplier of 0.75 based on Carnot and de Castro (2015), and budget balance semi-elasticities based on Mourre et al. (2019).

The projection of the implicit interest rate on government debt further relies on ECB data on government debt stocks, shares of short- and long-term debt issuance, and average annual debt redemption, as well as market expectations for interest rates from Bloomberg. All data sources are documented in the `Sources` sheet and the country sheets of the input workbook (see [Input data](#input-data)).

#### Projecting Nominal Growth

The effect of fiscal stimulus and the cyclical dependence of the budget balance makes growth and primary balance projections mutually dependent. These dependencies affect the variables from the beginning of the adjustment period in $`T+1`$. In $`T`$, the model relies directly on the Commission forecast for the primary balance and nominal growth. From $`T+1`$, real growth is affected by annual adjustments of the structural primary balance. Specifically, in a given year, the effect of the fiscal multiplier effect is proportional to annual adjustments in the structural primary balance relative to its baseline trajectory:

```math
m_t = 0.75 \times (\Delta \text{spb}_t - \Delta \text{spb}_t^{BL}) 
```

Here, 0.75 is the fiscal multiplier of Carnot and de Castro (2015) and $`\Delta \text{spb}_t^{BL}`$ denotes the annual change in baseline structural primary balance, which is based on the Commission forecast and held constant thereafter. By default, the multiplier $`m_t`$ affects real growth via its persistent effect on the output gap, narrowing the output gap by $`m_t`$ in the year of the adjustment $`t`$, and reducing its impact by one-third of its initial effect in the two consecutive periods. Thus, the total impact on the output gap in a particular year is the sum of the impact in that year plus 2/3 of the impact from the previous year plus 1/3 of the impact from two years before. With `fiscal_multiplier_type='ec'`, the model instead applies the output gap closure rule of the Commission prior guidance sheets (which use a multiplier of 0.6 in guidance issued from 2025).

For euro area countries, Bulgaria, Czechia, Denmark, and Sweden, inflation numbers used to compute nominal growth rates are based on the Commission forecast (GDP deflator), which are linearly interpolated with market expectations for $`T+10`$ implied by euro area inflation swaps, before converging to the 2 percent inflation targets of these countries by $`T+30`$, in line with the Commission’s methodology. For Hungary, Poland, and Romania, where the central banks have higher than 2 percent inflation targets, the Commission’s methodology assumes that half of the spread vis-à-vis euro area inflation at the end of the forecast remains by $`T+10`$, which in turn gradually converges to the national inflation targets by $`T+30`$. The $`T+10`$ and $`T+30`$ values are parameters of the input workbook.

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

where $`\beta_{t-1}`$ is the share of new long-term debt issuance in total long-term debt stock in $`t-1`$. Long-term market rates are projected by linearly interpolating from 10-year government bond benchmark rates (ECB) to 10Y10Y forward rates in $`T+10`$ (Bloomberg, as used in the Commission prior guidance). Between $`T+10`$ and $`T+30`$, long-term market rates converge linearly to 2 percent plus national inflation targets, which yields 4.5 percent for Poland and Romania, 5 percent for Hungary, and 4 percent for all other countries. Short-term market rates are calculated using 3 months benchmark rates, 3M10Y forward rates, and 0.5 times the country-specific values for the long-term rate in $`T+30`$.

To project the implicit interest rate forward, we calculate the new issuance and total stock of short-term and long-term debt in each period $`t`$. Gross financing needs, i.e. the size of new issuance, are the sum of all interest and amortization payments, and the primary balance. Here, interest on short-term debt is the product of short-term market rates and the stock of short-term debt in $`t-1`$. Interest on long-term debt is the product of the implied interest rate on long-term debt $`iir_t^{LT}`$ and the long-term debt stock in $`t-1`$. Short-term debt is redeemed entirely each period. The share of long-term debt maturing each year starts at the share of long-term debt with maturity below one year in total long-term debt in the latest ECB data and converges by $`T+10`$ to its historical average (ECB). Loans from the ESM/EFSF follow their repayment schedule. Given gross financing needs, the share of newly issued short- and long-term debt is calculated such that the share of short-term debt in total debt is held constant. The resulting debt issuances and stocks in period $`t`$ are then used to calculate the implicit interest rate in $`t+1`$

### Stochastic Debt Projections

Stochastic projections of the debt ratio are based on Annex A4 of the DSM. This approach involves drawing multiple shock series from a joint normal distribution of historical quarterly shocks for the primary balance, nominal short- and long-term interest rates, nominal GDP growth, and the exchange rate. After transforming these shocks to annual frequency and constructing the shocks to the implicit interest rate, each series is combined with the projected deterministic path of the respective variable. By recalculating the debt ratio path for each draw using the debt equation above, we obtain the probability distribution of debt ratio projections. The distribution is based on 100,000 draws by default (`simulate(N=...)`).

The Commission’s methodology assumes no shocks during the adjustment period. Stochastic shocks are simulated for 5 years after the end of the adjustment period, and the DSA criterion requires the public debt to GDP ratio to decline with a 70 percent probability over these five years.

#### Definition of Historical Shocks

Quarterly shocks are defined as the first differences in the historical quarterly time series. We correct for outliers by replacing observations that fall outside the 5th and 95th percentiles with the respective thresholds. Historical series are collected from the sources listed in Table A4.1 of the DSM: quarterly series for exchange rates, nominal GDP growth, long-term interest rates and the primary balance from Eurostat, short-term interest rates from Eurostat (euro area) and the OECD (other countries), and the ECB for Estonian long-term rates. These data sources are documented in the `Sources` sheet of the input workbook.

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
