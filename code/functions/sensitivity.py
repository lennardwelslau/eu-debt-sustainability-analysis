# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Sensitivity Analysis     #
# ========================================================================================= #
#
# Helper functions for code/exercises/sensitivity_analysis.ipynb. The exercise runs the model
# (StochasticDsaModel.find_spb_binding) under alternative data and assumptions and records how the SPB
# target responds. It has four parts:
#
# 1. Specifications: each check is one spec, a dict with model keyword arguments, model attributes,
#    input overrides (relative to the baseline data), find_spb_binding keyword arguments and optional
#    setup functions applied after initialisation (build_oat_specs, build_menu_specs, build_global_specs). Settings
#    are grouped in blocks by what settles them (BLOCKS): 1. forecasts that set the centre line, with ranges calibrated
#    to historical forecast errors and Ageing Report scenarios (CALIBRATED_RANGES, calibration_table); 2. model
#    parameters, with ranges from the literature (LITERATURE_RANGES); 3. estimation of risk; 4. risk standard, shown as
#    a menu with prices (build_menu_specs); and data and definitions. Blocks 1 and 2 are drawn jointly in the global
#    analysis (JOINT_RUNS).
# 2. Runner: run_specs runs specs for countries and adjustment periods in parallel (ProcessPoolExecutor)
#    and caches results by task, so reruns only compute missing tasks.
# 3. Results: one tidy DataFrame with one row per spec, country and adjustment period, deviations from the
#    baseline in three tiers (own criterion, DSA target, binding target; add_own_criterion), noise band from
#    alternative seeds, tables of effects by check and of the prices of the risk standard, accommodative and
#    restrictive combinations from the global draws and an optional variance decomposition.
# 4. Charts: mean effect by check, range of targets by country, debt paths by country.
#
# Setup functions must be defined at module level (not in a notebook) so that specs can be sent to worker
# processes; use functools.partial to pass arguments.
#
# Author: Lennard Welslau
# Updated: 2026-10-05
# ========================================================================================= #

import contextlib
import functools
import io
import json
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_pipeline import read_country, latest_input_file, resolve_input_path, REPO_ROOT, PARAMETERS

OUTPUT_DIR = REPO_ROOT / 'output' / 'sensitivity'

# Colours: matplotlib default cycle, black for the baseline, grey for ranges
GREY, LIGHT_GREY, INK, MUTED = 'grey', 'lightgrey', 'black', 'dimgrey'
BASELINE, ACCOMMODATIVE, RESTRICTIVE, COMMISSION, CURRENT = 'black', 'C2', 'C3', 'C1', 'grey'
TARGET_LABELS = {'dsa': 'DSA criteria', 'binding': 'DSA, EDP and safeguards'}
# Default upper bound of the annual adjustment in find_spb_binding (pp. per year, FiscalRules._find_spb_binding_rules);
# runs can set a higher bound with the model attribute adjustment_bound (run_specs(attributes=...))
SEARCH_BOUND = 3


# ========================================================================================= #
#                                     SPECIFICATIONS                                        #
# ========================================================================================= #

def make_spec(id, group, check, variant, value=np.nan, base_value=np.nan, knob=None, model_kwargs=None,
              attributes=None, overrides=None, binding_kwargs=None, setup=None, input_file=None, seed=None,
              params=None):
    """
    One sensitivity check.

    Parameters:
        id (str): Unique identifier, e.g. 'macro:INTEREST_RATE_LT_T10:+1'.
        group, check, variant (str): Labels (e.g. 'Macro', 'Long-term rate T+10', '+1').
        value (float): Numeric value of the variant (x-axis of response curves), NaN if not numeric.
        base_value (float or str): Value of the baseline on the same scale, or the name of a recorded result column
            (e.g. 'fiscal_multiplier') if it differs by country.
        knob (str): Setting changed by the spec; specs with the same knob are alternatives.
        model_kwargs (dict): Keyword arguments of StochasticDsaModel. 'adjustment_start_delay' (years) shifts the
            adjustment start relative to the input file reference year.
        attributes (dict): Model attributes set after initialisation (e.g. {'adverse_r_g_shock': 1.0}).
        overrides (list of dict): Changes to the input data relative to the baseline data, with 'code', 'years'
            ('T': reference year, 'T+': reference year and following forecast years, 'after': all years after T,
            ignored for parameters) and one of 'delta' (added), 'scale' (multiplied) or 'value' (replaced). With
            years='values', 'values' gives country-specific values: {ISO3: value} for parameters, {ISO3: {year: value}}
            for series (countries not listed keep the baseline data).
        binding_kwargs (dict): Keyword arguments of find_spb_binding (e.g. {'rules': 'default'}, {'stochastic': False}).
        setup (list): Functions applied to the model after initialisation (module level, see header).
        input_file (str): Input workbook replacing the baseline workbook.
        seed (int): Random seed replacing the baseline seed.
        params (dict): Parameter values of global sensitivity draws.
    """
    return {
        'id': id, 'group': group, 'check': check, 'variant': variant, 'value': value, 'base_value': base_value,
        'knob': knob or check, 'model_kwargs': model_kwargs or {}, 'attributes': attributes or {},
        'overrides': overrides or [], 'binding_kwargs': binding_kwargs or {}, 'setup': list(setup or []),
        'input_file': input_file, 'seed': seed, 'params': params or {},
    }


def baseline_spec(stochastic=True):
    """
    Baseline: input workbook, rules and seed of the run.
    """
    if stochastic:
        return make_spec('baseline', 'Baseline', 'Baseline', 'baseline')
    return make_spec('baseline_deterministic', 'Baseline', 'Baseline (deterministic)', 'baseline',
                     binding_kwargs={'stochastic': False})


# Setup functions ----------------------------------------------------------------------- #

def shift_potential_growth(model, delta):
    """
    Shift potential growth and baseline real growth in all projection years after T by delta (pp.), so that the
    baseline output gap is unchanged. (Shifting potential growth alone would open a growing output gap where the input
    file provides baseline real GDP for all years, as the Commission workbooks do.)
    """
    for t in range(1, model.projection_period):
        model.rg_pot_bl[t] += delta
        model.rg_bl[t] += delta
        model.rgdp_pot_bl[t] = model.rgdp_pot_bl[t - 1] * (1 + model.rg_pot_bl[t] / 100)
        model.rgdp_bl[t] = model.rgdp_bl[t - 1] * (1 + model.rg_bl[t] / 100)
    model.output_gap_bl = (model.rgdp_bl / model.rgdp_pot_bl - 1) * 100
    model.rg_pot, model.rgdp_pot = model.rg_pot_bl.copy(), model.rgdp_pot_bl.copy()
    model.rg, model.rgdp, model.output_gap = model.rg_bl.copy(), model.rgdp_bl.copy(), model.output_gap_bl.copy()


def shift_anchor(model, variable, anchor, delta):
    """
    Shift the path of a market rate or inflation ('i_st', 'i_lt' or 'pi') as if its T+10 or T+30 anchor (anchor=10 or
    30) were higher by delta (pp.), with linear interpolation as in DsaModel._clean_market_rates: for the T+10 anchor,
    the shift rises from zero in the last forecast year to delta in T+10 and falls back to zero by T+30; for the T+30
    anchor, it rises from zero in T+10 to delta in T+30 and stays there. Works whether the input file provides the
    anchors as parameters or full paths (as the Commission workbooks do, where the anchor parameters are not used).
    """
    t = np.arange(model.projection_period, dtype=float)
    t0 = model.last_data
    if anchor == 10:
        path = np.interp(t, [t0, 10, 30], [0, delta, 0])
    elif anchor == 30:
        path = np.interp(t, [10, 30], [0, delta])
    else:
        raise ValueError('anchor must be 10 or 30')
    if variable == 'pi':
        model.pi = model.pi + path
        for s in range(model.last_data + 1, model.projection_period):
            model.ng_bl[s] = (1 + model.rg_bl[s] / 100) * (1 + model.pi[s] / 100) * 100 - 100
            model.ngdp_bl[s] = model.ngdp_bl[s - 1] * (1 + model.ng_bl[s] / 100)
        model.ng, model.ngdp = model.ng_bl.copy(), model.ngdp_bl.copy()
    else:
        setattr(model, f'{variable}_bl', getattr(model, f'{variable}_bl') + path)
        setattr(model, variable, getattr(model, f'{variable}_bl').copy())


def shift_potential_growth_t10(model, delta):
    """
    Shift potential growth and baseline real growth with the profile of a T+10 anchor (as shift_anchor): the shift rises
    from zero in the last forecast year to delta (pp.) in T+10 and falls back to zero by T+30. Real growth moves with
    potential growth, so that the baseline output gap is unchanged.
    """
    path = np.interp(np.arange(model.projection_period, dtype=float), [model.last_data, 10, 30], [0, delta, 0])
    for t in range(1, model.projection_period):
        model.rg_pot_bl[t] += path[t]
        model.rg_bl[t] += path[t]
        model.rgdp_pot_bl[t] = model.rgdp_pot_bl[t - 1] * (1 + model.rg_pot_bl[t] / 100)
        model.rgdp_bl[t] = model.rgdp_bl[t - 1] * (1 + model.rg_bl[t] / 100)
    model.output_gap_bl = (model.rgdp_bl / model.rgdp_pot_bl - 1) * 100
    model.rg_pot, model.rgdp_pot = model.rg_pot_bl.copy(), model.rgdp_pot_bl.copy()
    model.rg, model.rgdp, model.output_gap = model.rg_bl.copy(), model.rgdp_bl.copy(), model.output_gap_bl.copy()


def scale_ageing_cost(model, factor):
    """
    Scale the change in ageing costs relative to T by factor.
    """
    model.ageing_cost = model.ageing_cost[0] + factor * (model.ageing_cost - model.ageing_cost[0])


def shift_ageing_cost(model, delta, horizon=None):
    """
    Add delta (pp. of GDP) to the change in ageing costs between T and T+horizon, rising linearly from zero in T and
    continuing at the same slope afterwards (default horizon AGEING_HORIZON, the year in which the Ageing Report
    scenarios are compared, see ageing_scenario_deviations).
    """
    horizon = horizon or AGEING_HORIZON
    model.ageing_cost = model.ageing_cost + delta * np.arange(len(model.ageing_cost)) / horizon


def use_repayment_profile(model):
    """
    Repayment profile of long-term debt from the input workbook (BOND_REPAYMENT, bond_data=True) where the workbook has
    one; other countries keep the share of long-term debt maturing each year (used in the global draws).
    """
    if model.df_deterministic_data.loc[model.start_year + 1:, 'BOND_REPAYMENT'].notna().any():
        model.bond_data = True
        model._clean_bond_repayment()


def stochastic_start_at_adjustment(model):
    """
    Start the stochastic projection in the first adjustment year instead of the year after the adjustment period.
    Primary balance shocks are zero during the adjustment period (see StochasticDsaModel._draw_shocks_normal).
    """
    model.stochastic_start_year = model.adjustment_start_year
    model.stochastic_start = model.adjustment_start
    model.stochastic_end = model.stochastic_start + model.stochastic_period - 1
    model.draw_period = model.stochastic_period * (4 if model.shock_frequency == 'quarterly' else 1)


# Calibration of the ranges ------------------------------------------------------------ #

# The forecasts that set the centre line of the projection are varied over the interquartile range of historical
# errors (realised minus forecast), the same range for all countries:
# - real GDP growth and CPI inflation: IMF WEO forecasts, average error over t+1 to t+5, autumn vintages 2000-2018,
#   27 EU countries, demeaned by country (the average error, a bias, is reported separately);
# - short- and long-term rates: German zero-coupon curve (Bundesbank), realised 1-year and 10-year rates minus the
#   forward rates priced ten years earlier (the DSA uses the 3M10Y and 10Y10Y forwards), year ends 1986-2015, demeaned;
# - ageing costs: deviation of the 2024 Ageing Report scenarios (all except the pension policy scenarios) from the
#   baseline change in ageing costs between 2024 and 2038 (T+14, the end of the DSA horizon for a 4-year adjustment),
#   all countries and scenarios.
# The forward-rate errors are nominal but shift the real rates; with inflation drawn on top, the range of nominal rates
# is somewhat wider than in the data. In the joint draws, the short- and long-term rates are one factor
# (COMPOSITE_ASSUMPTIONS): their errors come from the same yield curve and are highly correlated.
# Rates, inflation and growth are shifted at the T+10 anchor and converge back to the baseline by T+30 (shift_anchor,
# shift_potential_growth_t10); ageing costs by shift_ageing_cost. calibration_table() recomputes the ranges from the
# data; CALIBRATED_RANGES holds the rounded values used in the checks, so that run caches do not depend on data
# revisions.

REFERENCE_DIR = REPO_ROOT / 'data' / 'RawData' / 'reference'
WEO_FILE = REFERENCE_DIR / 'WEO-MacroForecasts-042024.xlsx'
WEO_URL = ('https://data.mendeley.com/public-files/datasets/8dt6xpp6x4/files/8c41f201-1546-4374-bf79-1aab66bf056e/'
           'file_downloaded')  # IMF WEO macroeconomic forecasts panel dataset (Mendeley Data, version 6, CC BY-NC)
BUNDESBANK_URL = ('https://api.statistiken.bundesbank.de/rest/data/BBSIS/'
                  'M.I.ZST.ZI.EUR.S1311.B.A604.R{m:02d}XX.R.A.A._Z._Z.A?format=csv&lang=en')  # zero-coupon rates, m years
WEO_NAMES = {
    'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia', 'CYP': 'Cyprus', 'CZE': 'Czech Republic',
    'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
    'HUN': 'Hungary', 'IRL': 'Ireland', 'ITA': 'Italy', 'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
    'MLT': 'Malta', 'NLD': 'Netherlands', 'POL': 'Poland', 'PRT': 'Portugal', 'ROU': 'Romania', 'SVK': 'Slovak Republic',
    'SVN': 'Slovenia', 'ESP': 'Spain', 'SWE': 'Sweden'}
WEO_VINTAGES = (2000, 2018)    # autumn WEO vintages with five realised forecast years
FORWARD_YEARS = (1986, 2015)   # year ends with forward rates ten years ahead and realised rates
AGEING_BASE_YEAR, AGEING_HORIZON = 2024, 14
AGEING_SCENARIOS = {  # Ageing Report 2024 statistical annex: total cost of ageing, % of GDP
    'baseline': 'Table II.1.135',
    'Higher life expectancy (+2 years)': 'Table II.1.137',
    'Higher migration (+33%)': 'Table II.1.138',
    'Lower migration (-33%)': 'Table II.1.139',
    'Lower fertility (-20%)': 'Table II.1.140',
    'Higher employment rate of older workers (+10 pp.)': 'Table II.1.141',
    'Higher TFP growth (+0.2 pp.)': 'Table II.1.142',
    'Lower TFP growth (-0.2 pp.)': 'Table II.1.143',
    'Risk scenario (health care and long-term care)': 'Table II.1.136',
}

# Ranges used in the checks: interquartile range of the errors (calibration_table, rounded to 0.01)
CALIBRATED_RANGES = {
    'rate_st_T10': (-1.22, 0.40),
    'rate_lt_T10': (-1.24, 0.89),
    'inflation_T10': (-0.74, 0.64),
    'potential_growth': (-0.84, 1.08),
    'ageing_shift': (-0.24, 0.14),
}


def weo_forecast_errors(path=WEO_FILE):
    """
    Average error (realised minus forecast, pp.) of the IMF WEO forecasts of real GDP growth and CPI inflation over the
    five years after each autumn vintage (WEO_VINTAGES), by EU country and vintage. Downloads the dataset if missing.
    """
    if not os.path.exists(path):
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(requests.get(WEO_URL, timeout=300).content)
    d = pd.read_excel(path, sheet_name='Dataset')
    d = d[d['Country'].isin(WEO_NAMES.values()) & (d['exercise'] == 2) & d['h'].between(2, 6)
          & d['weo_year'].between(*WEO_VINTAGES)].copy()
    d['growth'] = d['Rngdp_rpc'] - d['ngdp_rpc']
    d['inflation'] = d['Rpcpi_pch'] - d['pcpi_pch']
    g = d.groupby(['Country', 'weo_year'])
    e = g[['growth', 'inflation']].mean()[g[['growth', 'inflation']].count().min(axis=1) == 5]
    e.index = e.index.set_levels(e.index.levels[0].map({v: k for k, v in WEO_NAMES.items()}), level=0)
    return e.rename_axis(['country', 'vintage'])


def forward_rate_errors():
    """
    Errors of German forward rates as forecasts (realised minus forward, pp.), by year end (FORWARD_YEARS): the 1-year
    rate and the 10-year rate realised ten years later minus the forward rates priced at the year end (from the
    Bundesbank zero-coupon curve, annual compounding).
    """
    import requests
    spot = {}
    for m in (1, 10, 11, 20):
        rows = [line.split(',')[:2] for line in requests.get(BUNDESBANK_URL.format(m=m), timeout=120).text.splitlines()
                if line[:4].isdigit() and line[4:5] == '-']
        s = pd.Series({a: float(b) for a, b in rows if b not in ('', '.')})
        spot[m] = s[s.index.str.endswith('-12')].rename(lambda x: int(x[:4]))
    s = pd.DataFrame(spot) / 100
    forward = lambda a, b: (((1 + s[b]) ** b / (1 + s[a]) ** a) ** (1 / (b - a)) - 1) * 100
    e = pd.DataFrame({'rate_st': s[1].shift(-10) * 100 - forward(10, 11), 'rate_lt': s[10].shift(-10) * 100 - forward(10, 20)})
    return e.loc[FORWARD_YEARS[0]:FORWARD_YEARS[1]].dropna().rename_axis('year')


def ageing_scenario_deviations():
    """
    Deviation of the change in the total cost of ageing between AGEING_BASE_YEAR and AGEING_BASE_YEAR + AGEING_HORIZON
    in the Ageing Report 2024 scenarios (AGEING_SCENARIOS) from the baseline change (pp. of GDP), by country
    (rows) and scenario (columns).
    """
    import openpyxl
    from data_pipeline.sources import download_reference_files
    from data_pipeline.commission import ISO3_FROM_EC
    ws = openpyxl.load_workbook(download_reference_files()['awg'], read_only=True, data_only=True)['cross-country_tables']
    rows = list(ws.iter_rows(values_only=True))
    tables = {}
    for name, title in AGEING_SCENARIOS.items():
        start = next(i for i, r in enumerate(rows) if any(isinstance(v, str) and v.strip().startswith(title) for v in r))
        header, data = None, {}
        for r in rows[start + 1:]:
            if header is None:
                if any(isinstance(v, (int, float)) and 2000 < v < 2100 for v in r):
                    header = {j: int(v) for j, v in enumerate(r) if isinstance(v, (int, float)) and 2000 < v < 2100}
                continue
            label = next((v.strip() for v in r if isinstance(v, str)), None)
            if label is None or label in ('EA', 'EU'):
                break
            if label in ISO3_FROM_EC:
                data[ISO3_FROM_EC[label]] = {y: r[j] for j, y in header.items() if isinstance(r[j], (int, float))}
        tables[name] = pd.DataFrame(data)
    y0, y1 = AGEING_BASE_YEAR, AGEING_BASE_YEAR + AGEING_HORIZON
    change = {k: t.loc[y1] - t.loc[y0] for k, t in tables.items()}
    return pd.DataFrame({k: v - change['baseline'] for k, v in change.items() if k != 'baseline'})


def calibration_table(weo=None, forwards=None, ageing=None):
    """
    Ranges of the forecasts that set the centre line (CALIBRATED_RANGES), recomputed from the data: source, sample,
    number of errors, average error (bias, not used) and 25th and 75th percentile of the errors (demeaned by country
    for the WEO forecasts, demeaned for the forward rates; scenario deviations for ageing costs), with the range used.
    """
    weo = weo_forecast_errors() if weo is None else weo
    forwards = forward_rate_errors() if forwards is None else forwards
    ageing = ageing_scenario_deviations() if ageing is None else ageing
    v0, v1 = WEO_VINTAGES
    f0, f1 = FORWARD_YEARS
    rows = []
    for name, x, demeaned, source, sample in [
            ('rate_st_T10', forwards['rate_st'], forwards['rate_st'] - forwards['rate_st'].mean(),
             'Bundesbank: 1-year rate in ten years minus forward rate', f'Germany, year ends {f0}-{f1}'),
            ('rate_lt_T10', forwards['rate_lt'], forwards['rate_lt'] - forwards['rate_lt'].mean(),
             'Bundesbank: 10-year rate in ten years minus 10Y10Y forward rate', f'Germany, year ends {f0}-{f1}'),
            ('inflation_T10', weo['inflation'], weo['inflation'] - weo['inflation'].groupby(level=0).transform('mean'),
             'IMF WEO: CPI inflation, average error over five years', f'EU27, autumn vintages {v0}-{v1}'),
            ('potential_growth', weo['growth'], weo['growth'] - weo['growth'].groupby(level=0).transform('mean'),
             'IMF WEO: real GDP growth, average error over five years', f'EU27, autumn vintages {v0}-{v1}'),
            ('ageing_shift', ageing.stack(), ageing.stack(),
             f'Ageing Report 2024: scenarios except pension policy, change in ageing costs {AGEING_BASE_YEAR}-'
             f'{AGEING_BASE_YEAR + AGEING_HORIZON}', f'EU27, {ageing.shape[1]} scenarios')]:
        rows.append({'assumption': name, 'source': source, 'sample': sample, 'n': int(x.count()), 'bias': x.mean(),
                     'p25': demeaned.quantile(0.25), 'p75': demeaned.quantile(0.75),
                     'range_low': CALIBRATED_RANGES[name][0], 'range_high': CALIBRATED_RANGES[name][1]})
    t = pd.DataFrame(rows).set_index('assumption')
    t.loc['ageing_shift', 'bias'] = np.nan  # scenario deviations, not forecast errors
    return t


# Ranges of the model parameters, from the literature (values of the individual checks; the global analysis draws
# uniformly between the lowest and highest value, persistence with equal probability for each value):
# - fiscal multiplier (first-year output effect of a change in the SPB; baseline 0.75, Carnot and de Castro 2015):
#   0.5 to 1.5. Most estimates of spending multipliers lie between 0.6 and 1 (Ramey 2019; Ramey and Zubairy 2018;
#   Barro and Redlick 2011: 0.4-0.5 on impact); multipliers of the consolidations in Europe after 2010 were about 1.5
#   (Blanchard and Leigh 2013: about 1 above the 0.5 forecasters assumed), and multipliers without a monetary policy
#   response, as for one country in a monetary union, are 1.5 or more (Nakamura and Steinsson 2014; Chodorow-Reich 2019).
# - persistence of the multiplier effect (baseline 3 years): 2 to 5 years. Two-year cumulative multipliers are at
#   least as large as impact multipliers (Barro and Redlick 2011; Ramey and Zubairy 2018), and consolidations can have
#   lasting effects on output (Fatas and Summers 2018; Gechert, Horn and Paetz 2019).
# - budget semi-elasticity (scale): 0.9 to 1.1, the size of the largest revisions of the Commission estimates between
#   the 2014 and 2018 updates (-0.06 to +0.05 on an average of 0.55; Mourre et al. 2019).
LITERATURE_RANGES = {
    'fiscal_multiplier': (0.5, 1.0, 1.5),
    'multiplier_persistence': (2, 3, 4, 5),
    'elasticity_scale': (0.9, 1.1),
}


# Individual checks ------------------------------------------------------------------ #

# Continuous assumptions: name -> (group, check label, function value -> spec fields, baseline value, OAT values).
# The global sensitivity analysis draws the same assumptions over the range of the OAT values.
def _override(code, years, **change):
    return {'overrides': [{'code': code, 'years': years, **change}]}


def _anchor(variable, anchor, delta):
    return {'setup': [functools.partial(shift_anchor, variable=variable, anchor=anchor, delta=delta)]}


def _inflation_real_rates(anchor, delta):
    """
    Inflation anchor shifted by delta with real interest rates unchanged: the short- and long-term nominal rates move
    with inflation (real_rates=True in build_oat_specs and build_global_specs).
    """
    return {'setup': [functools.partial(shift_anchor, variable=v, anchor=anchor, delta=delta) for v in ('pi', 'i_st', 'i_lt')]}


ASSUMPTIONS = {
    # Forecasts that set the centre line: interquartile range of historical errors (CALIBRATED_RANGES), shifted at the
    # T+10 anchor and converging back to the baseline by T+30
    'rate_st_T10': ('Macro', 'Short-term rate T+10', lambda v: _anchor('i_st', 10, v), 0, list(CALIBRATED_RANGES['rate_st_T10'])),
    'rate_lt_T10': ('Macro', 'Long-term rate T+10', lambda v: _anchor('i_lt', 10, v), 0, list(CALIBRATED_RANGES['rate_lt_T10'])),
    'inflation_T10': ('Macro', 'Inflation T+10', lambda v: _anchor('pi', 10, v), 0, list(CALIBRATED_RANGES['inflation_T10'])),
    'potential_growth': ('Macro', 'Potential growth T+10', lambda v: {'setup': [functools.partial(shift_potential_growth_t10, delta=v)]}, 0, list(CALIBRATED_RANGES['potential_growth'])),
    'ageing_shift': ('Macro', 'Ageing costs (shift)', lambda v: {'setup': [functools.partial(shift_ageing_cost, delta=v)]}, 0, list(CALIBRATED_RANGES['ageing_shift'])),
    # Long-run anchors (no evidence on errors 30 years ahead; individual checks only) and earlier specifications
    'rate_st_T30': ('Macro', 'Short-term rate T+30', lambda v: _anchor('i_st', 30, v), 0, [-0.5, 0.5]),
    'rate_lt_T30': ('Macro', 'Long-term rate T+30', lambda v: _anchor('i_lt', 30, v), 0, [-0.5, 0.5]),
    'inflation_T30': ('Macro', 'Inflation T+30', lambda v: _anchor('pi', 30, v), 0, [-0.25, 0.25]),
    'potential_growth_all': ('Macro', 'Potential growth (all years)', lambda v: {'setup': [functools.partial(shift_potential_growth, delta=v)]}, 0, [-0.5, 0.5]),
    'ageing_scale': ('Macro', 'Ageing cost change (scale)', lambda v: {'setup': [functools.partial(scale_ageing_cost, factor=v)]}, 1, [0.5, 1.5]),
    # Model parameters: ranges from the literature (LITERATURE_RANGES)
    'elasticity_scale': ('Macro', 'Budget balance elasticity (scale)', lambda v: _override('BUDGET_BALANCE_ELASTICITY', None, scale=v), 1, list(LITERATURE_RANGES['elasticity_scale'])),
    'fiscal_multiplier': ('Multiplier', 'Fiscal multiplier', lambda v: {'model_kwargs': {'fiscal_multiplier': v}}, 'fiscal_multiplier', list(LITERATURE_RANGES['fiscal_multiplier'])),
    'adverse_r_g_shock': ('Stress tests', 'Adverse r-g shock', lambda v: {'attributes': {'adverse_r_g_shock': v}}, 0.5, [0.25, 0.75]),
    'financial_stress_shock': ('Stress tests', 'Financial stress shock', lambda v: {'attributes': {'financial_stress_shock': v}}, 1.0, [0.5, 1.5]),
    'lower_spb_shock': ('Stress tests', 'Lower SPB shock', lambda v: {'attributes': {'lower_spb_shock': v}}, 0.5, [0.25, 0.75]),
}

# Discrete settings drawn in the global sensitivity analysis: name -> (label, [(value, spec fields)]), the baseline
# setting included. Values are numbers (used as such in the variance decomposition) or strings (categories).
DISCRETE_ASSUMPTIONS = {
    'ageing_cost_period': ('Ageing cost period', [(p, {'model_kwargs': {'ageing_cost_period': p}}) for p in [5, 10]]),
    'stock_flow_zero': ('Stock-flow adjustment zero after T', [
        (0, {}), (1, _override('STOCK_FLOW_RATIO', 'after', value=0))]),
    'repayment_profile': ('Repayment profile (Eurostat)', [(0, {}), (1, {'setup': [use_repayment_profile]})]),
    'multiplier_persistence': ('Multiplier persistence', [
        (p, {'model_kwargs': {'fiscal_multiplier_persistence': p}}) for p in LITERATURE_RANGES['multiplier_persistence']]),
    'prob_target': ('Probability target', [(p, {'attributes': {'prob_target': p}}) for p in [0.6, 0.7, 0.8]]),
    'stochastic_period': ('Stochastic period', [(p, {'model_kwargs': {'stochastic_period': p}}) for p in [5, 10]]),
    'stochastic_start': ('Stochastic start in first adjustment year', [
        (0, {}), (1, {'setup': [stochastic_start_at_adjustment]})]),
    'shock_sample_start': ('Shock sample start', [(y, {'model_kwargs': {'shock_sample_start': y}}) for y in [1990, 2000, 2010]]),
    'shock_frequency_annual': ('Annual shock frequency', [(0, {}), (1, {'model_kwargs': {'shock_frequency': 'annual'}})]),
    'var_bootstrap': ('Shock estimation: VAR bootstrap', [(0, {}), (1, {'model_kwargs': {'estimation': 'var_bootstrap'}})]),
    'winsorize_off': ('Winsorised shocks off', [(0, {}), (1, {'model_kwargs': {'winsorize_sample': False}})]),
}

# Blocks of settings, by what settles them (group labels of the individual checks):
# 1. Forecasts that set the centre line of the projection (rates, inflation, potential growth, ageing costs): settled by
#    evidence; ranges from historical forecast errors and Ageing Report scenarios (CALIBRATED_RANGES). The long-run
#    anchors (T+30) are plain checks.
# 2. Model parameters (fiscal multiplier, its persistence, budget semi-elasticity): settled by the literature
#    (LITERATURE_RANGES). Blocks 1 and 2 are drawn jointly in the global analysis (JOINT_RUNS).
# 3. Estimation of risk in the stochastic analysis (start and horizon of the stochastic projection, shock sample,
#    frequency, distribution, winsorisation): settled by statistical practice; individual checks.
# 4. Risk standard (probability threshold, stress-test sizes, ageing cost period): no true value; a menu with the price
#    of each choice (build_menu_specs) and the lenient and strict CALIBRATION_SCENARIOS.
# Data and definitions: data update, stock-flow adjustments after T (non-zero after the forecast years only for
# Finland, Luxembourg and Greece, a question of gross versus net debt) and the repayment profile of debt.
BLOCKS = {
    'Forecasts': '1. Forecasts (centre line)',
    'Parameters': '2. Model parameters',
    'Risk estimation': '3. Estimation of risk',
    'Risk standard': '4. Risk standard',
    'Data': 'Data and definitions',
}
FORECASTS, MODEL_PARAMETERS, ESTIMATION, RISK_STANDARD, DATA = BLOCKS  # PARAMETERS is the list of parameter codes (data_pipeline)

FORECAST_ASSUMPTIONS = ['rates_T10', 'inflation_T10', 'potential_growth', 'ageing_shift']
PARAMETER_ASSUMPTIONS = ['fiscal_multiplier', 'multiplier_persistence', 'elasticity_scale']
ESTIMATION_SETTINGS = ['stochastic_start', 'stochastic_period', 'shock_sample_start', 'shock_frequency_annual',
                       'var_bootstrap', 'winsorize_off']
RISK_STANDARD_SETTINGS = ['prob_target', 'adverse_r_g_shock', 'financial_stress_shock', 'lower_spb_shock',
                          'ageing_cost_period']

# Global analysis: blocks 1 and 2 drawn jointly, all other settings at the baseline
JOINT_ASSUMPTIONS = FORECAST_ASSUMPTIONS + PARAMETER_ASSUMPTIONS
JOINT_RUNS = {'joint': JOINT_ASSUMPTIONS}
GLOBAL_ASSUMPTIONS = JOINT_ASSUMPTIONS

# Categories of the variance decomposition (optional): by block, and by input
GLOBAL_CATEGORIES = {'Forecasts': FORECAST_ASSUMPTIONS, 'Model parameters': PARAMETER_ASSUMPTIONS}
INPUT_CATEGORIES = {
    'Interest rates': ['rates_T10'],
    'Inflation': ['inflation_T10'],
    'Potential growth': ['potential_growth'],
    'Ageing costs': ['ageing_shift'],
    'Fiscal multiplier': ['fiscal_multiplier', 'multiplier_persistence'],
    'Budget semi-elasticity': ['elasticity_scale'],
}

# Calibration scenarios of the risk standard: all settings at the lenient or at the strict end of their range (the
# ageing cost period cannot be stricter than the baseline of 10 years, which covers the horizon of the DSA criteria)
CALIBRATION_SCENARIOS = {
    'lenient': {'attributes': {'prob_target': 0.6, 'adverse_r_g_shock': 0.25, 'financial_stress_shock': 0.5,
                               'lower_spb_shock': 0.25},
                'model_kwargs': {'ageing_cost_period': 5}},
    'strict': {'attributes': {'prob_target': 0.8, 'adverse_r_g_shock': 0.75, 'financial_stress_shock': 1.5,
                              'lower_spb_shock': 0.75}},
}

# Block of each continuous assumption (group label of its individual checks)
KIND_OF = {name: (RISK_STANDARD if name in RISK_STANDARD_SETTINGS
                  else MODEL_PARAMETERS if name in ('fiscal_multiplier', 'elasticity_scale') else FORECASTS)
           for name in ASSUMPTIONS}


# Assumptions drawn as one factor in the global analysis: name -> (label, members). One draw sets the position in the
# range of each member (ASSUMPTIONS), e.g. both rates at the same quantile of their ranges.
COMPOSITE_ASSUMPTIONS = {
    'rates_T10': ('Short- and long-term rates T+10', ['rate_st_T10', 'rate_lt_T10']),
}


def assumption_label(name):
    """
    Label of an assumption of the global analysis.
    """
    if name in COMPOSITE_ASSUMPTIONS:
        return COMPOSITE_ASSUMPTIONS[name][0]
    return ASSUMPTIONS[name][1] if name in ASSUMPTIONS else DISCRETE_ASSUMPTIONS[name][0]


def _fmt(v):
    return f'{v:+g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def assumption_fields(name, value, real_rates=False):
    """
    Spec fields for one value of a continuous assumption (see ASSUMPTIONS). With real_rates=True, the inflation anchors
    are shifted with real interest rates unchanged (nominal rates move with inflation), so that the rate assumptions are
    real rates.
    """
    if real_rates and name in ('inflation_T10', 'inflation_T30'):
        return _inflation_real_rates(int(name[-2:]), value)
    return ASSUMPTIONS[name][2](value)


def assumption_spec(name, value, real_rates=False, **kwargs):
    """
    Spec for one value of a continuous assumption (see ASSUMPTIONS).
    """
    prefix, check, _, base, _ = ASSUMPTIONS[name]  # the id prefix (e.g. 'macro') keeps ids stable across versions
    variant = f'x{value:g}' if name.endswith('scale') else (f'{value:g}' if not isinstance(base, (int, float)) or base != 0 else _fmt(value))
    return make_spec(f'{prefix.lower().split()[0]}:{name}:{variant}', KIND_OF[name], check, variant, value=value,
                     base_value=base, knob=name, **{**assumption_fields(name, value, real_rates), **kwargs})


def build_oat_specs(latest_file=None, real_rates=False):
    """
    Individual checks (baseline first), grouped by block (BLOCKS): forecasts, model parameters, estimation of risk,
    risk standard (with the lenient and strict CALIBRATION_SCENARIOS) and data and definitions, then the noise band. latest_file: workbook for the data update check (default latest_input_file()). real_rates:
    inflation checks keep real interest rates unchanged (see assumption_fields). Ids do not depend on the grouping, so
    that cached runs remain valid.
    """
    specs = [baseline_spec()]
    spec = make_spec

    # 1. Forecasts that set the centre line (calibrated ranges), then the long-run anchors
    for name in ['rate_st_T10', 'rate_lt_T10', 'inflation_T10', 'potential_growth', 'ageing_shift',
                 'rate_st_T30', 'rate_lt_T30', 'inflation_T30']:
        specs += [assumption_spec(name, v, real_rates=real_rates) for v in ASSUMPTIONS[name][4]]

    # 2. Model parameters (literature ranges)
    specs += [assumption_spec('fiscal_multiplier', v) for v in ASSUMPTIONS['fiscal_multiplier'][4]]
    specs += [spec(f'multiplier:persistence:{p}', MODEL_PARAMETERS, 'Multiplier persistence', f'{p} years', value=p,
                   base_value=3, model_kwargs={'fiscal_multiplier_persistence': p})
              for p in LITERATURE_RANGES['multiplier_persistence'] if p != 3]
    specs += [assumption_spec('elasticity_scale', v) for v in ASSUMPTIONS['elasticity_scale'][4]]

    # 3. Estimation of risk
    specs.append(spec('stochastic:start:adjustment', ESTIMATION, 'Stochastic start', 'first adjustment year',
                      setup=[stochastic_start_at_adjustment]))
    specs.append(spec('stochastic:period:10', ESTIMATION, 'Stochastic period', '10 years', value=10, base_value=5,
                      model_kwargs={'stochastic_period': 10}))
    specs += [spec(f'stochastic:sample_start:{y}', ESTIMATION, 'Shock sample start', str(y), value=y, base_value=2000,
                   model_kwargs={'shock_sample_start': y}) for y in [1990, 2010]]
    specs.append(spec('stochastic:frequency:annual', ESTIMATION, 'Shock frequency', 'annual',
                      model_kwargs={'shock_frequency': 'annual'}))
    specs.append(spec('stochastic:estimation:var_bootstrap', ESTIMATION, 'Shock estimation', 'var_bootstrap',
                      model_kwargs={'estimation': 'var_bootstrap'}))
    specs.append(spec('stochastic:winsorize:off', ESTIMATION, 'Winsorised shocks', 'off',
                      model_kwargs={'winsorize_sample': False}))

    # 4. Risk standard (the full menu is build_menu_specs)
    specs += [spec(f'stochastic:prob_target:{p}', RISK_STANDARD, 'Probability target', f'{p:g}', value=p,
                   base_value=0.7, attributes={'prob_target': p}) for p in [0.6, 0.8]]
    for name in ['adverse_r_g_shock', 'financial_stress_shock', 'lower_spb_shock']:
        specs += [assumption_spec(name, v) for v in ASSUMPTIONS[name][4]]
    specs += [spec(f'macro:ageing_cost_period:{p}', RISK_STANDARD, 'Ageing cost period', f'{p} years', value=p,
                   base_value=10, model_kwargs={'ageing_cost_period': p}) for p in [5]]
    specs += [spec(f'calibration:{k}', RISK_STANDARD, 'Calibration scenario', k, **v)
              for k, v in CALIBRATION_SCENARIOS.items()]

    # Data and definitions. Starting values (SPB, balances, debt in T) are data, not assumptions: they are not varied.
    # The data update shows how the latest data (later reference year, forecasts replaced by outturns) change the
    # targets. Repayment profile of long-term debt from Eurostat debt by residual maturity (BOND_REPAYMENT) instead of
    # the share of long-term debt maturing each year; countries without Eurostat data fail these runs.
    latest_file = latest_file or latest_input_file()
    specs.append(spec('data:latest', DATA, 'Data update',
                      f"latest vintage ({os.path.splitext(os.path.basename(latest_file))[0].replace('dsa_inputs_', '')})",
                      input_file=latest_file))
    specs.append(spec('macro:stock_flow_zero', DATA, 'Stock-flow adjustment', 'zero after T',
                      overrides=[{'code': 'STOCK_FLOW_RATIO', 'years': 'after', 'value': 0}]))
    specs.append(spec('data:repayment_profile', DATA, 'Repayment profile (Eurostat)', 'residual maturity buckets',
                      model_kwargs={'bond_data': True}))

    specs += [spec(f'noise:seed:{s}', 'Noise', 'Random seed', str(s), seed=s) for s in range(1, 10)]
    return specs


# Risk standard: settings without a true value, shown as a menu with the price of each choice (change of the target
# relative to the baseline). The stochastic criterion is a grid of probability threshold, start and horizon of the
# stochastic projection (start and horizon are modelling choices, but they set how much the threshold costs); the
# stress tests and the ageing cost period are varied one at a time.
MENU_PROB = [0.6, 0.7, 0.8]
MENU_START = {'after': ('after adjustment', []), 'adjustment': ('first adjustment year', [stochastic_start_at_adjustment])}
MENU_HORIZON = [5, 10]
MENU_STRESS = {'adverse_r_g_shock': [0.25, 0.75, 1.0], 'financial_stress_shock': [0.5, 1.5, 2.0],
               'lower_spb_shock': [0.25, 0.75, 1.0]}
MENU_AGEING_PERIOD = [5]


def build_menu_specs():
    """
    Menu of the risk standard (baseline first): grid of probability threshold (MENU_PROB), start (MENU_START) and
    horizon (MENU_HORIZON) of the stochastic criterion, stress-test sizes (MENU_STRESS) and ageing cost period
    (MENU_AGEING_PERIOD). Settings equal to the baseline are left out of the spec fields, so that runs of the individual
    checks with the same settings are read from the cache.
    """
    specs = [baseline_spec()]
    for p in MENU_PROB:
        for start, (start_label, setup) in MENU_START.items():
            for h in MENU_HORIZON:
                if (p, start, h) == (0.7, 'after', 5):
                    continue
                specs.append(make_spec(
                    f'menu:stochastic:p{p:g}:{start}:h{h}', RISK_STANDARD, 'Stochastic criterion',
                    f'{p * 100:.0f}%, start {start_label}, {h}-year horizon', knob='stochastic',
                    attributes={'prob_target': p} if p != 0.7 else {},
                    model_kwargs={'stochastic_period': h} if h != 5 else {}, setup=setup))
    for name, values in MENU_STRESS.items():
        base = ASSUMPTIONS[name][3]
        specs += [make_spec(f'menu:{name}:{v:g}', RISK_STANDARD, ASSUMPTIONS[name][1], f'{v:g} pp.', value=v,
                            base_value=base, knob=name, attributes={name: v}) for v in values]
    specs += [make_spec(f'menu:ageing_cost_period:{y}', RISK_STANDARD, 'Ageing cost period', f'{y} years', value=y,
                        base_value=10, knob='ageing_cost_period', model_kwargs={'ageing_cost_period': y})
              for y in MENU_AGEING_PERIOD]
    return specs


def menu_table(df, adjustment_period=4, threshold=0.1):
    """
    Price of each choice of the risk standard (build_menu_specs) in three tiers (TIERS): change of the SPB required by
    the criterion the choice acts on, of the DSA target and of the binding target relative to the baseline; mean across
    countries, number of countries whose target changes by more than threshold (pp.), largest change and the country
    with the largest change.
    """
    if 'own_delta_target' not in df:
        df = add_own_criterion(df)
    d = df[(df['group'] == RISK_STANDARD) & (df['adjustment_period'] == adjustment_period)]
    rows = []
    for (check, variant), g in d.groupby(['check', 'variant'], sort=False):
        row = {'check': check, 'variant': variant}
        for key, (_, col) in TIERS.items():
            x = feasible(g, key).set_index('country')[col].dropna()
            row.update({f'{key} mean': x.mean(), f'{key} countries > {threshold:g} pp.': int((x.abs() > threshold).sum()),
                        f'{key} largest': x.loc[x.abs().idxmax()] if len(x) else np.nan,
                        f'{key} country': x.abs().idxmax() if len(x) else ''})
        rows.append(row)
    return pd.DataFrame(rows).set_index(['check', 'variant'])


def build_global_specs(n_draws=200, seed=0, assumptions=GLOBAL_ASSUMPTIONS, real_rates=False):
    """
    Global sensitivity analysis: n_draws Latin hypercube draws of the given assumptions at once (JOINT_RUNS), with the
    stochastic criterion; all other settings stay at the baseline. Continuous assumptions (ASSUMPTIONS) are drawn uniformly over the range of their values in the individual
    checks, discrete settings (DISCRETE_ASSUMPTIONS) with equal probability for each setting. Composite assumptions
    (COMPOSITE_ASSUMPTIONS) set all their members at the same position in their ranges. The first
    spec is the baseline. The data update and the random seed are not drawn. With real_rates=True, the interest rate
    draws are real rates: nominal rates move with the inflation draws of the same anchor (see assumption_fields).
    """
    from scipy.stats import qmc
    sample = qmc.LatinHypercube(d=len(assumptions), seed=seed).random(n_draws)
    specs = [baseline_spec()]
    for i, u in enumerate(sample):
        fields, params = {'model_kwargs': {}, 'attributes': {}, 'overrides': [], 'setup': [], 'binding_kwargs': {}}, {}
        for name, x in zip(assumptions, u):
            if name in COMPOSITE_ASSUMPTIONS:
                part_fields = {}
                for member in COMPOSITE_ASSUMPTIONS[name][1]:
                    lo, hi = min(ASSUMPTIONS[member][4]), max(ASSUMPTIONS[member][4])
                    params[member] = float(lo + x * (hi - lo))
                    for k, part in assumption_fields(member, params[member], real_rates).items():
                        part_fields[k] = {**part_fields.get(k, {}), **part} if isinstance(part, dict) else part_fields.get(k, []) + part
                v = float(x)  # position in the ranges
            elif name in DISCRETE_ASSUMPTIONS:
                options = DISCRETE_ASSUMPTIONS[name][1]
                v, part_fields = options[min(int(x * len(options)), len(options) - 1)]
            else:
                lo, hi = min(ASSUMPTIONS[name][4]), max(ASSUMPTIONS[name][4])
                v = float(lo + x * (hi - lo))
                part_fields = assumption_fields(name, v, real_rates)
            params[name] = v
            for k, part in part_fields.items():
                fields[k] = {**fields[k], **part} if isinstance(part, dict) else fields[k] + part
        specs.append(make_spec(f'global:{i:03d}', 'Global', 'Global draw', f'{i:03d}', params=params, **fields))
    return specs


def spec_table(specs):
    """
    Specs as a table: one row per check with its model arguments, attributes, input changes, rule options and setup.
    """
    def fmt(d):
        return ', '.join(f'{k}={v}' for k, v in d.items()) if d else ''

    def fmt_override(o):
        if o['years'] == 'values':
            return f"{o['code']}: country-specific values ({len(o['values'])} countries)"
        change = next(f'{k} {v:+g}' if k == 'delta' else f'{k} {v:g}' for k, v in o.items() if k in ('delta', 'scale', 'value'))
        return f"{o['code']} ({o['years']}): {change}" if o['years'] else f"{o['code']}: {change}"

    return pd.DataFrame([{
        'id': s['id'], 'group': s['group'], 'check': s['check'], 'variant': s['variant'],
        'model arguments': fmt(s['model_kwargs']), 'attributes': fmt(s['attributes']),
        'input changes': '; '.join(fmt_override(o) for o in s['overrides']),
        'find_spb_binding options': fmt(s['binding_kwargs']),
        'setup': ', '.join(_setup_label(f) for f in s['setup']),
        'other': ', '.join(x for x in [f"input file {s['input_file']}" if s['input_file'] else '',
                                       f"seed {s['seed']}" if s['seed'] is not None else ''] if x),
    } for s in specs]).set_index('id')


# ========================================================================================= #
#                                           RUNNER                                          #
# ========================================================================================= #

def resolve_overrides(changes, input_file, country, end_year=2070):
    """
    Absolute overrides (CODE, VALUE, YEAR) from changes relative to the baseline data of the input workbook.
    """
    if not changes:
        return None
    series, params = read_country(input_file, country)
    T = int(params['REFERENCE_YEAR'])
    rows = []
    for c in changes:
        code = c['code']
        if c['years'] == 'values':
            v = c['values'].get(country)
            if isinstance(v, dict):
                rows += [{'CODE': code, 'VALUE': float(x), 'YEAR': int(y)} for y, x in v.items()]
            elif v is not None:
                rows.append({'CODE': code, 'VALUE': float(v), 'YEAR': None})
            continue
        if code in PARAMETERS:
            targets = [(None, params[code])]
        else:
            col = series[code] if code in series else pd.Series(dtype=float)
            if c['years'] == 'T':
                years = [T]
            elif c['years'] == 'T+':
                years = [y for y in range(T, T + 3) if y == T or pd.notna(col.get(y, np.nan))]
            elif c['years'] == 'after':
                years = list(range(T + 1, end_year + 1))
            else:
                raise ValueError(f"Unknown years '{c['years']}' for {code}")
            targets = [(y, col.get(y, np.nan)) for y in years]
        for year, base in targets:
            if 'value' in c:
                value = c['value']
            elif 'delta' in c:
                value = base + c['delta']
            else:
                value = base * c['scale']
            if pd.isna(value):
                raise ValueError(f'No baseline value for {code} {year or ""} in {input_file}')
            rows.append({'CODE': code, 'VALUE': float(value), 'YEAR': year})
    return rows


def build_model(spec, country, adjustment_period, input_file, model_kwargs=None, attributes=None):
    """
    StochasticDsaModel for one spec, country and adjustment period (before find_spb_binding). model_kwargs and
    attributes are baseline model arguments and attributes of all runs (e.g. {'adjustment_bound': 5}); those of the
    spec take precedence.
    """
    from classes import StochasticDsaModel as DSA  # local import avoids a circular import
    file = spec['input_file'] or input_file
    kwargs = {**(model_kwargs or {}), **spec['model_kwargs']}
    delay = kwargs.pop('adjustment_start_delay', 0)
    if delay:
        kwargs['adjustment_start_year'] = int(read_country(file, country)[1]['REFERENCE_YEAR']) + 1 + delay
    model = DSA(country, adjustment_period=adjustment_period, input_file=file,
                overrides=resolve_overrides(spec['overrides'], file, country), **kwargs)
    for k, v in {**(attributes or {}), **spec['attributes']}.items():
        setattr(model, k, v)
    for f in spec['setup']:
        f(model)
    return model


NON_DSA_KEYS = ('binding', 'edp', 'debt_safeguard', 'deficit_resilience')


def _next_criterion(targets, value, exclude):
    """
    Next criterion: highest SPB target of the other criteria that does not exceed value. Returns (criterion, gap).
    """
    others = {k: v for k, v in targets.items() if k not in exclude and v <= value + 1e-9}
    if not others:
        return None, np.nan
    k = max(others, key=others.get)
    return k, value - others[k]


def dsa_criterion(model, targets, spb_start):
    """
    Binding DSA criterion (before the EDP and the safeguards) from the SPB targets of the criteria.
    """
    n = model.adjustment_period
    dsa = {k: v for k, v in targets.items() if k not in NON_DSA_KEYS}
    if 'annual_adjustment_dsa' in model.binding_parameter_dict:  # constant annual adjustment (e.g. Commission rules)
        return model._combine_dsa_criteria({k: (v - spb_start) / n for k, v in dsa.items()})[1]
    return max(dsa, key=dsa.get) if dsa else None


def collect_results(model, years_after=10):
    """
    Results of find_spb_binding recorded for each run, for two targets:
        binding target: SPB at the end of the adjustment period including the EDP and the safeguards;
        DSA target (columns 'dsa_*'): SPB required by the DSA criteria alone (deterministic scenarios, deficit and
            stochastic criteria), with the debt path of a linear adjustment to it.
    The next criterion is the highest SPB target of the other criteria that does not exceed the target, the gap to the
    next criterion is how far the target would fall if the binding criterion were dropped.
    """
    n, s, e = model.adjustment_period, model.adjustment_start, model.adjustment_end
    spb_start = float(model.spb_bca[s - 1])
    targets = {k: float(v) for k, v in model.spb_target_dict.items()}
    binding = targets['binding']
    criterion = model.binding_criterion
    summary = model.binding_tables['Summary'].iloc[:, 0]
    dsa_target = float(summary.get('DSA-based SPB target (% of GDP)', np.nan))
    years = range(model.start_year, min(model.start_year + e + years_after, model.end_year) + 1)
    at = lambda t: float(model.d[t]) if t < model.projection_period else np.nan
    path = lambda: {y: float(model.d[y - model.start_year]) for y in years}

    # Binding path (EDP and deficit resilience entries describe the final path, not a separate criterion)
    next_c, gap = _next_criterion(targets, binding, ('binding', criterion, 'edp', 'deficit_resilience'))
    out = {
        'T': model.start_year,
        'adjustment_end_year': model.adjustment_end_year,
        'spb_T': spb_start,
        'debt_T': float(model.d[s - 1]),
        'binding_target': binding,
        'annual_adjustment': (binding - spb_start) / n,
        'binding_criterion': criterion,
        'next_criterion': next_c,
        'gap_to_next': gap,
        'net_expenditure_growth': float(np.mean(model.net_expenditure_growth[s:e + 1])),
        'debt_end_adjustment': at(e),
        'debt_10y_after': at(e + years_after),
        'edp_binding': getattr(model, 'edp_binding', None),
        'debt_safeguard_binding': getattr(model, 'debt_safeguard_binding', None),
        'deficit_resilience_binding': getattr(model, 'deficit_resilience_binding', None),
        'guidance': getattr(model, 'guidance', ''),
        'fiscal_multiplier': float(model.fiscal_multiplier),
        'adjustment_bound': float(getattr(model, 'adjustment_bound', SEARCH_BOUND)),
        'spb_target_dict': targets,
        'debt_path': path(),
    }

    # DSA criteria alone: linear adjustment to the DSA target without EDP and deficit resilience steps
    dsa_c = dsa_criterion(model, targets, spb_start)
    dsa_next, dsa_gap = _next_criterion({k: v for k, v in targets.items() if k not in NON_DSA_KEYS}, dsa_target, (dsa_c,))
    model.project(spb_steps=np.full(n, (dsa_target - spb_start) / n))
    out.update({
        'dsa_target': dsa_target,
        'dsa_annual_adjustment': (dsa_target - spb_start) / n,
        'dsa_criterion': dsa_c,
        'dsa_next_criterion': dsa_next,
        'dsa_gap_to_next': dsa_gap,
        'dsa_net_expenditure_growth': float(np.mean(model.net_expenditure_growth[s:e + 1])),
        'dsa_debt_end_adjustment': at(e),
        'dsa_debt_10y_after': at(e + years_after),
        'dsa_debt_path': path(),
    })
    return out


def run_task(task):
    """
    Run one task (spec, country, adjustment period, input file, rules, seed, baseline model arguments) and return
    (key, results). Top-level function so that it can be sent to worker processes.
    """
    key, spec, country, n, input_file, rules, seed, model_kwargs, attributes = task
    t0 = time.time()
    out = {}
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            model = build_model(spec, country, n, input_file, model_kwargs, attributes)
            kwargs = {'rules': rules, 'stochastic': True, 'print_results': False, **spec['binding_kwargs']}
            np.random.seed(seed if spec['seed'] is None else spec['seed'])
            model.find_spb_binding(**kwargs)
            out = collect_results(model)
    except Exception as err:
        out = {'error': f'{type(err).__name__}: {err}'}
    out['runtime'] = time.time() - t0
    return key, out


def _setup_signature(f):
    if isinstance(f, functools.partial):
        return [f.func.__name__, list(f.args), f.keywords]
    return f.__name__


def _setup_label(f):
    if isinstance(f, functools.partial):
        args = [f'{a:g}' if isinstance(a, float) else str(a) for a in f.args]
        args += [f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in f.keywords.items()]
        return f"{f.func.__name__}({', '.join(args)})"
    return f.__name__


def task_key(spec, country, n, input_file, rules, seed, model_kwargs=None, attributes=None):
    """
    Cache key of a task: all fields that change results (labels are not part of the key). Baseline attributes enter
    the key only if set, so that keys of earlier runs remain valid.
    """
    fields = {k: spec[k] for k in ['model_kwargs', 'attributes', 'overrides', 'binding_kwargs', 'input_file', 'seed']}
    fields['setup'] = [_setup_signature(f) for f in spec['setup']]
    key = [fields, country, n, str(input_file), rules, seed, model_kwargs or {}]
    if attributes:
        key.append(attributes)
    return json.dumps(key, sort_keys=True, default=str)


def _load_cache(path):
    if path is not None and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def _save_cache(cache, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(cache, f)
    os.replace(tmp, path)


def run_specs(specs, countries, adjustment_periods=(4, 7), input_file=None, rules='commission', seed=0,
              model_kwargs=None, attributes=None, cache_file=None, run=True, parallel=True, max_workers=None,
              save_every=100, verbose=True):
    """
    Run specs for all countries and adjustment periods and return one tidy DataFrame (one row per spec, country and
    adjustment period, with the spec labels). Results are cached by task in cache_file: tasks in the cache are not
    rerun. With run=False, only cached results are returned.

    Parameters:
        specs (list): Specs (build_oat_specs, build_global_specs).
        input_file (str): Baseline input workbook.
        rules (str): Baseline rules of find_spb_binding ('commission' or 'default').
        seed (int): Random seed set before each run (specs can set their own).
        model_kwargs (dict): Baseline model arguments of all runs (e.g. {'fiscal_multiplier_type': 'pers'}); the
            model arguments of a spec take precedence.
        attributes (dict): Baseline model attributes of all runs (e.g. {'adjustment_bound': 5}, the upper bound of
            the annual adjustment in find_spb_binding); the attributes of a spec take precedence.
        max_workers (int): Number of worker processes (default: number of CPUs).
    """
    input_file = input_file or latest_input_file()
    cache = _load_cache(cache_file)
    tasks = [(task_key(s, c, n, input_file, rules, seed, model_kwargs, attributes), s, c, n, input_file, rules, seed,
              model_kwargs, attributes) for s in specs for n in adjustment_periods for c in countries]
    todo = list({t[0]: t for t in tasks if t[0] not in cache}.values())

    if todo and not run:
        warnings.warn(f'{len(todo)} of {len(tasks)} tasks not in the cache {cache_file} (set RUN = True to compute them)')
    elif todo:
        # Tasks with the same input workbook run together (the workbook reader caches one file per process)
        todo.sort(key=lambda t: str(t[1]['input_file']))
        t0, done = time.time(), 0
        if verbose:
            print(f'Running {len(todo)} tasks ({len(tasks) - len(todo)} cached)')

        def store(key, out):
            nonlocal done
            cache[key] = out
            done += 1
            if cache_file is not None and done % save_every == 0:
                _save_cache(cache, cache_file)
            if verbose and (done % max(1, len(todo) // 20) == 0 or done == len(todo)):
                print(f'  {done}/{len(todo)} tasks, {time.time() - t0:.0f} s', flush=True)

        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_task, t) for t in todo]
                for future in as_completed(futures):
                    store(*future.result())
        else:
            for t in todo:
                store(*run_task(t))
        if cache_file is not None:
            _save_cache(cache, cache_file)

    rows = []
    for key, s, c, n, *_ in tasks:
        if key in cache:
            rows.append({'id': s['id'], 'group': s['group'], 'check': s['check'], 'variant': s['variant'],
                         'value': s['value'], 'knob': s['knob'], 'country': c, 'adjustment_period': n, **s['params'],
                         **cache[key]})
    df = pd.DataFrame(rows)
    if 'error' in df and verbose and df['error'].notna().any():
        print(f"{df['error'].notna().sum()} runs failed, see column 'error'")
    return df


# ========================================================================================= #
#                                          RESULTS                                          #
# ========================================================================================= #

# Two targets are recorded for each run: the binding target (DSA criteria, EDP and safeguards) and the DSA target
# (DSA criteria only). Result functions and charts take target='dsa' or 'binding' and read the columns below.
TARGETS = {
    'binding': {'target': 'binding_target', 'criterion': 'binding_criterion', 'debt_path': 'debt_path',
                'annual_adjustment': 'annual_adjustment', 'debt_10y_after': 'debt_10y_after',
                'delta_target': 'delta_target', 'criterion_switch': 'criterion_switch',
                'delta_adjustment': 'delta_adjustment', 'delta_debt_10y': 'delta_debt_10y',
                'baseline_target': 'baseline_binding_target', 'commission': 'commission_target',
                'label': 'binding SPB target', 'long': 'binding SPB target (DSA criteria, EDP and safeguards)'},
    'dsa': {'target': 'dsa_target', 'criterion': 'dsa_criterion', 'debt_path': 'dsa_debt_path',
            'annual_adjustment': 'dsa_annual_adjustment', 'debt_10y_after': 'dsa_debt_10y_after',
            'delta_target': 'dsa_delta_target', 'criterion_switch': 'dsa_criterion_switch',
            'delta_adjustment': 'dsa_delta_adjustment', 'delta_debt_10y': 'dsa_delta_debt_10y',
            'baseline_target': 'baseline_dsa_target', 'commission': 'commission_dsa_target',
            'label': 'DSA-based SPB target', 'long': 'DSA-based SPB target (DSA criteria only)'},
}


def commission_targets(countries, vintage='2024'):
    """
    SPB at the end of the adjustment period in the Commission prior guidance (4 and 7 years): after the safeguards
    (commission_target, as published) and from the DSA criteria alone (commission_dsa_target: SPB in T plus the
    DSA-based annual adjustment times the adjustment period). The sheets for countries receiving technical information
    report no DSA-based adjustment, only an SPB target that includes the deficit resilience safeguard: their DSA
    target is missing.
    """
    from data_pipeline.commission import load_prior_guidance
    sheets = load_prior_guidance(countries, vintage=vintage)
    rows = []
    for c in countries:
        res = sheets[c].results()
        for n in (4, 7):
            rows.append({'country': c, 'adjustment_period': n, 'commission_target': res[f'spb_end_{n}y'],
                         'commission_dsa_target': res['spb_T'] + n * res[f'annual_adjustment_dsa_{n}y']})
    return pd.DataFrame(rows)


def flag_infeasible(df):
    """
    Flag runs whose binding target is at the upper bound of the search (column 'safeguard_infeasible'): the debt
    sustainability safeguard is not met even with the largest annual adjustment searched (adjustment_bound recorded
    with each run, SEARCH_BOUND for earlier runs). The binding target of these runs is censored.
    """
    out = df.copy()
    if {'binding_target', 'spb_T'} <= set(out):
        bound = out['adjustment_bound'].fillna(SEARCH_BOUND) if 'adjustment_bound' in out else SEARCH_BOUND
        out['safeguard_infeasible'] = (out['binding_target']
                                       >= out['spb_T'] + out['adjustment_period'] * bound - 0.005).fillna(False)
    return out


# Three tiers of each effect: the SPB required by the criterion a check acts on (own criterion), the DSA target (all
# DSA criteria) and the binding target (DSA criteria, EDP and safeguards). The own criterion of the stress-test sizes is
# their stress test, of the stochastic settings the stochastic criterion, of all other checks the deterministic debt
# criterion of the baseline scenario (debt declines or stays below 60%).
SCENARIOS = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']
OWN_SCENARIO = {'adverse_r_g_shock': 'adverse_r_g', 'lower_spb_shock': 'lower_spb', 'financial_stress_shock': 'financial_stress'}
TIERS = {'own': ('Own criterion', 'own_delta_target'), 'dsa': (TARGET_LABELS['dsa'], 'dsa_delta_target'),
         'binding': (TARGET_LABELS['binding'], 'delta_target')}


def _own_criterion(row, base):
    """
    Key of spb_target_dict for the own criterion of a run (row), given the baseline run of its country (base). The
    Commission's DSA uses either the 'debt declines' or the 'debt below 60%' variant of all scenarios, whichever gives
    the lower target (FiscalRules._combine_dsa_criteria); the own criterion of a stress test uses the variant of the
    baseline. None if the criterion does not apply (adjustment floor, no reference trajectory).
    """
    d = base['spb_target_dict'] if isinstance(base['spb_target_dict'], dict) else {}
    knob, group = row.get('knob'), row.get('group')
    if group == ESTIMATION or knob in ('stochastic', 'Probability target'):
        return 'stochastic' if 'stochastic' in d else None
    sc = OWN_SCENARIO.get(knob, 'main_adjustment')
    if f'debt_declines_{sc}' not in d:  # no reference trajectory: only the baseline scenario, debt below 60%
        return 'debt_below_60_main_adjustment' if sc == 'main_adjustment' and 'debt_below_60_main_adjustment' in d else None
    declines = [d[k] for k in d if k.startswith('debt_declines_') or k in ('deficit_reduction', 'stochastic')]
    below = [d[k] for k in d if k.startswith('debt_below_60_') or k == 'deficit_reduction']
    return f'debt_declines_{sc}' if max(declines) <= max(below) else f'debt_below_60_{sc}'


def add_own_criterion(df, baseline_id='baseline'):
    """
    Add the own-criterion effect (see TIERS): own_criterion (key of spb_target_dict) and own_delta_target, the change of
    the SPB that the own criterion requires relative to the baseline run of the same country and adjustment period
    (NaN if the criterion does not apply or a run failed).
    """
    df = df.copy()
    base = df[df['id'] == baseline_id].set_index(['country', 'adjustment_period'])
    crit, delta = [], []
    for _, r in df.iterrows():
        key = (r['country'], r['adjustment_period'])
        b = base.loc[key] if key in base.index else None
        c = _own_criterion(r, b) if b is not None else None
        own = r['spb_target_dict'] if isinstance(r.get('spb_target_dict'), dict) else {}
        crit.append(c)
        delta.append(own[c] - b['spb_target_dict'][c] if c and c in own else np.nan)
    df['own_criterion'], df['own_delta_target'] = crit, delta
    return df


def feasible(df, target):
    """
    Runs without a censored target: for the binding target, runs where the safeguard can be met (flag_infeasible).
    """
    if target == 'binding' and 'safeguard_infeasible' in df:
        return df[~df['safeguard_infeasible'].astype(bool)]
    return df


def add_deviations(df, baseline_id='baseline'):
    """
    Add deviations from the baseline run of the same country and adjustment period, for both targets (see TARGETS):
    change of the target (pp.), of the annual adjustment and of debt 10 years after the adjustment period, and whether
    the binding criterion differs from the baseline. Also flags censored binding targets (flag_infeasible).
    """
    keys = ['country', 'adjustment_period']
    df = flag_infeasible(df)
    out = df.drop(columns=[c for c in df if c.startswith('baseline_')])
    for cols in TARGETS.values():
        base = df.loc[df['id'] == baseline_id, keys + [cols['target'], cols['annual_adjustment'], cols['debt_10y_after'],
                                                       cols['criterion']]]
        base = base.rename(columns=lambda c: c if c in keys else f'baseline_{c}')
        out = out.merge(base, on=keys, how='left')
        out[cols['delta_target']] = out[cols['target']] - out[f"baseline_{cols['target']}"]
        out[cols['delta_adjustment']] = out[cols['annual_adjustment']] - out[f"baseline_{cols['annual_adjustment']}"]
        out[cols['delta_debt_10y']] = out[cols['debt_10y_after']] - out[f"baseline_{cols['debt_10y_after']}"]
        out[cols['criterion_switch']] = (out[cols['criterion']].notna()
                                         & (out[cols['criterion']] != out[f"baseline_{cols['criterion']}"]))
    return out


def noise_band(df, target='dsa', group='Noise'):
    """
    Range of the change of the target across seeds (baseline seed included) by country and adjustment period.
    """
    noise = df.loc[df['group'].isin([group, 'Baseline']) & (df['id'] != 'baseline_deterministic')]
    return noise.groupby(['country', 'adjustment_period'])[TARGETS[target]['delta_target']].agg(
        noise_low='min', noise_high='max').reset_index()


def effects_table(df, adjustment_periods=(4, 7), exclude_groups=('Baseline', 'Noise', 'Global'), threshold=0.1):
    """
    Main table: change of the SPB target relative to the baseline by check variant, mean, minimum and maximum across
    countries (pp. of GDP) and the number of countries whose target changes by more than threshold (pp.), for the three
    tiers (TIERS): own criterion (if add_own_criterion was applied), DSA target (DSA criteria only) and binding target
    (DSA criteria, EDP and safeguards), by adjustment period. Failed
    runs (e.g. no data for a country) and, for the binding target, runs where the debt sustainability safeguard cannot
    be met (feasible) are left out.
    """
    d = df[~df['group'].isin(exclude_groups)]
    cols = {}
    for n in adjustment_periods:
        dn = d[d['adjustment_period'] == n]
        for target, (label, col) in TIERS.items():
            if col not in dn:  # own criterion only after add_own_criterion
                continue
            g = feasible(dn, target).groupby(['group', 'check', 'variant'], sort=False)[col]
            for stat in ['mean', 'min', 'max']:
                cols[(f'{n}-year adjustment', label, stat)] = g.agg(stat)
            cols[(f'{n}-year adjustment', label, f'countries > {threshold:g} pp.')] = g.agg(
                lambda x: int((x.abs() > threshold + 1e-9).sum()))
    out = pd.DataFrame(cols)
    out.columns.names = ['adjustment', 'target', 'statistic']
    return out


# Accommodative and restrictive combinations: 10th and 90th percentile of the draws. The outer tails are combinations
# of assumptions at the ends of their ranges, drawn independently, which are economically implausible jointly.
COMBINATIONS = {'draws:accommodative': ('accommodative', 0.10), 'draws:restrictive': ('restrictive', 0.90)}


def draw_combinations(glob, combinations=COMBINATIONS):
    """
    Accommodative and restrictive combinations of assumptions from the global draws (after add_deviations): for each
    country, adjustment period and target, the 10th and 90th percentile of the target across draws (COMBINATIONS). The draws are a sample,
    so percentiles do not depend on single extreme combinations as the lowest and highest draw would. Returns runs in the
    format of run_specs (group 'Global draws') with the percentile of the target and of its change, and the annual
    adjustment, binding criterion, debt path and id ('<target>_draw') of the draw whose target is closest to the
    percentile.
    """
    draws = glob[glob['group'] == 'Global']
    rows = []
    for (c, n), g in draws.groupby(['country', 'adjustment_period']):
        for cid, (label, q) in combinations.items():
            row = {'id': cid, 'group': 'Global draws', 'check': 'Combined assumptions (global draws)',
                   'variant': f'{label} ({q * 100:.0f}th percentile)', 'country': c, 'adjustment_period': n,
                   'T': g['T'].iloc[0]}
            for key, cols in TARGETS.items():
                d = feasible(g, key).dropna(subset=[cols['target']])
                if not len(d):
                    continue
                row[cols['target']] = d[cols['target']].quantile(q)
                row[cols['delta_target']] = d[cols['delta_target']].quantile(q)
                i = (d[cols['target']] - row[cols['target']]).abs().idxmin()
                for col in ['debt_path', 'annual_adjustment', 'criterion']:
                    row[cols[col]] = d.loc[i, cols[col]]
                row[f'{key}_draw'] = d.loc[i, 'id']
            rows.append(row)
    return pd.DataFrame(rows)


def variance_decomposition(df, target='dsa', assumptions=GLOBAL_ASSUMPTIONS, categories=GLOBAL_CATEGORIES):
    """
    Variance decomposition of the global sensitivity analysis by country and adjustment period: squared standardised
    regression coefficients of a linear regression of the target on the drawn assumptions (share of the variance of the
    target explained by each assumption), rescaled to add up to the R2 of the regression (the draws are close to, but
    not exactly, uncorrelated, so the raw squared coefficients can add up to more than R2), summed by category. Continuous
    assumptions and discrete settings with numeric values (e.g. ageing cost period, probability threshold) enter
    linearly, on/off settings as 0/1 indicators; categorical settings (string values) enter as dummies, whose shares
    are added up. The remainder (1 - R2) is the variance from non-linear effects and interactions, e.g. switches of the
    binding criterion. For the binding target, runs where the debt sustainability safeguard cannot be met are left out
    (feasible).
    """
    y = TARGETS[target]['target']
    rows = []
    for (c, n), g in feasible(df, target).dropna(subset=[y]).groupby(['country', 'adjustment_period']):
        X, owner = _design(g, assumptions)
        res = _shares(X, g[y].to_numpy(float), owner, assumptions)
        if res is None:
            continue
        shares, r2, std = res
        shares = pd.Series(shares, index=assumptions)
        row = {'country': c, 'adjustment_period': n, 'target': target, 'std': std, 'r2': r2}
        row.update({k: shares[v].sum() for k, v in categories.items()})
        row['Non-linear and interactions'] = max(0.0, 1 - r2)
        row.update({f'src2_{k}': v for k, v in shares.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _design(g, assumptions):
    """
    Regressors of the variance decomposition: drawn assumptions, numeric settings linearly, categorical settings as
    dummies. Returns (X, owner), owner the assumption of each column.
    """
    parts, owner = [], []
    for a in assumptions:
        x = g[a]
        if x.dtype == object:
            dummies = pd.get_dummies(x.astype(str), drop_first=True, dtype=float)
            parts.append(dummies.to_numpy())
            owner += [a] * dummies.shape[1]
        else:
            parts.append(x.to_numpy(float)[:, None])
            owner.append(a)
    return np.column_stack(parts), owner


def _shares(X, Y, owner, assumptions):
    """
    Squared standardised regression coefficients summed by assumption (array in the order of assumptions) and rescaled
    to add up to R2, R2 and the standard deviation of the target; None if the target does not vary or there are too few
    runs.
    """
    keep = X.std(0) > 0
    X, owner = X[:, keep], [o for o, k in zip(owner, keep) if k]
    if len(Y) <= X.shape[1] + 1 or Y.std() < 1e-10:
        return None
    Xs = np.column_stack([np.ones(len(Y)), (X - X.mean(0)) / X.std(0)])
    coef = np.linalg.lstsq(Xs, Y, rcond=None)[0]
    pos = {a: i for i, a in enumerate(assumptions)}
    shares = np.zeros(len(assumptions))
    np.add.at(shares, [pos[o] for o in owner], coef[1:] ** 2 / Y.var())
    r2 = 1 - np.var(Y - Xs @ coef) / Y.var()
    if shares.sum() > 0:
        shares *= r2 / shares.sum()
    return shares, r2, Y.std()


def variance_table(vd, assumptions=GLOBAL_ASSUMPTIONS, categories=GLOBAL_CATEGORIES):
    """
    Variance decomposition of the global sensitivity analysis aggregated over countries (variance_decomposition):
    share of the variance of the target explained by each assumption and category of assumptions, and by non-linear
    effects and interactions, by adjustment period and target. Countries are weighted by the variance of their target,
    so the shares refer to the variance summed over countries (countries whose target hardly moves count little).
    The last row is the mean standard deviation of the target across countries.
    """
    names = {a: assumption_label(a) for a in assumptions}
    cols = {}
    for (n, target), g in vd.groupby(['adjustment_period', 'target'], sort=False):
        w = g['std'] ** 2 / (g['std'] ** 2).sum()
        s = {}
        for category, members in categories.items():
            s[(category, 'Total')] = (g[category] * w).sum()
            s.update({(category, names[a]): (g[f'src2_{a}'] * w).sum() for a in members})
        s[('Non-linear and interactions', '')] = (g['Non-linear and interactions'] * w).sum()
        s[('Standard deviation of the target (pp.)', '')] = g['std'].mean()
        cols[(f'{n}-year adjustment', TARGET_LABELS[target])] = pd.Series(s)
    return pd.DataFrame(cols)


# ========================================================================================= #
#                                           CHARTS                                          #
# ========================================================================================= #

def _order_countries(df, period, target='dsa'):
    base = df[(df['id'] == 'baseline') & (df['adjustment_period'] == period)]
    return base.sort_values(TARGETS[target]['target'])['country'].tolist()


def plot_mean_effects(df, adjustment_periods=(4, 7), exclude_groups=('Baseline', 'Noise', 'Global')):
    """
    Mean change of the SPB target across countries by check variant (pp. of GDP, see effects_table): bars for the DSA
    target, diamonds for the binding target (DSA criteria, EDP and safeguards). One panel per adjustment period.
    """
    t = effects_table(df, adjustment_periods, exclude_groups)
    groups = t.index.get_level_values('group')
    labels = [f'{check}: {variant}' for _, check, variant in t.index]
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, len(adjustment_periods), figsize=(6 * len(adjustment_periods) + 5, 0.22 * len(t) + 1.6),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], adjustment_periods):
        period = f'{n}-year adjustment'
        ax.barh(y, t[(period, TARGET_LABELS['dsa'], 'mean')], height=0.7, color='C0', label='DSA criteria')
        ax.scatter(t[(period, TARGET_LABELS['binding'], 'mean')], y, marker='D', s=18, color='black', zorder=3,
                   label='DSA, EDP and safeguards')
        ax.axvline(0, color='black', lw=0.8)
        for k in range(1, len(groups)):
            if groups[k] != groups[k - 1]:
                ax.axhline(y[k] + 0.5, color='grey', lw=0.8)
        if len(adjustment_periods) > 1:
            ax.set_title(period)
        ax.set_xlabel('Mean change across countries (pp. of GDP)')
        ax.grid(axis='y', visible=False)
    axes[0, 0].set_yticks(y, labels, fontsize=8)
    axes[0, 0].set_ylim(-0.6, len(t) - 0.4)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False)
    title = 'Change of the SPB target relative to the baseline'
    if len(adjustment_periods) == 1:
        title += f', {adjustment_periods[0]}-year adjustment'
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout(rect=(0, 0.6 / fig.get_figheight(), 1, 1))
    return fig


def plot_ranges(df, adjustment_period, target='dsa', combinations=tuple(COMBINATIONS),
                exclude_groups=('Noise', 'Global'), exclude_ids=('data:latest',), figsize=(14, 5.5)):
    """
    SPB targets by country (ordered by the baseline target): range across the individual checks of model assumptions
    (grey bar; data revisions excluded), baseline, accommodative and restrictive combinations of the global draws
    (10th and 90th percentile, draw_combinations), Commission target and SPB in T.
    """
    cols = TARGETS[target]
    d = df[df['adjustment_period'] == adjustment_period]
    countries = _order_countries(d, adjustment_period, target)
    oat = d[~d['group'].isin(exclude_groups + ('Global draws',)) & ~d['id'].isin(exclude_ids)]
    rng = oat.groupby('country')[cols['target']].agg(['min', 'max']).reindex(countries)
    base = d[d['id'] == 'baseline'].set_index('country').reindex(countries)
    x = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, rng['max'] - rng['min'], bottom=rng['min'], width=0.6, color=LIGHT_GREY, zorder=1,
           label='Range of individual checks')
    ax.scatter(x, base['spb_T'], marker='_', s=180, lw=2, color=CURRENT, zorder=2, label='SPB in T')
    if cols['commission'] in base and base[cols['commission']].notna().any():
        ax.scatter(x, base[cols['commission']], marker='D', s=30, facecolor='none', edgecolor=COMMISSION, lw=1.4,
                   zorder=3, label='Commission')
    (lo, q_lo), (hi, q_hi) = COMBINATIONS[combinations[0]], COMBINATIONS[combinations[1]]
    for bid, color, label, marker in [(combinations[0], ACCOMMODATIVE, f'Accommodative ({q_lo * 100:.0f}th pct. of draws)', 'v'),
                                      (combinations[1], RESTRICTIVE, f'Restrictive ({q_hi * 100:.0f}th pct. of draws)', '^')]:
        b = d[d['id'] == bid].set_index('country').reindex(countries)
        if b[cols['target']].notna().any():
            ax.scatter(x, b[cols['target']], marker=marker, s=40, color=color, zorder=4, label=label)
    ax.scatter(x, base[cols['target']], marker='o', s=36, color=BASELINE, zorder=5, label='Baseline')
    ax.set_xticks(x, countries)
    ax.set_xlim(-0.6, len(countries) - 0.4)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('SPB at the end of the adjustment period (% of GDP)')
    ax.set_title(f"{TARGET_LABELS[target]}, {adjustment_period}-year adjustment", loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    plt.tight_layout()
    return fig


def plot_debt_paths(df, adjustment_period=4, target='dsa', countries=None, ncols=5,
                    exclude_groups=('Baseline', 'Noise', 'Global', 'Global draws'), exclude_ids=('data:latest',)):
    """
    Debt by country (one panel each) under a linear adjustment to the target of each run: baseline and the accommodative
    and restrictive combinations of the global draws (draws closest to the 10th and 90th percentile of the target), with
    the range across the
    individual checks of model assumptions (grey; data revisions excluded).
    """
    col = TARGETS[target]['debt_path']
    d = df[(df['adjustment_period'] == adjustment_period) & df[col].notna()]
    countries = countries or list(dict.fromkeys(d['country']))
    lines = [('baseline', 'Baseline', BASELINE),
             ('draws:accommodative', f"Accommodative ({COMBINATIONS['draws:accommodative'][1] * 100:.0f}th pct. of draws)", ACCOMMODATIVE),
             ('draws:restrictive', f"Restrictive ({COMBINATIONS['draws:restrictive'][1] * 100:.0f}th pct. of draws)", RESTRICTIVE)]
    nrows = int(np.ceil(len(countries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows + 0.8), squeeze=False)
    for ax, c in zip(axes.ravel(), countries):
        dc = d[d['country'] == c]
        T = dc.loc[dc['id'] == 'baseline', 'T'].iloc[0]
        oat = dc[~dc['group'].isin(exclude_groups) & ~dc['id'].isin(exclude_ids) & (dc['T'] == T)]
        paths = pd.DataFrame({i: pd.Series(p) for i, p in zip(oat['id'], oat[col])}).sort_index()
        ax.fill_between(paths.index, paths.min(axis=1), paths.max(axis=1), color=LIGHT_GREY, lw=0)
        for i, label, color in lines:
            r = dc[dc['id'] == i]
            if len(r):
                p = pd.Series(r[col].iloc[0]).sort_index()
                ax.plot(p.index, p.values, color=color, lw=1.5, label=label)
        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
    for ax in axes.ravel()[len(countries):]:
        ax.set_visible(False)
    handles = [Patch(color=LIGHT_GREY, label='Range of individual checks')]
    handles += [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in lines]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False)
    fig.suptitle(f"Debt (% of GDP) under an adjustment to the SPB target ({TARGET_LABELS[target]}), "
                 f"{adjustment_period}-year adjustment", fontweight='bold')
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight(), 1, 1))
    return fig
