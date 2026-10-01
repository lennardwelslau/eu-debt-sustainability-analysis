# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Sensitivity Analysis     #
# ========================================================================================= #
#
# Helper functions for code/exercises/sensitivity_analysis.ipynb. The exercise runs the model
# (StochasticDsaModel.find_spb_binding) under alternative data, assumptions and rules and records
# how the binding SPB target responds. It has five parts:
#
# 1. Specifications: each check is one spec, a dict with model keyword arguments, model attributes,
#    input overrides (relative to the baseline data), find_spb_binding keyword arguments and optional
#    setup functions applied after initialisation (build_oat_specs, build_global_specs, combine_specs).
# 2. Runner: run_specs runs specs for countries and adjustment periods in parallel (ProcessPoolExecutor)
#    and caches results by task, so reruns only compute missing tasks.
# 3. Results: one tidy DataFrame with one row per spec, country and adjustment period, deviations from the
#    baseline, noise band from alternative seeds, bundles and variance decomposition.
# 4. Charts: heatmap, ranges, tornado, response curves, binding criterion shares, debt paths, global
#    variance shares.
#
# Setup functions must be defined at module level (not in a notebook) so that specs can be sent to worker
# processes; use functools.partial to pass arguments.
#
# Author: Lennard Welslau
# Updated: 2026-09-30
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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_pipeline import read_country, latest_input_file, resolve_input_path, REPO_ROOT, PARAMETERS
from data_pipeline.validate import RULE_SWITCHES

OUTPUT_DIR = REPO_ROOT / 'output' / 'sensitivity'

# Colours (categorical order fixed; diverging blue-grey-red for deviations from the baseline)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GREY, LIGHT_GREY, INK, MUTED = '#8c8b87', '#e4e3df', '#0b0b0b', '#52514e'
LOWER, HIGHER = '#2a78d6', '#e34948'  # lower / higher SPB target than the baseline
DIVERGING = mcolors.LinearSegmentedColormap.from_list('dsa_div', ['#104281', '#6da7ec', '#f0efec', '#f0908f', '#a3201f'])
COMMISSION, BASELINE, ACCOMMODATIVE, STRICT, CURRENT = INK, COLORS[3], LOWER, HIGHER, GREY


# ========================================================================================= #
#                                     SPECIFICATIONS                                        #
# ========================================================================================= #

def make_spec(id, group, check, variant, value=np.nan, base_value=np.nan, knob=None, model_kwargs=None,
              attributes=None, overrides=None, binding_kwargs=None, setup=None, input_file=None, seed=None,
              bundle=True, params=None):
    """
    One sensitivity check.

    Parameters:
        id (str): Unique identifier, e.g. 'macro:INTEREST_RATE_LT_T10:+1'.
        group, check, variant (str): Labels (e.g. 'Macro', 'Long-term rate T+10', '+1').
        value (float): Numeric value of the variant (x-axis of response curves), NaN if not numeric.
        base_value (float or str): Value of the baseline on the same scale, or the name of a recorded result column
            (e.g. 'fiscal_multiplier') if it differs by country.
        knob (str): Setting changed by the spec; specs with the same knob are alternatives (one per bundle).
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
        bundle (bool): Whether the spec can enter the accommodative and strict bundles (see define_bundles).
        params (dict): Parameter values of global sensitivity draws.
    """
    return {
        'id': id, 'group': group, 'check': check, 'variant': variant, 'value': value, 'base_value': base_value,
        'knob': knob or check, 'model_kwargs': model_kwargs or {}, 'attributes': attributes or {},
        'overrides': overrides or [], 'binding_kwargs': binding_kwargs or {}, 'setup': list(setup or []),
        'input_file': input_file, 'seed': seed, 'bundle': bundle, 'params': params or {},
    }


def baseline_spec(stochastic=True):
    """
    Baseline: input workbook, rules and seed of the run.
    """
    if stochastic:
        return make_spec('baseline', 'Baseline', 'Baseline', 'baseline', bundle=False)
    return make_spec('baseline_deterministic', 'Baseline', 'Baseline (deterministic)', 'baseline',
                     binding_kwargs={'stochastic': False}, bundle=False)


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


def scale_ageing_cost(model, factor):
    """
    Scale the change in ageing costs relative to T by factor.
    """
    model.ageing_cost = model.ageing_cost[0] + factor * (model.ageing_cost - model.ageing_cost[0])


def stochastic_start_at_adjustment(model):
    """
    Start the stochastic projection in the first adjustment year instead of the year after the adjustment period.
    Primary balance shocks are zero during the adjustment period (see StochasticDsaModel._draw_shocks_normal).
    """
    model.stochastic_start_year = model.adjustment_start_year
    model.stochastic_start = model.adjustment_start
    model.stochastic_end = model.stochastic_start + model.stochastic_period - 1
    model.draw_period = model.stochastic_period * (4 if model.shock_frequency == 'quarterly' else 1)


# One-at-a-time checks ------------------------------------------------------------------ #

# Continuous assumptions: name -> (group, check label, function value -> spec fields, baseline value, OAT values).
# The global sensitivity analysis draws the same assumptions over the range of the OAT values.
def _override(code, years, **change):
    return {'overrides': [{'code': code, 'years': years, **change}]}


def _anchor(variable, anchor, delta):
    return {'setup': [functools.partial(shift_anchor, variable=variable, anchor=anchor, delta=delta)]}


ASSUMPTIONS = {
    'rate_st_T10': ('Macro', 'Short-term rate T+10', lambda v: _anchor('i_st', 10, v), 0, [-1, 1]),
    'rate_lt_T10': ('Macro', 'Long-term rate T+10', lambda v: _anchor('i_lt', 10, v), 0, [-1, 1]),
    'rate_st_T30': ('Macro', 'Short-term rate T+30', lambda v: _anchor('i_st', 30, v), 0, [-0.5, 0.5]),
    'rate_lt_T30': ('Macro', 'Long-term rate T+30', lambda v: _anchor('i_lt', 30, v), 0, [-0.5, 0.5]),
    'inflation_T10': ('Macro', 'Inflation T+10', lambda v: _anchor('pi', 10, v), 0, [-0.5, 0.5]),
    'inflation_T30': ('Macro', 'Inflation T+30', lambda v: _anchor('pi', 30, v), 0, [-0.5, 0.5]),
    'potential_growth': ('Macro', 'Potential growth', lambda v: {'setup': [functools.partial(shift_potential_growth, delta=v)]}, 0, [-0.5, 0.5]),
    'ageing_scale': ('Macro', 'Ageing cost change (scale)', lambda v: {'setup': [functools.partial(scale_ageing_cost, factor=v)]}, 1, [0.5, 1.5]),
    'elasticity_scale': ('Macro', 'Budget balance elasticity (scale)', lambda v: _override('BUDGET_BALANCE_ELASTICITY', None, scale=v), 1, [0.8, 1.2]),
    'fiscal_multiplier': ('Multiplier', 'Fiscal multiplier', lambda v: {'model_kwargs': {'fiscal_multiplier': v}}, 'fiscal_multiplier', [0, 0.5, 1.0, 1.25]),
    'adverse_r_g_shock': ('Stress tests', 'Adverse r-g shock', lambda v: {'attributes': {'adverse_r_g_shock': v}}, 0.5, [0.25, 0.75, 1.0]),
    'financial_stress_shock': ('Stress tests', 'Financial stress shock', lambda v: {'attributes': {'financial_stress_shock': v}}, 1.0, [0.5, 1.5]),
    'lower_spb_shock': ('Stress tests', 'Lower SPB shock', lambda v: {'attributes': {'lower_spb_shock': v}}, 0.5, [0.25, 1.0]),
}

# Assumptions varied in the global sensitivity analysis
GLOBAL_ASSUMPTIONS = ['rate_st_T10', 'rate_lt_T10', 'rate_st_T30', 'rate_lt_T30', 'inflation_T10',
                      'inflation_T30', 'potential_growth', 'ageing_scale', 'elasticity_scale', 'fiscal_multiplier',
                      'adverse_r_g_shock', 'financial_stress_shock', 'lower_spb_shock']

# Categories of the global variance decomposition
GLOBAL_CATEGORIES = {
    'Interest rates': ['rate_st_T10', 'rate_lt_T10', 'rate_st_T30', 'rate_lt_T30'],
    'Inflation': ['inflation_T10', 'inflation_T30'],
    'Growth and ageing': ['potential_growth', 'ageing_scale'],
    'Multiplier and elasticity': ['fiscal_multiplier', 'elasticity_scale'],
    'Stress-test calibration': ['adverse_r_g_shock', 'financial_stress_shock', 'lower_spb_shock'],
}


def _fmt(v):
    return f'{v:+g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def assumption_spec(name, value, **kwargs):
    """
    Spec for one value of a continuous assumption (see ASSUMPTIONS).
    """
    group, check, fields, base, _ = ASSUMPTIONS[name]
    variant = f'x{value:g}' if name.endswith('scale') else (f'{value:g}' if not isinstance(base, (int, float)) or base != 0 else _fmt(value))
    return make_spec(f'{group.lower().split()[0]}:{name}:{variant}', group, check, variant, value=value, base_value=base, knob=name,
                     **{**fields(value), **kwargs})


def eurostat_repayment_profiles(input_file, countries, end_year=2070):
    """
    Repayment profiles from Eurostat for the reference year T of each country in input_file (data_pipeline.maturity),
    {ISO3: {year: value}}: repayments of the long-term debt outstanding at the end of T-1 (excluding short-term debt and
    the ESM/EFSF loans of the workbook), T+1 to end_year, rescaled to the long-term debt excluding ESM/EFSF loans of the
    workbook in T. The model scales the profile to that debt in any case (DsaModel._clean_bond_repayment); rescaling
    here makes it independent of the units of the workbook (the Commission workbooks index nominal GDP, so debt is not
    in bn). Countries without Eurostat data are omitted. Fetches data from Eurostat (call once, before the runs).
    """
    from data_pipeline.maturity import fetch_maturity_data, repayment_profiles
    data = fetch_maturity_data(countries)
    T, esm, stock = {}, {}, {}
    for c in countries:
        series, params = read_country(input_file, c)
        T[c] = int(params['REFERENCE_YEAR'])
        esm[c] = series['ESM_REPAYMENT']
        stock[c] = (series.loc[T[c], 'DEBT_TOTAL'] * (1 - params['DEBT_ST_SHARE'])
                    - series.loc[T[c] + 1:, 'ESM_REPAYMENT'].fillna(0).sum())
    profiles, _ = repayment_profiles(data, countries, T, pd.DataFrame(esm), end_year=end_year)
    profiles = profiles / profiles.sum() * pd.Series(stock)[profiles.columns]
    return {c: {int(y): float(v) for y, v in profiles[c].dropna().items()} for c in profiles}


def build_oat_specs(latest_file=None, repayment_profiles=None):
    """
    One-at-a-time checks (baseline first). latest_file: workbook for the data vintage check (default latest_input_file()).
    repayment_profiles: Eurostat repayment profiles for the repayment profile check (eurostat_repayment_profiles), None
    omits it.
    """
    specs = [baseline_spec()]
    spec = make_spec

    # 1. Data
    latest_file = latest_file or latest_input_file()
    # Starting values (SPB, balances, debt in T) are data, not assumptions: they are not varied. The data vintage
    # check shows how revisions of the data (forecasts for T and later years replaced by outturns) change the targets.
    specs.append(spec('data:latest', 'Data', 'Data revisions',
                      f"latest vintage ({os.path.splitext(os.path.basename(latest_file))[0].replace('dsa_inputs_', '')})",
                      input_file=latest_file, bundle=False))
    # Repayment profile of long-term debt from Eurostat debt by residual maturity (bond_data=True) instead of the share
    # of long-term debt maturing each year. Countries without Eurostat data fail these runs (no result).
    if repayment_profiles is not None:
        specs.append(spec('data:repayment_profile', 'Data', 'Repayment profile (Eurostat)', 'residual maturity buckets',
                          model_kwargs={'bond_data': True}, bundle=False,
                          overrides=[{'code': 'BOND_REPAYMENT', 'years': 'values', 'values': repayment_profiles}]))

    # 2. Macro assumptions
    for name in ['rate_st_T10', 'rate_lt_T10', 'rate_st_T30', 'rate_lt_T30', 'inflation_T10', 'inflation_T30',
                 'potential_growth', 'ageing_scale']:
        specs += [assumption_spec(name, v) for v in ASSUMPTIONS[name][4]]
    specs.append(spec('macro:stock_flow_zero', 'Macro', 'Stock-flow adjustment', 'zero after T',
                      overrides=[{'code': 'STOCK_FLOW_RATIO', 'years': 'after', 'value': 0}]))
    specs += [spec(f'macro:ageing_cost_period:{p}', 'Macro', 'Ageing cost period', f'{p} years', value=p, base_value=10,
                   model_kwargs={'ageing_cost_period': p}) for p in [0, 5, 15]]
    specs += [assumption_spec('elasticity_scale', v) for v in ASSUMPTIONS['elasticity_scale'][4]]

    # 3. Fiscal multiplier
    specs += [assumption_spec('fiscal_multiplier', v) for v in ASSUMPTIONS['fiscal_multiplier'][4]]
    specs += [spec(f'multiplier:persistence:{p}', 'Multiplier', 'Multiplier persistence', f'{p} years', value=p,
                   base_value=3, model_kwargs={'fiscal_multiplier_persistence': p}) for p in [1, 2, 4, 5]]

    # 4. Stress-test calibration
    for name in ['adverse_r_g_shock', 'financial_stress_shock', 'lower_spb_shock']:
        specs += [assumption_spec(name, v) for v in ASSUMPTIONS[name][4]]

    # 5. Stochastic analysis
    specs += [spec(f'stochastic:prob_target:{p}', 'Stochastic', 'Probability target', f'{p:g}', value=p, base_value=0.7,
                   attributes={'prob_target': p}) for p in [0.6, 0.8, 0.9]]
    specs.append(spec('stochastic:period:10', 'Stochastic', 'Stochastic period', '10 years', value=10, base_value=5,
                      model_kwargs={'stochastic_period': 10}))
    specs.append(spec('stochastic:start:adjustment', 'Stochastic', 'Stochastic start', 'first adjustment year',
                      setup=[stochastic_start_at_adjustment]))
    specs += [spec(f'stochastic:sample_start:{y}', 'Stochastic', 'Shock sample start', str(y), value=y, base_value=2000,
                   model_kwargs={'shock_sample_start': y}) for y in [1990, 2010]]
    specs.append(spec('stochastic:frequency:annual', 'Stochastic', 'Shock frequency', 'annual',
                      model_kwargs={'shock_frequency': 'annual'}))
    specs += [spec(f'stochastic:estimation:{e}', 'Stochastic', 'Shock estimation', e, model_kwargs={'estimation': e})
              for e in ['var_cholesky', 'var_bootstrap']]
    specs.append(spec('stochastic:winsorize:off', 'Stochastic', 'Winsorised shocks', 'off',
                      model_kwargs={'winsorize_sample': False}))
    for criteria in (['debt_declines'], ['debt_declines', 'debt_below_60']):
        label = ' or '.join(c.replace('debt_', 'debt ').replace('_', ' ') for c in criteria)
        specs.append(spec(f'stochastic:criteria:{"+".join(criteria)}', 'Stochastic', 'Stochastic criteria', label,
                          binding_kwargs={'stochastic_criteria': criteria}))
    specs += [spec(f'noise:seed:{s}', 'Noise', 'Random seed', str(s), seed=s, bundle=False) for s in range(1, 10)]

    # 6. Rules
    specs.append(spec('rules:default', 'Rules', 'Rules', 'default rules', knob='rules', binding_kwargs={'rules': 'default'}))
    for label, options in RULE_SWITCHES.items():
        if label not in ('Commission rules', 'Default rules'):
            specs.append(spec(f'rules:switch:{label}', 'Rules', 'Rules', f'{label} switched to default', knob='rules',
                              binding_kwargs=dict(options)))
    # EDP status True/False overrides a Council decision (a fact, like the data) and is not used in the bundles
    specs += [spec(f'rules:edp_status:{s}', 'Rules', 'EDP status in T', str(s), binding_kwargs={'edp_status': s},
                   bundle=s == 'infer') for s in ['infer', True, False]]
    return specs


def build_global_specs(n_draws=200, seed=0, assumptions=GLOBAL_ASSUMPTIONS):
    """
    Global sensitivity analysis: n_draws Latin hypercube draws of the assumptions over the range of their one-at-a-time
    values, deterministic criteria only (stochastic=False). First spec is the deterministic baseline.
    """
    from scipy.stats import qmc
    sample = qmc.LatinHypercube(d=len(assumptions), seed=seed).random(n_draws)
    specs = [baseline_spec(stochastic=False)]
    for i, u in enumerate(sample):
        fields, params = {'model_kwargs': {}, 'attributes': {}, 'overrides': [], 'setup': []}, {}
        for name, x in zip(assumptions, u):
            lo, hi = min(ASSUMPTIONS[name][4]), max(ASSUMPTIONS[name][4])
            v = float(lo + x * (hi - lo))
            params[name] = v
            for k, part in ASSUMPTIONS[name][2](v).items():
                fields[k] = {**fields[k], **part} if isinstance(part, dict) else fields[k] + part
        specs.append(make_spec(f'global:{i:03d}', 'Global', 'Global draw', f'{i:03d}', binding_kwargs={'stochastic': False},
                               bundle=False, params=params, **fields))
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


def combine_specs(specs, id, label):
    """
    Combine specs into one (e.g. a bundle). Raises an error if two specs set the same model argument, attribute,
    input or rule option to different values.
    """
    out = make_spec(id, 'Bundles', 'Bundle', label, bundle=False)
    for s in specs:
        for field in ['model_kwargs', 'attributes', 'binding_kwargs']:
            for k, v in s[field].items():
                if k in out[field] and out[field][k] != v:
                    raise ValueError(f"Specs set {field}['{k}'] to different values ({out[field][k]} and {v})")
                out[field][k] = v
        codes = {o['code'] for o in out['overrides']}
        clash = codes & {o['code'] for o in s['overrides']}
        if clash:
            raise ValueError(f'Specs change the same inputs: {sorted(clash)}')
        out['overrides'] += s['overrides']
        out['setup'] += s['setup']
        if s['input_file']:
            raise ValueError('Specs with another input file cannot be combined')
    out['components'] = [s['id'] for s in specs]
    return out


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


def build_model(spec, country, adjustment_period, input_file, model_kwargs=None):
    """
    StochasticDsaModel for one spec, country and adjustment period (before find_spb_binding). model_kwargs are
    baseline model arguments of all runs; the spec's model arguments take precedence.
    """
    from classes import StochasticDsaModel as DSA  # local import avoids a circular import
    file = spec['input_file'] or input_file
    kwargs = {**(model_kwargs or {}), **spec['model_kwargs']}
    delay = kwargs.pop('adjustment_start_delay', 0)
    if delay:
        kwargs['adjustment_start_year'] = int(read_country(file, country)[1]['REFERENCE_YEAR']) + 1 + delay
    model = DSA(country, adjustment_period=adjustment_period, input_file=file,
                overrides=resolve_overrides(spec['overrides'], file, country), **kwargs)
    for k, v in spec['attributes'].items():
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
    key, spec, country, n, input_file, rules, seed, model_kwargs = task
    t0 = time.time()
    out = {}
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            model = build_model(spec, country, n, input_file, model_kwargs)
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


def task_key(spec, country, n, input_file, rules, seed, model_kwargs=None):
    """
    Cache key of a task: all fields that change results (labels are not part of the key).
    """
    fields = {k: spec[k] for k in ['model_kwargs', 'attributes', 'overrides', 'binding_kwargs', 'input_file', 'seed']}
    fields['setup'] = [_setup_signature(f) for f in spec['setup']]
    return json.dumps([fields, country, n, str(input_file), rules, seed, model_kwargs or {}], sort_keys=True, default=str)


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
              model_kwargs=None, cache_file=None, run=True, parallel=True, max_workers=None, save_every=100, verbose=True):
    """
    Run specs for all countries and adjustment periods and return one tidy DataFrame (one row per spec, country and
    adjustment period, with the spec labels). Results are cached by task in cache_file: tasks in the cache are not
    rerun. With run=False, only cached results are returned.

    Parameters:
        specs (list): Specs (build_oat_specs, build_global_specs, combine_specs).
        input_file (str): Baseline input workbook.
        rules (str): Baseline rules of find_spb_binding ('commission' or 'default').
        seed (int): Random seed set before each run (specs can set their own).
        model_kwargs (dict): Baseline model arguments of all runs (e.g. {'fiscal_multiplier_type': 'pers'}); the
            model arguments of a spec take precedence.
        max_workers (int): Number of worker processes (default: number of CPUs).
    """
    input_file = input_file or latest_input_file()
    cache = _load_cache(cache_file)
    tasks = [(task_key(s, c, n, input_file, rules, seed, model_kwargs), s, c, n, input_file, rules, seed, model_kwargs)
             for s in specs for n in adjustment_periods for c in countries]
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


def add_deviations(df, baseline_id='baseline'):
    """
    Add deviations from the baseline run of the same country and adjustment period, for both targets (see TARGETS):
    change of the target (pp.), of the annual adjustment and of debt 10 years after the adjustment period, and whether
    the binding criterion differs from the baseline.
    """
    keys = ['country', 'adjustment_period']
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


CRITERION_GROUPS = ['Debt, deterministic scenarios', 'Debt, stochastic', 'Deficit below 3%', 'Debt safeguard',
                    'Non-negative adjustment', 'Other']


def criterion_group(criterion):
    """
    Broad group of a binding criterion.
    """
    if not isinstance(criterion, str):
        return 'Other'
    if criterion.startswith(('debt_declines', 'debt_below_60')) or criterion in (
            'main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g'):
        return 'Debt, deterministic scenarios'
    return {'stochastic': 'Debt, stochastic', 'deficit_reduction': 'Deficit below 3%', 'debt_safeguard': 'Debt safeguard',
            'floor': 'Non-negative adjustment'}.get(criterion, 'Other')


def check_effects(df, target='dsa', noise=None):
    """
    Effect of each spec on the target: median, lower and upper quartile of the change across countries and adjustment
    periods, largest absolute change, share of runs with a switch of the binding criterion, and the median noise band.
    """
    dt, sw = TARGETS[target]['delta_target'], TARGETS[target]['criterion_switch']
    effects = df.groupby(['id', 'group', 'check', 'variant', 'knob'], sort=False).agg(
        median=(dt, 'median'), q25=(dt, lambda x: x.quantile(0.25)), q75=(dt, lambda x: x.quantile(0.75)),
        max_abs=(dt, lambda x: x.abs().max()), criterion_switches=(sw, 'mean'), runs=(dt, 'count')).reset_index()
    if noise is not None:
        effects['noise'] = (noise['noise_high'] - noise['noise_low']).median() / 2
    return effects


def define_bundles(df, specs, noise_halfwidth, min_effect=0.0, target='dsa', exclude_groups=('Data',)):
    """
    Accommodative and strict bundles from the one-at-a-time results. For each knob (setting changed by a spec), the
    accommodative bundle takes the variant with the lowest median effect on the target across countries and adjustment
    periods, the strict bundle the variant with the highest, provided the effect exceeds the noise band in that
    direction and min_effect (pp.). Specs marked bundle=False (baseline, data vintage, seeds, EDP status) and groups in
    exclude_groups are left out. Variants are added in order of the size of their effect; a variant that sets an option
    already set by a variant with a larger effect is skipped (status 'conflict'). Returns (accommodative spec, strict
    spec, table).
    """
    by_id = {s['id']: s for s in specs}
    eligible = df[df['id'].map(lambda i: by_id[i]['bundle']) & ~df['group'].isin(exclude_groups)]
    effects = eligible.groupby(['knob', 'id', 'check', 'variant'], sort=False)[TARGETS[target]['delta_target']].median()
    effects = effects.rename('median').reset_index()
    threshold = max(noise_halfwidth, min_effect)
    picks = []
    for knob, g in effects.groupby('knob', sort=False):
        low, high = g.loc[g['median'].idxmin()], g.loc[g['median'].idxmax()]
        if low['median'] < -threshold:
            picks.append({**low.to_dict(), 'bundle': 'accommodative'})
        if high['median'] > threshold:
            picks.append({**high.to_dict(), 'bundle': 'strict'})
    table = pd.DataFrame(picks, columns=['knob', 'id', 'check', 'variant', 'median', 'bundle'])
    table = table.reindex(table['median'].abs().sort_values(ascending=False).index)
    bundles, status = {}, {}
    for name in ['accommodative', 'strict']:
        selected = []
        for i in table.loc[table['bundle'] == name, 'id']:
            try:
                combine_specs([by_id[j] for j in selected + [i]], 'check', '')
                selected.append(i)
                status[(name, i)] = 'included'
            except ValueError:
                status[(name, i)] = 'conflict'
        bundles[name] = combine_specs([by_id[i] for i in selected], f'bundle:{name}', f'{name} bundle')
    table['status'] = [status[(b, i)] for b, i in zip(table['bundle'], table['id'])]
    table = table[['bundle', 'check', 'variant', 'median', 'status', 'id']].sort_values(['bundle', 'median'])
    return bundles['accommodative'], bundles['strict'], table.reset_index(drop=True)


def target_comparison(df, exclude_groups=('Noise', 'Bundles', 'Global', 'Baseline')):
    """
    Effect of each check on the DSA target and on the binding target: median absolute change across countries,
    adjustment periods and variants, and the share of runs in which the EDP or the safeguards raise the binding target
    above the DSA target (baseline share in column 'baseline_share_above_dsa').
    """
    d = df[~df['group'].isin(exclude_groups)].copy()
    d['above_dsa'] = d['binding_target'] > d['dsa_target'] + 1e-3
    dsa, binding = TARGETS['dsa']['delta_target'], TARGETS['binding']['delta_target']
    out = d.groupby(['group', 'check'], sort=False).agg(
        dsa=(dsa, lambda x: x.abs().median()), binding=(binding, lambda x: x.abs().median()),
        dsa_mean=(dsa, lambda x: x.abs().mean()), binding_mean=(binding, lambda x: x.abs().mean()),
        share_above_dsa=('above_dsa', 'mean')).reset_index()
    base = df[df['id'] == 'baseline']
    out['baseline_share_above_dsa'] = (base['binding_target'] > base['dsa_target'] + 1e-3).mean()
    return out


def variance_decomposition(df, target='dsa', assumptions=GLOBAL_ASSUMPTIONS, categories=GLOBAL_CATEGORIES):
    """
    Variance decomposition of the global sensitivity analysis by country and adjustment period: squared standardised
    regression coefficients of a linear regression of the target on the drawn assumptions (share of the variance of the
    target explained by each assumption; the draws are close to uncorrelated), summed by category. The remainder
    (1 - R2) is the variance from non-linear effects and interactions, e.g. switches of the binding criterion.
    """
    y = TARGETS[target]['target']
    rows = []
    for (c, n), g in df.dropna(subset=[y]).groupby(['country', 'adjustment_period']):
        X, Y = g[assumptions].to_numpy(float), g[y].to_numpy(float)
        if len(g) <= len(assumptions) + 1 or Y.std() < 1e-10:
            continue
        Xs = np.column_stack([np.ones(len(Y)), (X - X.mean(0)) / X.std(0)])
        coef = np.linalg.lstsq(Xs, Y, rcond=None)[0]
        shares = pd.Series(coef[1:] ** 2 / Y.var(), index=assumptions)
        r2 = 1 - np.var(Y - Xs @ coef) / Y.var()
        row = {'country': c, 'adjustment_period': n, 'target': target, 'std': Y.std(), 'r2': r2}
        row.update({k: shares[v].sum() for k, v in categories.items()})
        row['Non-linear and interactions'] = max(0.0, 1 - r2)
        row.update({f'src2_{k}': v for k, v in shares.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ========================================================================================= #
#                                           CHARTS                                          #
# ========================================================================================= #

def _order_countries(df, period, target='dsa'):
    base = df[(df['id'] == 'baseline') & (df['adjustment_period'] == period)]
    return base.sort_values(TARGETS[target]['target'])['country'].tolist()


def _despine(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


def plot_heatmap(df, adjustment_period, target='dsa', countries=None, groups=None, vmax=1.0, figsize=None, title=None):
    """
    Heatmap of the change of the target by country (rows) and check variant (columns); dots mark a switch of the
    binding criterion. Countries are ordered by their baseline target.
    """
    cols = TARGETS[target]
    d = df[(df['adjustment_period'] == adjustment_period) & (df['id'] != 'baseline')]
    if groups is not None:
        d = d[d['group'].isin(groups)]
    columns = list(dict.fromkeys(d['id']))
    countries = countries or _order_countries(df, adjustment_period, target)[::-1]
    values = d.pivot_table(index='country', columns='id', values=cols['delta_target'], aggfunc='first')
    values = values.reindex(index=countries, columns=columns)
    switch = d.pivot_table(index='country', columns='id', values=cols['criterion_switch'], aggfunc='first')
    switch = switch.reindex(index=countries, columns=columns)
    labels = d.drop_duplicates('id').set_index('id')
    figsize = figsize or (max(10, 0.26 * len(columns) + 3), 0.3 * len(countries) + 3.2)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values.to_numpy(float), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect='auto', interpolation='none')
    yy, xx = np.where(switch.fillna(False).to_numpy(bool))
    ax.scatter(xx, yy, s=9, color=INK, marker='o', linewidths=0, label='Binding criterion switches')
    ax.set_yticks(range(len(countries)), countries, fontsize=9)
    ax.set_xticks(range(len(columns)), [f"{labels.loc[i, 'check']}: {labels.loc[i, 'variant']}" for i in columns],
                  rotation=90, fontsize=7.5)
    grp = labels.loc[columns, 'group'].tolist()
    starts = [0] + [i for i in range(1, len(grp)) if grp[i] != grp[i - 1]]
    for k, st in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(grp)
        if st > 0:
            ax.axvline(st - 0.5, color='white', lw=3)
        ax.text((st + end - 1) / 2, -0.9, grp[st], ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend='both')
    cbar.set_label('Change vs. baseline (pp.)')
    cbar.outline.set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.07, 1.0), frameon=False, fontsize=9)
    ax.set_title(title or f"Sensitivity of the {cols['label']}, {adjustment_period}-year adjustment", loc='left', pad=28)
    plt.tight_layout()
    return fig


def plot_ranges(df, adjustment_period, target='dsa', bundles=('bundle:accommodative', 'bundle:strict'),
                exclude_groups=('Noise', 'Global'), figsize=None):
    """
    Range of targets across one-at-a-time checks by country, with the Commission target, the baseline, the bundles and
    the SPB in T.
    """
    cols = TARGETS[target]
    d = df[df['adjustment_period'] == adjustment_period]
    countries = _order_countries(d, adjustment_period, target)
    oat = d[~d['group'].isin(exclude_groups + ('Bundles',))]
    rng = oat.groupby('country')[cols['target']].agg(['min', 'max']).reindex(countries)
    base = d[d['id'] == 'baseline'].set_index('country').reindex(countries)
    fig, ax = plt.subplots(figsize=figsize or (9, 0.33 * len(countries) + 1.8))
    y = np.arange(len(countries))
    ax.hlines(y, rng['min'], rng['max'], color=LIGHT_GREY, lw=7, zorder=1, label='Range of one-at-a-time checks',
              capstyle='round')
    ax.scatter(base['spb_T'], y, marker='|', s=140, lw=2, color=CURRENT, zorder=2, label='SPB in T')
    for bid, color, label, marker in [(bundles[0], ACCOMMODATIVE, 'Accommodative bundle', '<'),
                                      (bundles[1], STRICT, 'Strict bundle', '>')]:
        b = d[d['id'] == bid].set_index('country').reindex(countries)
        if len(b) and b[cols['target']].notna().any():
            ax.scatter(b[cols['target']], y, marker=marker, s=48, color=color, zorder=3, label=label,
                       edgecolor='white', linewidth=0.8)
    ax.scatter(base[cols['target']], y, marker='o', s=50, color=BASELINE, zorder=4, label='Model baseline',
               edgecolor='white', linewidth=0.8)
    if cols['commission'] in base and base[cols['commission']].notna().any():
        ax.scatter(base[cols['commission']], y, marker='D', s=26, facecolor='none', edgecolor=COMMISSION, lw=1.3,
                   zorder=5, label='Commission')
    ax.set_yticks(y, countries)
    ax.set_ylim(-0.7, len(countries) - 0.3)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('SPB at the end of the adjustment period (% of GDP)')
    ax.set_title(f"{cols['label'][0].upper() + cols['label'][1:]}s, {adjustment_period}-year adjustment", loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    _despine(ax)
    plt.tight_layout()
    return fig


def tornado_data(df, country=None, adjustment_period=4, target='dsa',
                 exclude_groups=('Noise', 'Bundles', 'Global', 'Baseline')):
    """
    Lowest and highest change of the target by check for one country, or the median across countries (country=None).
    """
    dt = TARGETS[target]['delta_target']
    d = df[(df['adjustment_period'] == adjustment_period) & ~df['group'].isin(exclude_groups)]
    if country is not None:
        d = d[d['country'] == country]
    per_variant = d.groupby(['group', 'check', 'variant'], sort=False)[dt].median().reset_index()
    rows = []
    for (g, c), v in per_variant.groupby(['group', 'check'], sort=False):
        lo, hi = v.loc[v[dt].idxmin()], v.loc[v[dt].idxmax()]
        rows.append({'group': g, 'check': c, 'low': min(lo[dt], 0), 'high': max(hi[dt], 0),
                     'low_variant': lo['variant'] if lo[dt] < 0 else '', 'high_variant': hi['variant'] if hi[dt] > 0 else ''})
    t = pd.DataFrame(rows)
    t['span'] = t['high'] - t['low']
    return t.sort_values('span')


def plot_tornado(df, noise, country=None, adjustment_period=4, target='dsa', top=18, ax=None):
    """
    Tornado chart: range of the change of the target by check (lower target in blue, higher in red), with the noise
    band from alternative seeds (grey). country=None shows the median across countries.
    """
    t = tornado_data(df, country, adjustment_period, target).tail(top)
    nb = noise[noise['adjustment_period'] == adjustment_period]
    if country is not None:
        nb = nb[nb['country'] == country]
    lo, hi = nb['noise_low'].median(), nb['noise_high'].median()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.axvspan(lo, hi, color=LIGHT_GREY, zorder=0, label='Noise band (seeds)')
    ax.barh(y, t['low'], color=LOWER, height=0.62, zorder=2, label='Lower target')
    ax.barh(y, t['high'], color=HIGHER, height=0.62, zorder=2, label='Higher target')
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    span = max(t['high'].max(), -t['low'].min(), 0.05)
    for yi, (_, r) in zip(y, t.iterrows()):
        if r['low_variant']:
            ax.text(r['low'] - 0.02 * span, yi, r['low_variant'], ha='right', va='center', fontsize=8, color=MUTED)
        if r['high_variant']:
            ax.text(r['high'] + 0.02 * span, yi, r['high_variant'], ha='left', va='center', fontsize=8, color=MUTED)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_yticks(y, t['check'], fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f"Change in {TARGETS[target]['label']} (pp.)")
    ax.set_title(f"{country or 'Median across countries'}, {adjustment_period}-year adjustment", loc='left', fontsize=12)
    _despine(ax)
    return ax


def plot_response_curves(df, specs, adjustment_period=4, target='dsa', ncols=4, highlight=None):
    """
    Small multiples: change of the target by value of each numeric assumption, one line per country (grey) and the
    median across countries (bold). The baseline point (zero change) is included.
    """
    dt = TARGETS[target]['delta_target']
    by_id = {s['id']: s for s in specs}
    d = df[(df['adjustment_period'] == adjustment_period) & df['value'].notna()
           & ~df['group'].isin(['Noise', 'Bundles', 'Global', 'Baseline'])]
    checks = [c for c, g in d.groupby('check', sort=False) if g['variant'].nunique() >= 2]
    nrows = int(np.ceil(len(checks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, check in zip(axes, checks):
        g = d[d['check'] == check]
        base_value = by_id[g['id'].iloc[0]]['base_value']
        curves = {}
        for c, gc in g.groupby('country'):
            x0 = gc[base_value].iloc[0] if isinstance(base_value, str) else base_value
            pts = pd.concat([gc[['value', dt]], pd.DataFrame({'value': [x0], dt: [0.0]})])
            curves[c] = pts.groupby('value')[dt].mean().sort_index()
            ax.plot(curves[c].index, curves[c].values, color=GREY, lw=0.8, alpha=0.45, zorder=1)
        med = pd.DataFrame(curves).median(axis=1)
        ax.plot(med.index, med.values, color=INK, lw=2.2, marker='o', ms=4, zorder=3)
        if highlight in curves:
            ax.plot(curves[highlight].index, curves[highlight].values, color=COLORS[1], lw=1.8, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(check, fontsize=10, loc='left')
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    for ax in axes[len(checks):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel('Change (pp.)', fontsize=9)
    handles = [Line2D([], [], color=INK, lw=2.2, marker='o', ms=4, label='EU median'),
               Line2D([], [], color=GREY, lw=0.8, label='Countries')]
    if highlight:
        handles.append(Line2D([], [], color=COLORS[1], lw=1.8, label=highlight))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    fig.legend(handles=handles, loc='upper right', ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle(f"Response of the {TARGETS[target]['label']}, {adjustment_period}-year adjustment", x=0.01, y=0.995,
                 ha='left', va='top', fontsize=13, fontweight='bold')
    return fig


def plot_criterion_shares(df, adjustment_period=4, target='dsa', by='check', exclude_groups=('Noise', 'Global')):
    """
    Share of runs by group of the binding criterion (see criterion_group), for the baseline and each check (share across
    all variants of a check) or each variant (by='id').
    """
    d = df[(df['adjustment_period'] == adjustment_period) & ~df['group'].isin(exclude_groups)].copy()
    d['criterion_group'] = d[TARGETS[target]['criterion']].map(criterion_group)
    d['label'] = d['check'] if by == 'check' else d['check'] + ': ' + d['variant'].astype(str)
    shares = pd.crosstab(d['label'], d['criterion_group'], normalize='index').reindex(columns=CRITERION_GROUPS, fill_value=0)
    shares = shares.loc[list(dict.fromkeys(d['label']))]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(shares) + 1.8))
    left = np.zeros(len(shares))
    y = np.arange(len(shares))[::-1]
    colors = dict(zip(CRITERION_GROUPS, [COLORS[0], COLORS[2], COLORS[1], COLORS[6], COLORS[3], GREY]))
    for cg in CRITERION_GROUPS:
        if shares[cg].sum() == 0:
            continue
        ax.barh(y, shares[cg], left=left, color=colors[cg], height=0.72, label=cg, edgecolor='white', linewidth=1)
        left += shares[cg].to_numpy()
    ax.set_yticks(y, shares.index, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', visible=False)
    ax.set_title(f"Binding criterion of the {TARGETS[target]['label']}, {adjustment_period}-year adjustment", loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    _despine(ax)
    plt.tight_layout()
    return fig


def plot_target_comparison(df):
    """
    Two panels by check: mean absolute change of the DSA target and of the binding target (left), and the share of
    runs in which the EDP or the safeguards raise the binding target above the DSA target (right).
    """
    t = target_comparison(df).sort_values('dsa_mean')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.32 * len(t) + 1.8), sharey=True,
                                   gridspec_kw={'width_ratios': [2.2, 1]})
    y = np.arange(len(t))
    ax1.hlines(y, t[['dsa_mean', 'binding_mean']].min(axis=1), t[['dsa_mean', 'binding_mean']].max(axis=1),
               color=LIGHT_GREY, lw=3, zorder=1)
    ax1.scatter(t['dsa_mean'], y, color=COLORS[0], s=40, zorder=3, label='DSA criteria only', edgecolor='white', lw=0.8)
    ax1.scatter(t['binding_mean'], y, color=COLORS[1], s=40, zorder=3, marker='D', label='With EDP and safeguards',
                edgecolor='white', lw=0.8)
    ax1.set_yticks(y, t['check'], fontsize=9)
    ax1.set_xlabel('Mean absolute change of the SPB target (pp.)')
    ax1.grid(axis='y', visible=False)
    ax1.legend(loc='lower right', frameon=False, fontsize=9)
    ax1.set_title('Effect on the SPB target', loc='left', fontsize=12)
    ax2.barh(y, t['share_above_dsa'], color=COLORS[6], height=0.62, zorder=2)
    ax2.axvline(t['baseline_share_above_dsa'].iloc[0], color=INK, lw=1.2, ls='--', zorder=3, label='Baseline')
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.set_xlabel('Share of runs')
    ax2.grid(axis='y', visible=False)
    ax2.legend(loc='lower right', frameon=False, fontsize=9)
    ax2.set_title('EDP or safeguards raise the target', loc='left', fontsize=12)
    for ax in (ax1, ax2):
        _despine(ax)
    plt.tight_layout()
    return fig


def plot_debt_paths(df, country, adjustment_period=4, target='dsa',
                    highlight=('baseline', 'bundle:accommodative', 'bundle:strict'), exclude_groups=('Global',)):
    """
    Debt paths for one country (adjustment to the target of each run, baseline assumptions of each run): all checks in
    grey, baseline and bundles highlighted.
    """
    col = TARGETS[target]['debt_path']
    d = df[(df['country'] == country) & (df['adjustment_period'] == adjustment_period) & ~df['group'].isin(exclude_groups)]
    d = d[d[col].notna()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, r in d[~d['id'].isin(highlight)].iterrows():
        p = pd.Series(r[col])
        ax.plot(p.index, p.values, color=GREY, lw=0.8, alpha=0.35, zorder=1)
    styles = {'baseline': (BASELINE, 'Model baseline'), 'bundle:accommodative': (ACCOMMODATIVE, 'Accommodative bundle'),
              'bundle:strict': (STRICT, 'Strict bundle')}
    for hid in highlight:
        r = d[d['id'] == hid]
        if len(r):
            p = pd.Series(r[col].iloc[0])
            color, label = styles.get(hid, (COLORS[4], hid))
            ax.plot(p.index, p.values, color=color, lw=2.4, zorder=3, label=label)
            ax.annotate(f'{p.iloc[-1]:.0f}', (p.index[-1], p.iloc[-1]), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=9, color=MUTED)
    base = d[d['id'] == 'baseline']
    if len(base):
        end = base['adjustment_end_year'].iloc[0]
        ax.axvspan(end - adjustment_period + 0.5, end + 0.5, color=LIGHT_GREY, alpha=0.5, zorder=0,
                   label='Adjustment period (baseline)')
    ax.plot([], [], color=GREY, lw=0.8, label='One-at-a-time checks')
    ax.set_ylabel('Debt (% of GDP)')
    ax.set_title(f"{country}: debt, adjustment to the {TARGETS[target]['label']}, {adjustment_period}-year adjustment",
                 loc='left')
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _despine(ax)
    plt.tight_layout()
    return fig


def plot_variance_shares(vd, adjustment_period=4, categories=GLOBAL_CATEGORIES):
    """
    Stacked bars: share of the variance of the target by category of assumptions (global sensitivity analysis),
    countries ordered by the standard deviation of the target (shown on the right).
    """
    d = vd[vd['adjustment_period'] == adjustment_period].sort_values('std')
    target = d['target'].iloc[0] if 'target' in d and len(d) else 'dsa'
    cols = list(categories) + ['Non-linear and interactions']
    colors = dict(zip(cols, COLORS[:len(categories)] + [LIGHT_GREY]))
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(d) + 2))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    total = d[cols].sum(axis=1).to_numpy()
    for c in cols:
        w = d[c].to_numpy() / total
        ax.barh(y, w, left=left, color=colors[c], height=0.72, label=c, edgecolor='white', linewidth=1)
        left += w
    for yi, s in zip(y, d['std']):
        ax.text(1.01, yi, f'{s:.2f}', va='center', fontsize=8, color=MUTED)
    ax.text(1.01, len(d) - 0.2, 'SD (pp.)', fontsize=8, color=MUTED, va='bottom')
    ax.set_yticks(y, d['country'], fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', visible=False)
    ax.set_title(f"Variance of the {TARGETS[target]['label']} by assumption, {adjustment_period}-year adjustment",
                 loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06 - 0.3 / max(len(d), 1)), ncol=4, frameon=False, fontsize=9)
    _despine(ax)
    plt.tight_layout()
    return fig
