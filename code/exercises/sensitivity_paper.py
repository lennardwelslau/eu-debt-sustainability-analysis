# ========================================================================================= #
#      Sensitivity of EU DSA - tables and figures of the working paper                       #
# ========================================================================================= #
#
# Helper functions for sensitivity_analysis.ipynb. They turn the cached runs of code/functions/sensitivity.py into the
# main tables (as DataFrames) and figures of the working paper "The Sensitivity of EU Debt Sustainability Analysis to
# Forecast Errors and Calibration Conventions" (Welslau, 2026).
#
# Contents:
#   1. Settings of the paper.
#   2. Runs: load_runs reads (and with run=True computes) the individual checks, the alternative settings of the
#      calibration conventions, the runs with the October 2026 data and the joint draws.
#   3. Samples: countries that must adjust, by regime.
#   4. Tables: baseline targets, effects of the individual checks, joint draws by country, comparison of forecasts and
#      calibration conventions, October 2026 data, largest effects by country.
#   5. Figures: SPB required by each criterion, debt sustainability safeguard and multiplier, targets in the joint
#      draws, forecasts against calibration conventions, debt under the baseline SPB path.
#
# Author: Lennard Welslau
# Updated: 2026-11-05
# ========================================================================================= #

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_pipeline import EU27, latest_input_file
from functions.sensitivity import (OUTPUT_DIR, JOINT_RUNS, METHODS, EMPIRICAL_DRAWS_PAIRED, FORECASTS,
                                   MODEL_PARAMETERS, ESTIMATION, RISK_STANDARD, DATA, TIERS, build_oat_specs,
                                   build_menu_specs, build_global_specs, run_specs, fixed_path_runs, commission_targets,
                                   add_deviations, add_own_criterion, effects_table, feasible, forecast_error_samples,
                                   applicable_dsa_criteria, dsa_target_criteria_only, paired_error_samples,
                                   baseline_spec, build_model)


# ========================================================================================= #
#                                       1. SETTINGS                                         #
# ========================================================================================= #

INPUT_FILE = 'dsa_inputs_commission_2024.xlsx'  # baseline: Commission prior guidance of 2024
METHOD = '2024'                   # Commission method of the baseline: multiplier 0.75, normal shocks
LATEST_METHODS = ('2024', '2026')  # methods with which the October 2026 data are run
LATEST_METHOD = '2026'            # method of the October 2026 data in the paper: multiplier 0.6, AR bootstrap
RULES = 'commission'              # Commission rules, without rounding of the annual adjustment
ATTRIBUTES = {'adjustment_bound': 5}  # largest annual adjustment searched (pp.); safeguard infeasible above
N_DRAWS = 500                     # joint draws of forecasts and model parameters
REAL_RATES = True                 # rate assumptions are real rates: nominal rates move with inflation
LARGE_ADJUSTMENT = 5              # total adjustment over four years counted as very large (pp.)
GROWTH_BIAS_ID = 'macro:potential_growth:bias'  # check with the uncorrected mean growth error (footnote only)
FIXED_PATHS_FILE = 'joint_debt_fixed_spb.pkl'  # debt under the baseline SPB path in the joint draws


def run_options(countries, run=False, period=None, method=METHOD, latest=False):
    """
    Keyword arguments of run_specs with the settings of the paper.
    """
    return dict(countries=countries, adjustment_periods=(4, 7) if period is None else (period,),
                input_file=latest_input_file() if latest else INPUT_FILE, rules=RULES, seed=0, run=run,
                model_kwargs=METHODS[method], attributes=ATTRIBUTES, verbose=run)


# ========================================================================================= #
#                                    2. RUNS                                                #
# ========================================================================================= #

def oat_specs(method=METHOD):
    """
    Individual checks.
    """
    return build_oat_specs(latest_file=latest_input_file(), real_rates=REAL_RATES, method=method)


def latest_specs(method):
    """
    Individual checks and alternative settings of the calibration conventions run with the October 2026 data.
    """
    specs = [s for s in oat_specs(method) if s['id'] != 'data:latest' and not s['id'].startswith('noise:')]
    return specs + [s for s in build_menu_specs() if s['id'] not in {x['id'] for x in specs}]


def joint_specs(n_draws=N_DRAWS, paired=False):
    """
    Joint draws of forecasts and model parameters. paired: growth, inflation and rate errors of the same year, with
    the rate errors applied to nominal rates (robustness check).
    """
    if paired:
        return build_global_specs(n_draws=n_draws, seed=0, assumptions=JOINT_RUNS['paired'], real_rates=False,
                                  draws=EMPIRICAL_DRAWS_PAIRED, errors=paired_error_samples())
    return build_global_specs(n_draws=n_draws, seed=0, assumptions=JOINT_RUNS['joint'], real_rates=REAL_RATES)


def add_tiers(runs):
    """
    DSA target from the DSA criteria alone (no adjustment floor) and the change of the own criterion, the DSA target
    and the binding target relative to the baseline.
    """
    return add_own_criterion(add_deviations(dsa_target_criteria_only(runs)))


def load_runs(countries=EU27, n_draws=N_DRAWS, cache_dir=OUTPUT_DIR, run=False, paired=False):
    """
    All runs of the paper from the cache in cache_dir; with run=True, runs missing from the cache are computed first.
    Returns a dict with
        oat:     individual checks, 2024 data, four- and seven-year adjustment, with the Commission's targets;
        menu:    alternative settings of the calibration conventions, 2024 data;
        latest:  {method: individual checks and settings with the October 2026 data, four-year adjustment};
        joint:   joint draws, four-year adjustment (None if not all draws are in the cache);
        paired:  paired joint draws (only if paired=True; None if not all draws are in the cache).
    """
    out = {}
    oat_file = cache_dir / 'oat_runs.pkl'
    specs = oat_specs()
    menu = build_menu_specs()
    if run:
        run_specs(specs + [s for s in menu if s['id'] not in {x['id'] for x in specs}], cache_file=oat_file,
                  **run_options(countries, run=True))
    oat = run_specs(specs, cache_file=oat_file, **run_options(countries))
    oat = oat.merge(commission_targets(countries, vintage='2024'), on=['country', 'adjustment_period'], how='left')
    out['oat'] = add_tiers(oat)
    out['menu'] = add_tiers(run_specs(menu, cache_file=oat_file, **run_options(countries)))
    out['latest'] = {m: add_tiers(run_specs(latest_specs(m), cache_file=cache_dir / f'oat_runs_latest_{m}.pkl',
                                            **run_options(countries, run, period=4, method=m, latest=True)))
                     for m in LATEST_METHODS}
    if run:  # seven-year baseline with the October 2026 data
        latest_seven_year(countries, cache_dir, run=True)
    out['joint'] = load_draws(countries, n_draws, cache_dir, run=run)
    if paired:
        out['paired'] = load_draws(countries, n_draws, cache_dir, run=run, paired=True)
    if run:  # debt under the baseline SPB path in the joint draws
        fixed_path_runs(joint_specs(n_draws), countries, 4, INPUT_FILE, RULES, 0, METHODS[METHOD], ATTRIBUTES,
                        cache_file=cache_dir / FIXED_PATHS_FILE)
    return out


def load_draws(countries=EU27, n_draws=N_DRAWS, cache_dir=OUTPUT_DIR, run=False, paired=False):
    """
    Joint draws (or paired joint draws), four-year adjustment; None if not all draws are in the cache.
    """
    file = 'global_runs_joint_paired.pkl' if paired else 'global_runs_joint.pkl'
    d = run_specs(joint_specs(n_draws, paired), cache_file=cache_dir / file, **run_options(countries, run, period=4))
    if len(d) < (n_draws + 1) * len(countries):
        return None
    return add_tiers(d)


def latest_seven_year(countries=EU27, cache_dir=OUTPUT_DIR, run=False):
    """
    Baseline with the October 2026 data and the 2026 method, seven-year adjustment.
    """
    specs = [s for s in latest_specs(LATEST_METHOD) if s['id'] == 'baseline']
    d = run_specs(specs, cache_file=cache_dir / f'oat_runs_latest_{LATEST_METHOD}.pkl',
                  **run_options(countries, run, period=7, method=LATEST_METHOD, latest=True))
    return add_deviations(dsa_target_criteria_only(d))


def fixed_paths(countries=EU27, n_draws=N_DRAWS, cache_dir=OUTPUT_DIR):
    """
    Debt paths of the joint draws under the SPB path of the baseline binding target, from the cache; None if not
    computed. The cache is keyed by draw id and country only: delete it after changing the error data or settings.
    """
    path = cache_dir / FIXED_PATHS_FILE
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        rows = pickle.load(f)
    ids = {s['id'] for s in joint_specs(n_draws)}
    return pd.DataFrame([{'id': i, 'country': c, 'debt_path': out['debt_path']} for (i, c), out in rows.items()
                         if out.get('debt_path') and i in ids and c in countries])


# ========================================================================================= #
#                                    3. SAMPLES                                             #
# ========================================================================================= #

def baseline(oat, period=4):
    """
    Baseline runs by country.
    """
    return oat[(oat['id'] == 'baseline') & (oat['adjustment_period'] == period)].set_index('country')


def by_debt(oat):
    """
    Countries ordered by debt in 2024, highest first.
    """
    return list(baseline(oat).sort_values('debt_T', ascending=False).index)


def regimes(oat, period=4):
    """
    Countries that must adjust (baseline binding target above the SPB in 2024), split into those that receive a
    reference trajectory (the paper's main sample) and those that receive technical information.
    """
    base = baseline(oat, period)
    adjust = base.index[base['binding_target'] - base['spb_T'] > 1e-9]
    ref = [c for c in adjust if base.loc[c, 'guidance'] == 'reference trajectory']
    return {'Reference trajectory': ref, 'Technical information': [c for c in adjust if c not in ref]}


# ========================================================================================= #
#                                    4. TABLES                                              #
# ========================================================================================= #

NAMES = {'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia', 'CYP': 'Cyprus', 'CZE': 'Czechia',
         'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
         'HUN': 'Hungary', 'IRL': 'Ireland', 'ITA': 'Italy', 'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
         'MLT': 'Malta', 'NLD': 'Netherlands', 'POL': 'Poland', 'PRT': 'Portugal', 'ROU': 'Romania', 'SVK': 'Slovakia',
         'SVN': 'Slovenia', 'ESP': 'Spain', 'SWE': 'Sweden'}
BLOCK_TITLES = {FORECASTS: '1. Forecasts', MODEL_PARAMETERS: '2. Model parameters', ESTIMATION: '3. Estimation choices',
                RISK_STANDARD: '4. Calibration conventions', DATA: 'Data'}
CRITERIA = {'debt_declines_adverse_r_g': 'Adverse r-g test: debt declines',
            'debt_declines_lower_spb': 'Lower-SPB test: debt declines',
            'debt_declines_financial_stress': 'Financial stress test: debt declines',
            'debt_declines_main_adjustment': 'Baseline: debt declines',
            'debt_below_60_adverse_r_g': 'Adverse r-g test: debt below 60%',
            'debt_below_60_lower_spb': 'Lower-SPB test: debt below 60%',
            'debt_below_60_main_adjustment': 'Baseline: debt below 60%',
            'deficit_reduction': 'Deficit below 3%', 'stochastic': 'Stochastic', 'floor': 'Adjustment floor',
            'debt_safeguard': 'Debt safeguard', 'deficit_resilience': 'Deficit resilience', 'edp': 'EDP'}
CHECKS = {'Short-term rate T+10': 'Real short-term rate, T+10', 'Long-term rate T+10': 'Real long-term rate, T+10',
          'Short-term rate T+30': 'Real short-term rate, T+30', 'Long-term rate T+30': 'Real long-term rate, T+30',
          'Ageing costs (shift)': 'Ageing costs', 'Ageing costs (vintage)': 'Ageing costs, previous vintage',
          'Ageing costs (scenario)': 'Ageing costs, 2024 AR scenario',
          'Budget balance elasticity (scale)': 'Budget semi-elasticity', 'Stochastic start': 'Start of stochastic projection',
          'Stochastic period': 'Horizon of stochastic projection', 'Shock sample start': 'Start of shock sample',
          'Shock estimation': 'Shock model', 'Probability target': 'Probability threshold',
          'Lower SPB shock': 'Lower-SPB shock', 'Calibration scenario': 'Scenario',
          'Repayment profile (Eurostat)': 'Repayment profile'}
VARIANTS = {'residual maturity buckets': 'Eurostat profile', 'var_bootstrap': 'VAR bootstrap',
            'ar_bootstrap': 'AR bootstrap', 'var_bootstrap, with intercept': 'VAR bootstrap, intercept',
            'ar_bootstrap, with intercept': 'AR bootstrap, intercept',
            'ar_bootstrap, not winsorised': 'AR bootstrap, not winsorised', '1st and 99th percentiles': 'P1 and P99',
            'lenient_margins': 'lenient, margins only', 'strict_margins': 'strict, margins only',
            'higher migration (+33%)': 'higher migration', 'lower fertility (-20%)': 'lower fertility'}
PP_CHECKS = ['Short-term rate T+10', 'Long-term rate T+10', 'Short-term rate T+30', 'Long-term rate T+30',
             'Inflation T+10', 'Inflation T+30', 'Potential growth T+10', 'Potential growth (mean error)',
             'Ageing costs (shift)', 'Adverse r-g shock', 'Financial stress shock', 'Lower SPB shock']


def check_label(check):
    return CHECKS.get(check, check)


def variant_label(check, variant):
    """
    Value of a check with its unit.
    """
    if variant in VARIANTS:
        return VARIANTS[variant]
    if variant.startswith('latest vintage'):
        return 'October 2026 data'
    if check == 'Probability target':
        return f'{float(variant) * 100:.0f}%'
    if variant.startswith('x'):
        return f'×{variant[1:]}'
    if check in PP_CHECKS and not variant.endswith('pp.'):
        return f'{variant} pp.'
    return variant


def binding_constraint(r):
    """
    Constraint that sets the binding target of a run r: the DSA criteria, the adjustment floor, the debt
    sustainability safeguard, the deficit resilience safeguard or the minimum adjustment of the EDP.
    """
    dsa = r.get('dsa_target_commission', r['dsa_target'])  # DSA target with the floor, as in the binding search
    if r['binding_target'] <= dsa + 0.005:
        return 'Adjustment floor' if r.get('dsa_criterion_commission') == 'floor' else 'DSA criteria'
    flag = lambda key: bool(r.get(key)) and not pd.isna(r.get(key))
    if flag('debt_safeguard_binding') or r['binding_criterion'] == 'debt_safeguard':
        return 'Debt safeguard'
    if flag('edp_binding') and flag('deficit_resilience_binding'):
        return 'EDP and deficit resilience'
    if flag('edp_binding'):
        return 'EDP minimum adjustment'
    if flag('deficit_resilience_binding'):
        return 'Deficit resilience'
    return CRITERIA.get(r['binding_criterion'], r['binding_criterion'])


def baseline_table(oat, period=4):
    """
    Baseline targets of the model and of the Commission's prior guidance by country, ordered by debt in 2024, with the
    criterion that sets the DSA target and the constraint that sets the binding target. The model's DSA target leaves
    out the adjustment floor. Also returns the mean, median and largest absolute difference to the Commission's
    targets; for the DSA target, with the floor, as in the Commission's DSA target.
    """
    b = baseline(oat, period).sort_values('debt_T', ascending=False)
    table = pd.DataFrame({
        'debt 2024': b['debt_T'], 'SPB 2024': b['spb_T'],
        'DSA target, Commission': b['commission_dsa_target'], 'DSA target, model': b['dsa_target'],
        'binding target, Commission': b['commission_target'], 'binding target, model': b['binding_target'],
        'DSA criterion': b['dsa_criterion'].map(lambda c: CRITERIA.get(c, c)),
        'binding constraint': b.apply(binding_constraint, axis=1), 'guidance': b['guidance']})
    diff = pd.DataFrame({'DSA target': b['dsa_target_commission'] - b['commission_dsa_target'],
                         'binding target': b['binding_target'] - b['commission_target']}).abs()
    return table.rename(index=NAMES), diff.agg(['mean', 'median', 'max'])


def results_rows(oat, menu, period=4):
    """
    Rows of the results tables, in the order of the paper: (block, input, value, runs). Blocks 1 to 3 are the
    individual checks; block 4 is the start and horizon of the stochastic projection, the stochastic criterion as a
    grid of threshold, start and horizon, the stress tests, the ageing cost period, TFP growth and the lenient and
    strict scenarios; the data block is the Eurostat repayment profile.
    """
    o = oat[oat['adjustment_period'] == period]
    m = menu[menu['adjustment_period'] == period]
    rows = []
    for group in (FORECASTS, MODEL_PARAMETERS, ESTIMATION):
        for (check, variant), g in o[o['group'] == group].groupby(['check', 'variant'], sort=False):
            rows.append((group, check_label(check), variant_label(check, variant), g))
    for check in ('Stochastic start', 'Stochastic period'):
        for variant, g in o[o['check'] == check].groupby('variant', sort=False):
            rows.append((RISK_STANDARD, check_label(check), variant_label(check, variant), g))
    grid = m[m['check'] == 'Stochastic criterion']
    for start, h in [('after adjustment', 5), ('after adjustment', 10), ('first adjustment year', 5),
                     ('first adjustment year', 10)]:
        for p in ('60%', '70%', '80%'):
            if (p, start, h) in [('70%', 'first adjustment year', 5), ('70%', 'after adjustment', 10)]:
                continue  # the same runs as the start and horizon checks above
            g = grid[grid['variant'] == f'{p}, start {start}, {h}-year horizon']
            if len(g):
                s = 'after adj.' if start.startswith('after') else 'first adj. year'
                rows.append((RISK_STANDARD, 'Threshold, start, horizon', f'{p}, {s}, {h} years', g))
    for check in ('Adverse r-g shock', 'Lower SPB shock', 'Financial stress shock', 'Ageing cost period',
                  'TFP growth after T+10'):
        for variant, g in m[m['check'] == check].groupby('variant', sort=False):
            rows.append((RISK_STANDARD, check_label(check), variant_label(check, variant), g))
    for variant, g in o[o['check'] == 'Calibration scenario'].groupby('variant', sort=False):
        margins = variant.endswith('_margins')  # only the threshold and the stress tests
        if margins:  # no single own criterion: the scenario combines stochastic and deterministic criteria
            g = g.assign(own_delta_target=np.nan)
        rows.append((RISK_STANDARD, 'Threshold and stress tests' if margins else 'All settings',
                     variant.replace('_margins', ''), g))
    data = o[(o['group'] == DATA) & (o['id'] != 'data:latest')]
    for (check, variant), g in data.groupby(['check', 'variant'], sort=False):
        rows.append((DATA, check_label(check), variant_label(check, variant), g))
    return rows


def results_table(oat, menu, countries, all_countries, period=4, blocks=None):
    """
    Change of the SPB relative to the baseline for each check and setting (pp. of GDP), for the own criterion, the
    DSA target and the binding target: mean across countries, the largest change in absolute terms and its country,
    and for the two targets the mean across all_countries ('all'). Runs in which the debt sustainability safeguard
    cannot be met are left out of the binding target; ('binding', 'n') counts the countries left.
    """
    rows = []
    for group, check, value, g in results_rows(oat, menu, period):
        if blocks is not None and group not in blocks:
            continue
        row = {}
        for tier, (_, col) in TIERS.items():
            x = feasible(g[g['country'].isin(countries)], tier).set_index('country')[col].dropna()
            largest = x.abs().idxmax() if len(x) else None
            row[(tier, 'mean')] = x.mean()
            if tier != 'own':
                row[(tier, 'all')] = feasible(g[g['country'].isin(all_countries)], tier)[col].mean()
            row[(tier, 'largest')] = x[largest] if largest else np.nan
            row[(tier, 'country')] = largest if largest and abs(x[largest]) >= 0.005 else ''
        row[('binding', 'n')] = len(x)
        rows.append(pd.Series(row, name=(BLOCK_TITLES[group], check, value)))
    table = pd.DataFrame(rows)
    table.index.names = ['block', 'input', 'value']
    return table


def country_bands(joint, oat, order, quantiles=(0.25, 0.75)):
    """
    Joint draws by country: debt and SPB in 2024, the adjustment that the baseline binding target requires, the
    baseline targets and, for each level, the 25th percentile (lo), median (med) and 75th percentile (hi) of the change
    across the draws and the band between the 25th and 75th percentiles. Also the number of draws in which the
    safeguard cannot be met and the share of draws whose binding target requires a total adjustment above
    LARGE_ADJUSTMENT.
    """
    base = baseline(oat)
    draws = joint[joint['group'] == 'Global']
    rows = []
    for c in order:
        g = draws[draws['country'] == c]
        row = {'country': c, 'debt_T': base.loc[c, 'debt_T'], 'spb_T': base.loc[c, 'spb_T'],
               'required_adjustment': base.loc[c, 'binding_target'] - base.loc[c, 'spb_T'],
               'dsa_base': base.loc[c, 'dsa_target'], 'binding_base': base.loc[c, 'binding_target'],
               'infeasible': int(g['safeguard_infeasible'].sum()),
               'share_large': ((g['binding_target'] - g['spb_T']) > LARGE_ADJUSTMENT + 1e-9).mean()}
        for tier, (_, col) in TIERS.items():
            x = feasible(g, tier)[col].dropna()
            row[f'{tier}_lo'], row[f'{tier}_med'], row[f'{tier}_hi'] = x.quantile([quantiles[0], 0.5, quantiles[1]])
            row[f'{tier}_band'] = row[f'{tier}_hi'] - row[f'{tier}_lo']
        rows.append(row)
    return pd.DataFrame(rows).set_index('country')


def bands_table(bands, groups):
    """
    Joint draws by country, with the means across groups of countries ({label: countries}).
    """
    cols = ['debt_T', 'required_adjustment', 'own_band'] + [f'{t}_{s}' for t in ('dsa', 'binding')
                                                            for s in ('base', 'lo', 'med', 'hi', 'band')]
    means = pd.DataFrame({f'Mean, {k} ({len(v)})': bands.loc[v, cols].mean() for k, v in groups.items()}).T
    return pd.concat([bands[cols].rename(index=NAMES), means])


COMPARISON_ROWS = {  # block: [(label, run id, or 'joint:lo' / 'joint:hi' for the 25th / 75th percentile of the draws)]
    'Forecasts and model parameters': [
        ('Joint draws, 25th percentile (favourable)', 'joint:lo'),
        ('Joint draws, 75th percentile (adverse)', 'joint:hi'),
        ('Potential growth T+10, -0.75 pp.', 'macro:potential_growth:-0.75'),
        ('Real long-term rate T+10, +1 pp.', 'macro:rate_lt_T10:+1'),
        ('Fiscal multiplier 1.0', 'multiplier:fiscal_multiplier:1'),
        ('Impulse-response output gap rule, 3 years', 'multiplier:rule:impulse_response:3'),
        ('Ageing costs of the 2021 Ageing Report', 'macro:ageing_vintage:2021')],
    'Estimation choices in the stochastic analysis': [
        ('AR bootstrap', 'stochastic:estimation:ar_bootstrap'),
        ('AR bootstrap with intercept', 'stochastic:estimation:ar_bootstrap:intercept'),
        ('VAR bootstrap', 'stochastic:estimation:var_bootstrap'),
        ('No winsorisation', 'stochastic:winsorize:off')],
    'Calibration conventions': [
        ('Probability threshold 60%', 'stochastic:prob_target:0.6'),
        ('Probability threshold 80%', 'stochastic:prob_target:0.8'),
        ('Threshold 80%, start in first adjustment year', 'menu:stochastic:p0.8:adjustment:h5'),
        ('Adverse r-g shocks 0.25 pp.', 'menu:adverse_r_g_shock:0.25'),
        ('Adverse r-g shocks 0.75 pp.', 'menu:adverse_r_g_shock:0.75'),
        ('Lower-SPB shock 0.25 pp.', 'menu:lower_spb_shock:0.25'),
        ('Lower-SPB shock 0.75 pp.', 'menu:lower_spb_shock:0.75'),
        ('Ageing cost period 5 years', 'macro:ageing_cost_period:5'),
        ('Ageing cost period 0 years', 'macro:ageing_cost_period:0'),
        ('TFP growth held at T+10 value', 'risk:tfp_hold'),
        ('Lenient scenario, threshold and stress tests', 'calibration:lenient_margins'),
        ('Strict scenario, threshold and stress tests', 'calibration:strict_margins'),
        ('Lenient scenario, all settings', 'calibration:lenient'),
        ('Strict scenario, all settings', 'calibration:strict')],
}


def comparison_table(oat, menu, bands, groups, period=4):
    """
    Mean change of the DSA target and of the binding target for the favourable and adverse ends of the joint draws and
    for selected checks and settings, across each group of countries ({label: countries}). Runs in which the debt
    sustainability safeguard cannot be met are left out of the binding target.
    """
    runs = pd.concat([oat, menu], ignore_index=True).drop_duplicates(['id', 'country', 'adjustment_period'])
    runs = runs[runs['adjustment_period'] == period]
    table = {}
    for block, rows in COMPARISON_ROWS.items():
        for label, run_id in rows:
            for tier in ('dsa', 'binding'):
                for name, countries in groups.items():
                    if run_id.startswith('joint:'):
                        x = bands.loc[countries, f"{tier}_{run_id.split(':')[1]}"]
                    else:
                        r = runs[(runs['id'] == run_id) & runs['country'].isin(countries)]
                        x = feasible(r, tier)[TIERS[tier][1]]
                    table.setdefault((block, label), {})[(tier, name)] = x.mean()
    return pd.DataFrame(table).T.rename_axis(['block', 'setting'])


MENU_COUNTRY_COLUMNS = [  # (check, variant, column)
    ('Stochastic criterion', '80%, start after adjustment, 5-year horizon', 'Threshold 80%'),
    ('Stochastic criterion', '70%, start first adjustment year, 5-year horizon', 'Early start'),
    ('Stochastic criterion', '80%, start first adjustment year, 5-year horizon', 'Early start, 80%'),
    ('Adverse r-g shock', '0.75 pp.', 'r-g 0.75 pp.'),
    ('Lower SPB shock', '0.75 pp.', 'Lower SPB 0.75 pp.'),
    ('Ageing cost period', '5 years', 'Ageing 5 years'),
    ('Ageing cost period', '0 years', 'Ageing 0 years'),
    ('TFP growth after T+10', 'held at T+10 value', 'TFP held'),
    ('Calibration scenario', 'lenient_margins', 'Lenient, margins'),
    ('Calibration scenario', 'strict_margins', 'Strict, margins'),
    ('Calibration scenario', 'lenient', 'Lenient'),
    ('Calibration scenario', 'strict', 'Strict')]


def menu_country_table(menu, oat, order, period=4):
    """
    Change of the DSA target by country under selected calibration conventions.
    """
    d = pd.concat([menu, oat[oat['check'] == 'Calibration scenario']])
    d = d[d['adjustment_period'] == period]
    return pd.DataFrame({col: d[(d['check'] == c) & (d['variant'] == v)].set_index('country')['dsa_delta_target']
                         for c, v, col in MENU_COUNTRY_COLUMNS}).reindex(order)


def vintage_table(oat, menu, latest, countries):
    """
    Mean change of the SPB for each check across countries, at the three levels, with the 2024 data and with the
    October 2026 data under each method in latest ({method: runs}), four-year adjustment. Also returns, by method and
    level, the correlation and the mean absolute difference of the mean effects with those of the 2024 data.
    """
    oat, menu = oat[oat['country'].isin(countries)], menu[menu['country'].isin(countries)]
    o = oat[oat['id'] != 'data:latest']
    m = menu[menu['group'] == RISK_STANDARD]
    in_oat = {(c, variant_label(c, v)) for c, v in zip(o['check'], o['variant'])}
    m = m[[(c, variant_label(c, v)) not in in_oat for c, v in zip(m['check'], m['variant'])]]
    p = '4-year adjustment'
    tables = {'2024': effects_table(pd.concat([o, m], ignore_index=True), (4,))}
    for meth, d in latest.items():
        tables[f'2026 ({meth} method)'] = effects_table(d[d['country'].isin(countries)], (4,))
    t24 = tables['2024']
    t24 = t24[t24.index.get_level_values('group').isin(list(BLOCK_TITLES))
              & (t24.index.get_level_values('check') != 'Stochastic criterion')]
    # block order, then the order in which checks first appear
    blocks, checks = list(BLOCK_TITLES), list(dict.fromkeys(t24.index.get_level_values('check')))
    t24 = t24.iloc[sorted(range(len(t24)), key=lambda i: (blocks.index(t24.index[i][0]), checks.index(t24.index[i][1])))]
    table = pd.DataFrame({(lab, data): t[(p, lab, 'mean')].reindex(t24.index)
                          for lab, _ in TIERS.values() for data, t in tables.items()})
    stats = {data: {lab: (table[(lab, '2024')].corr(table[(lab, data)]),
                          (table[(lab, '2024')] - table[(lab, data)]).abs().mean()) for lab, _ in TIERS.values()}
             for data in tables if data != '2024'}
    table.index = pd.MultiIndex.from_tuples([(BLOCK_TITLES[g], check_label(c), variant_label(c, v))
                                             for g, c, v in table.index], names=['block', 'check', 'value'])
    names = {lab: tier for tier, (lab, _) in TIERS.items()}  # 'own', 'dsa', 'binding'
    table.columns = pd.MultiIndex.from_tuples([(names[lab], data) for lab, data in table.columns])
    return table, {data: {names[lab]: v for lab, v in s.items()} for data, s in stats.items()}


def latest_targets(oat, latest, countries=EU27, cache_dir=OUTPUT_DIR):
    """
    Targets with the October 2026 data and the 2026 method by country (four-year adjustment unless stated): the DSA
    target and the criterion that sets it, the binding target and the constraint that sets it, the annual adjustment,
    the change of the binding target relative to the 2024 baseline (data and method), the binding target with a
    seven-year adjustment and the binding target with the October 2026 data and the 2024 method. Binding targets of
    runs in which the debt sustainability safeguard cannot be met are left empty.
    """
    binding = lambda d: d['binding_target'].where(~d['safeguard_infeasible'].astype(bool))
    new = latest[LATEST_METHOD][latest[LATEST_METHOD]['id'] == 'baseline'].set_index('country')
    old_method = oat[(oat['id'] == 'data:latest') & (oat['adjustment_period'] == 4)].set_index('country')
    t = pd.DataFrame({
        'debt 2026': new['debt_T'], 'SPB 2026': new['spb_T'], 'DSA target': new['dsa_target'],
        'DSA criterion': new['dsa_criterion'].map(lambda c: CRITERIA.get(c, c)), 'binding target': binding(new),
        'binding constraint': new.apply(binding_constraint, axis=1), 'annual adjustment': new['annual_adjustment'],
        'change vs 2024 baseline': binding(new) - binding(baseline(oat)),
        'binding target, 7 years': binding(latest_seven_year(countries, cache_dir).set_index('country')),
        'binding target, 2024 method': binding(old_method)})
    return t.sort_values('debt 2026', ascending=False).rename(index=NAMES)


SHORT_LABELS = [  # shorter labels for the largest effects by country
    ('Ageing costs, previous vintage: 2021 Ageing Report', '2021 Ageing Report'),
    ('Ageing costs, 2024 AR scenario: ', 'Ageing, '),
    ('Stochastic criterion: 80%, start first adjustment year, ', 'Threshold 80%, early start, '),
    ('Potential growth T+10:', 'Potential growth'), ('Real long-term rate, T+10:', 'Long-term rate'),
    ('Multiplier persistence: 5 years', 'Persistence 5 years'), ('Fiscal multiplier:', 'Multiplier'),
    ('Ageing cost period:', 'Ageing period'), ('Winsorised shocks: off', 'No winsorisation'),
    ('Impulse-response output gap rule:', 'Impulse-response rule,'), ('Shock model: ', '')]


def country_cases(oat, menu, bands, countries, period=4, n_largest=3):
    """
    By country: debt in 2024, the adjustment that the baseline binding target requires, the criterion that sets the
    DSA target and the constraint that sets the binding target, the 25th and 75th percentiles of the change of the DSA
    target in the joint draws, and the checks or settings with the largest effect on the DSA target (without the
    lenient and strict scenarios and the data update).
    """
    runs = pd.concat([oat, menu], ignore_index=True).drop_duplicates(['id', 'country', 'adjustment_period'])
    runs = runs[(runs['adjustment_period'] == period) & ~runs['id'].isin(['baseline', 'data:latest'])
                & ~runs['group'].isin(['Baseline', 'Noise']) & (runs['check'] != 'Calibration scenario')]
    base = baseline(oat, period)
    rows = {}
    for c in countries:
        d = runs[runs['country'] == c].dropna(subset=['dsa_delta_target'])
        d = d.reindex(d['dsa_delta_target'].abs().sort_values(ascending=False).index)
        largest = {}
        for _, x in d.iterrows():
            label = f"{check_label(x['check'])}: {variant_label(x['check'], x['variant'])}"
            for long, short in SHORT_LABELS:
                label = label.replace(long, short)
            largest.setdefault(label, x['dsa_delta_target'])
            if len(largest) == n_largest:
                break
        r = base.loc[c]
        rows[NAMES[c]] = {'debt 2024': r['debt_T'], 'required adjustment': r['binding_target'] - r['spb_T'],
                          'DSA criterion': CRITERIA.get(r['dsa_criterion'], r['dsa_criterion']),
                          'binding constraint': binding_constraint(r),
                          'joint draws P25': bands.loc[c, 'dsa_lo'], 'joint draws P75': bands.loc[c, 'dsa_hi'],
                          'largest effects': '; '.join(f'{k} ({v:+.2f})' for k, v in largest.items())}
    return pd.DataFrame(rows).T


CALIBRATION_LABELS = {'rate_st_T10': 'Real short-term rate, T+10', 'rate_lt_T10': 'Real long-term rate, T+10',
                      'inflation_T10': 'Inflation, T+10', 'potential_growth': 'Potential growth, T+10',
                      'ageing_shift': 'Change in ageing costs over 14 years'}


def calibration_ranges(ranges):
    """
    Distribution of past forecast errors: source, sample, number of errors, mean error (removed), and the 10th, 25th,
    75th and 90th percentiles of the errors drawn in the joint draws. ranges: calibration_ranges.csv (25th and 75th
    percentiles).
    """
    errors = forecast_error_samples()
    sample = {'rate_st_T10': errors['rates']['rate_st'], 'rate_lt_T10': errors['rates']['rate_lt'],
              'inflation_T10': errors['weo']['inflation'], 'potential_growth': errors['weo']['growth'],
              'ageing_shift': errors['ageing']['ageing']}
    return pd.DataFrame({CALIBRATION_LABELS[k]: {
        'source': r['source'], 'sample': r['sample'], 'n': int(r['n']), 'mean error': r['bias'],
        'P10': sample[k].quantile(0.1), 'P25': r['range_low'], 'P75': r['range_high'], 'P90': sample[k].quantile(0.9)}
        for k, r in ranges.iterrows()}).T


# ========================================================================================= #
#                                    5. FIGURES                                             #
# ========================================================================================= #

INK, MUTED, LIGHT, PALE = 'black', '0.45', '0.7', '0.88'
BLUE_TINT, BLUE_PALE, RED_TINT = '#aec7e8', '#dbe8f5', '#ff9896'  # light shades of C0 and C3
STYLE = {'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'axes.linewidth': 0.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': False,
         'legend.frameon': False}
WIDTH = 6.3  # figure width in inches


def rules_model(country, period=4, multiplier=None, edp=True, debt_safeguard=True, deficit_resilience=True):
    """
    Baseline model of a country with the Commission rules and only the given constraints of the binding target, with
    the minimum steps of the EDP and the deficit resilience safeguard but without front-loading.
    """
    model_kwargs = dict(METHODS[METHOD], **({} if multiplier is None else {'fiscal_multiplier': float(multiplier)}))
    model = build_model(baseline_spec(), country, period, INPUT_FILE, model_kwargs=model_kwargs, attributes=ATTRIBUTES)
    model.rules = model._set_rules(RULES, {}, edp, debt_safeguard, deficit_resilience, None)
    model.frontloading, model.edp_active, model.edp_status = False, True, 'input'
    model.edp_default_steps = np.full(period, np.nan)
    return model


def constraint_requirements(oat, period=4, cache_dir=OUTPUT_DIR):
    """
    SPB at the end of the adjustment period required by each constraint of the binding target on its own:
    - debt sustainability safeguard: the smallest constant annual adjustment that meets it, with the minimum steps of
      the EDP and the deficit resilience safeguard (countries with a reference trajectory and debt above 60%);
    - deficit resilience safeguard, EDP: the path of the DSA target with the minimum steps of the constraint, where
      the constraint applies in 2024 or its minimum steps raise the SPB later;
    - adjustment floor: the SPB in 2024 for countries with a reference trajectory, minus 1 pp. a year otherwise.
    Stored in constraint_requirements.csv, since it takes a few minutes; delete the file to recompute it.
    """
    path = cache_dir / 'constraint_requirements.csv'
    if path.exists():
        return pd.read_csv(path, index_col='country')
    rows = {}
    for c, r in baseline(oat, period).iterrows():
        ref = r['guidance'] == 'reference trajectory'
        spb0 = r['spb_T']
        out = {'floor': spb0 if ref else spb0 - period}
        for key in ('deficit_resilience', 'edp'):
            m = rules_model(c, period, edp=key == 'edp', debt_safeguard=False, deficit_resilience=key != 'edp')
            m._rules_path((r['dsa_target_commission'] - spb0) / period)
            v = float(m.spb_bca[m.adjustment_end])
            applies = m.sb[m.adjustment_start - 1] <= -1.55 if key == 'deficit_resilience' else m._in_edp_T()
            out[key] = v if applies or v > r['dsa_target_commission'] + 0.005 else np.nan
        out['debt_safeguard'] = np.nan
        if ref and r['debt_T'] > 60:
            m = rules_model(c, period)
            x = m._debt_safeguard_adjustment(0.0, getattr(m, 'adjustment_bound', 5))
            if m._debt_safeguard_met(m._rules_path(x)):
                out['debt_safeguard'] = spb0 + period * x
        rows[c] = out
    req = pd.DataFrame(rows).T.rename_axis('country')
    req.to_csv(path)
    return req


def all_criteria_runs(oat, countries, cache_dir=OUTPUT_DIR, run=True):
    """
    Baseline runs with all DSA criteria computed, for countries with technical information, whose baseline computes
    only some of them (figure_ladder only).
    """
    base = baseline(oat)
    missing = [c for c, d in base['spb_target_dict'].items() if 'debt_declines_main_adjustment' not in d]
    if not missing:
        return {}
    spec = {**baseline_spec(), 'id': 'figure:all_criteria', 'group': 'Figure', 'check': 'All criteria',
            'variant': 'all', 'binding_kwargs': {'dsa_criteria': 'default'}}
    runs = run_specs([spec], cache_file=cache_dir / 'figure_runs.pkl', **run_options(missing, run, period=4))
    return runs.set_index('country')['spb_target_dict'].to_dict()


LADDER = [  # criterion, label, marker, colour, column (0 debt criteria, 1 deficit and stochastic, 2 constraints)
    ('main_adjustment', 'Debt declines: baseline scenario', 'o', '#9ecae1', 0),
    ('financial_stress', 'Debt declines: financial stress test', 'o', '#4292c6', 0),
    ('lower_spb', 'Debt declines: lower-SPB test', 'o', '#2171b5', 0),
    ('adverse_r_g', 'Debt declines: adverse r-g test', 'o', '#08306b', 0),
    ('deficit_reduction', 'Deficit below 3%', 's', 'C1', 1),
    ('stochastic', 'Stochastic: debt declines', 's', 'C4', 1),
    ('debt_safeguard', 'Debt sustainability safeguard', '^', 'C2', 2),
    ('deficit_resilience', 'Deficit resilience safeguard', '^', 'C3', 2),
    ('edp', 'EDP minimum adjustment', '^', 'C5', 2),
    ('floor', 'Adjustment floor', '_', INK, 2),
]
BINDING_KEY = {'Adjustment floor': 'floor', 'Debt safeguard': 'debt_safeguard',
               'Deficit resilience': 'deficit_resilience', 'EDP minimum adjustment': 'edp'}


def criterion_values(targets, requirements, applicable):
    """
    SPB required by each criterion of LADDER for one country: the DSA criteria from targets (spb_target_dict), in the
    variant that applies to the country (debt declines, or debt below 60%), and the constraints from requirements.
    """
    below_60 = any(k.startswith('debt_below_60_') for k in applicable)
    values = dict(requirements)
    for key, *_ in LADDER[:6]:
        name = key if key in ('deficit_reduction', 'stochastic') else \
            f"{'debt_below_60' if below_60 else 'debt_declines'}_{key}"
        values[key] = targets.get(name, np.nan)
    return values


def figure_ladder(oat, order, cache_dir=OUTPUT_DIR, run=True):
    """
    SPB at the end of the adjustment period required by each criterion in the baseline, four-year adjustment, one
    column per country ordered by debt in 2024 (in brackets): debt criteria (circles), deficit and stochastic criteria
    (squares) and constraints of the binding target (triangles; bar for the adjustment floor). An open circle marks the
    criterion that sets the binding target.
    """
    base = baseline(oat).reindex(order)
    extra = all_criteria_runs(oat, order, cache_dir, run)
    req = constraint_requirements(oat, cache_dir=cache_dir).reindex(order)
    style = {key: (label, marker, color, col) for key, label, marker, color, col in LADDER}
    offset = [-0.27, 0.0, 0.27]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        shown = []
        for x, c in enumerate(order):
            r = base.loc[c]
            vals = criterion_values(extra.get(c, r['spb_target_dict']), req.loc[c],
                                    applicable_dsa_criteria(r['spb_target_dict']))
            # criterion that sets the binding target: a constraint, or the DSA criterion (with the floor)
            constraint = binding_constraint(r)
            if constraint in BINDING_KEY:
                key_b = BINDING_KEY[constraint]
            else:
                key_b = r['dsa_criterion_commission'].replace('debt_declines_', '').replace('debt_below_60_', '')
            if r['guidance'] != 'reference trajectory' and key_b != 'floor':
                vals['floor'] = np.nan  # floor for technical information shown only where it binds
            for key, (label, marker, color, col) in style.items():
                if pd.notna(vals.get(key)):
                    shown.append(vals[key])
                    ax.scatter(x + offset[col], vals[key], marker=marker, s=70 if marker == '_' else 30, color=color,
                               lw=1.5 if marker == '_' else 0, zorder=3, label=label)
            ax.scatter(x, r['spb_T'], marker='x', s=14, color=LIGHT, lw=0.9, zorder=2, label='SPB in 2024')
            _, marker, color, col = style[key_b]
            ax.scatter(x + offset[col], vals[key_b], s=120, facecolor='white', edgecolor=INK, lw=1.0, zorder=5,
                       label='Criterion that sets the binding target')
            ax.scatter(x + offset[col], vals[key_b], marker=marker, s=70 if marker == '_' else 30, color=color,
                       lw=1.5 if marker == '_' else 0, zorder=6)
        ax.axhline(0, color=INK, lw=0.5, zorder=0)
        for x in range(1, len(order)):
            ax.axvline(x - 0.5, color='0.8', lw=0.6, zorder=0)
        ax.set_xticks(range(len(order)), [f"{c}\n({base.loc[c, 'debt_T']:.0f})" for c in order])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(np.floor(min(shown)) - 0.3, np.ceil(max(shown)) + 0.3)
        ax.set_ylabel('SPB in 2028 (% of GDP)')
        handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
        fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.15, 1, 1))
    return fig


SAFEGUARD_COUNTRIES = ['FIN', 'ESP', 'AUT', 'DEU']
MULTIPLIERS = np.round(np.r_[np.arange(0.5, 1.399, 0.05), np.arange(1.4, 1.501, 0.01)], 2)
ADJUSTMENT_GRID = np.round(np.r_[np.arange(0, 7.9999, 0.01), np.arange(8, 20.001, 0.25)], 2)


def safeguard_requirement(period=4, cache_dir=OUTPUT_DIR):
    """
    Smallest constant annual adjustment (on ADJUSTMENT_GRID) that meets the debt sustainability safeguard, by country
    and fiscal multiplier, and the average annual adjustment of the resulting path (path_adjustment), which includes the
    minimum steps of the EDP and the deficit resilience safeguard; NaN if no adjustment on the grid meets it. The search
    runs upwards because the safeguard is not monotonic in the adjustment. Stored in safeguard_requirement(_7y).csv,
    since it takes long; delete the file to recompute it.
    """
    path = cache_dir / ('safeguard_requirement.csv' if period == 4 else f'safeguard_requirement_{period}y.csv')
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for c in SAFEGUARD_COUNTRIES:
        for m in MULTIPLIERS:
            model = rules_model(c, period, multiplier=m)
            row = {'country': c, 'multiplier': float(m), 'required_adjustment': np.nan, 'path_adjustment': np.nan}
            for x in ADJUSTMENT_GRID:
                if model._debt_safeguard_met(model._rules_path(x)):
                    s, e = model.adjustment_start, model.adjustment_end
                    row.update(required_adjustment=float(x),
                               path_adjustment=float(model.spb_bca[e] - model.spb_bca[s - 1]) / period)
                    break
            rows.append(row)
    req = pd.DataFrame(rows)
    req.to_csv(path, index=False)
    return req


def debt_paths_multiplier(country, adjustment, multipliers=(0.5, 0.75, 1.0, 1.25, 1.5), period=4, last_year=2038):
    """
    Debt paths of a country for the same constant annual adjustment and several fiscal multipliers, whether each path
    meets the debt sustainability safeguard, and the first and last adjustment year.
    """
    paths, met = {}, {}
    for m in multipliers:
        model = rules_model(country, period, multiplier=m)
        met[m] = bool(model._debt_safeguard_met(model._rules_path(adjustment)))
        first = model.adjustment_start_year
        years = model.start_year + np.arange(len(model.d))
        paths[m] = pd.Series(model.d, index=years).loc[first - 1:last_year]
    return pd.DataFrame(paths), met, (first, first + period - 1)


def figure_safeguard(oat, cache_dir=OUTPUT_DIR, country='FIN'):
    """
    (a) Average annual adjustment that the debt sustainability safeguard requires against the fiscal multiplier, four
    years (solid) and seven years (dotted). (b) Debt of one country under its baseline annual adjustment for several
    multipliers, with the adjustment period shaded. Returns the figure, the requirements (four and seven years) and
    whether each path of panel (b) meets the safeguard.
    """
    req, req7 = safeguard_requirement(4, cache_dir), safeguard_requirement(7, cache_dir)
    adjustment = round(float(baseline(oat).loc[country, 'annual_adjustment']), 2)
    paths, met, (first, last) = debt_paths_multiplier(country, adjustment)
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH * 1.4, 3.6))
        for c, color in zip(SAFEGUARD_COUNTRIES, ['C0', 'C3', 'C2', 'C1']):
            for d, ls in [(req, '-'), (req7, ':')]:
                d = d[d['country'] == c].sort_values('multiplier')
                ax.plot(d['multiplier'], d['path_adjustment'], color=color, lw=1.3, ls=ls,
                        label=NAMES[c] if ls == '-' else None)
        ax.plot([], [], color=MUTED, ls='-', label='Four years')
        ax.plot([], [], color=MUTED, ls=':', label='Seven years')
        ax.set(xlim=(0.5, 1.5), ylim=(0, None), xlabel='Fiscal multiplier', ylabel='Annual adjustment (pp. of GDP)')
        ax.set_title('(a) Annual adjustment required by the safeguard', loc='left')
        ax.legend(loc='upper left', ncol=2)
        bx.axvspan(first - 0.5, last + 0.5, color=PALE, lw=0, label='Adjustment')
        for m, color, ls in zip(paths, plt.cm.Blues(np.linspace(0.4, 1.0, len(paths.columns))),
                                ['-', '-', '--', '-.', ':']):
            bx.plot(paths.index, paths[m], color=color, lw=1.3, ls=ls, label=f'm = {m:g}')
        bx.set(xlim=(paths.index.min(), paths.index.max()), ylabel='Debt (% of GDP)')
        bx.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        bx.set_title(f'(b) Debt of {NAMES[country]}, annual adjustment {adjustment} pp.', loc='left')
        bx.legend(loc='upper right')
        fig.tight_layout()
    return fig, req, req7, met


def figure_ranges(bands, joint):
    """
    DSA target (blue) and binding target (red) by country across the joint draws: boxes from the 25th to the 75th
    percentile with the median, whiskers from the 10th to the 90th percentile; dots are the baseline targets and
    crosses the SPB in 2024. Draws in which the safeguard cannot be met are left out of the binding target.
    """
    order = list(bands.index)
    y = np.arange(len(order))
    draws = joint[joint['group'] == 'Global']
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 7.6))
        for tier, col, face, edge, off in [('dsa', 'dsa_target', BLUE_TINT, 'C0', -0.19),
                                           ('binding', 'binding_target', RED_TINT, 'C3', 0.19)]:
            d = feasible(draws, tier)
            stats = []
            for c in order:
                q = np.percentile(d.loc[d['country'] == c, col].dropna(), [10, 25, 50, 75, 90])
                stats.append({'whislo': q[0], 'q1': q[1], 'med': q[2], 'q3': q[3], 'whishi': q[4], 'fliers': []})
            ax.bxp(stats, positions=y + off, widths=0.3, vert=False, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=face, edgecolor=edge, lw=0.7), medianprops=dict(color=edge, lw=1.2),
                   whiskerprops=dict(color=edge, lw=0.7), capprops=dict(color=edge, lw=0.7))
            ax.scatter(bands[f'{tier}_base'], y + off, marker='o', s=7, color=INK, zorder=4)
        ax.scatter(bands['spb_T'], y, marker='x', s=14, color=LIGHT, lw=0.9, zorder=3)
        ax.axvline(0, color=INK, lw=0.5, zorder=0)
        for yy in range(1, len(order)):
            ax.axhline(yy - 0.5, color='0.8', lw=0.6, zorder=0)
        ax.set_yticks(y, [NAMES[c] for c in order])
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel('SPB in 2028 (% of GDP)')
        handles = [Patch(facecolor=BLUE_TINT, edgecolor='C0', label='DSA target'),
                   Patch(facecolor=RED_TINT, edgecolor='C3', label='Binding target'),
                   Line2D([], [], marker='o', ls='', ms=2.5, color=INK, label='Baseline'),
                   Line2D([], [], marker='x', ls='', ms=4, color=LIGHT, label='SPB in 2024')]
        fig.legend(handles=handles, loc='lower center', ncol=4)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


COMPARISON_FIGURE_GROUPS = [  # (group of inputs, [settings of COMPARISON_ROWS]); the joint draws come first as a range
    ('Forecasts', ['Potential growth T+10, -0.75 pp.', 'Real long-term rate T+10, +1 pp.',
                   'Ageing costs of the 2021 Ageing Report']),
    ('Model parameters', ['Fiscal multiplier 1.0', 'Impulse-response output gap rule, 3 years']),
    ('Estimation choices', [label for label, _ in COMPARISON_ROWS['Estimation choices in the stochastic analysis']]),
    ('Calibration conventions', [label for label, _ in COMPARISON_ROWS['Calibration conventions']]),
]


def figure_comparison(comparison, group='reference'):
    """
    Mean change of the DSA target (left) and of the binding target (right) across the countries of group (a column of
    comparison_table) for the checks and settings of COMPARISON_ROWS, in the four groups of inputs. The grey bar in
    the first row and the grey shading behind all rows span the 25th to the 75th percentile of the joint draws.
    """
    v = comparison.droplevel('block')
    lo, hi = v.loc['Joint draws, 25th percentile (favourable)'], v.loc['Joint draws, 75th percentile (adverse)']
    rows, heads, y = [(1, '25th to 75th percentile', None)], [(0, 'Forecasts and model parameters drawn jointly')], 1
    for title, labels in COMPARISON_FIGURE_GROUPS:
        y += 1.4
        heads.append((y, title))
        for label in labels:
            y += 1
            rows.append((y, label, v.loc[label]))
    fmt = lambda x: f'{x:+.2f}'.replace('+0.00', '0.00').replace('-0.00', '0.00').replace('-', '−')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 6.6), sharey=True)
        for ax, tier, title, face, edge in [(axes[0], 'dsa', 'DSA target', BLUE_TINT, 'C0'),
                                            (axes[1], 'binding', 'Binding target', RED_TINT, 'C3')]:
            a0, b0 = lo[(tier, group)], hi[(tier, group)]
            ax.axvspan(a0, b0, color=PALE, lw=0, zorder=0)
            ax.axvline(0, color=INK, lw=0.5, zorder=1)
            for yy, label, x in rows:
                if x is None:  # joint draws: bar from the 25th to the 75th percentile
                    ax.barh(yy, b0 - a0, left=a0, height=0.62, color=LIGHT, edgecolor=MUTED, lw=0.6, zorder=2)
                    ax.text(a0 - 0.03, yy, fmt(a0), va='center', ha='right', fontsize=6.8)
                    ax.text(b0 + 0.03, yy, fmt(b0), va='center', ha='left', fontsize=6.8)
                else:
                    b = x[(tier, group)]
                    ax.barh(yy, b, height=0.62, color=face, edgecolor=edge, lw=0.6, zorder=2)
                    ax.text(b + (0.03 if b >= 0 else -0.03), yy, fmt(b), va='center', ha='left' if b >= 0 else 'right',
                            fontsize=6.8)
            ax.set_title(title)
            ax.set_xlim(-0.95, 0.95)
            ax.tick_params(axis='y', length=0)
        fig.supxlabel('Change in the SPB relative to the baseline (pp. of GDP)', fontsize=STYLE['font.size'])
        ticks = [(yy, label) for yy, label, _ in rows] + heads
        axes[0].set_yticks([yy for yy, _ in ticks], [label for _, label in ticks])
        for tick, (yy, _) in zip(axes[0].get_yticklabels(), ticks):
            if (yy, tick.get_text()) in heads:
                tick.set_fontstyle('italic')
        axes[0].set_ylim(y + 0.7, -0.7)
        fig.tight_layout(w_pad=1.0)
    return fig


def figure_debt(oat, fixed, order):
    """
    Debt by country across the joint draws when the SPB follows the path of the baseline binding target in every draw:
    10th to 90th and 25th to 75th percentiles in each year, and the baseline path.
    """
    base = baseline(oat)
    draws = fixed[fixed['id'] != 'baseline']
    nrows = int(np.ceil(len(order) / 5))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, 5, figsize=(WIDTH * 1.4, 1.7 * nrows + 0.4), squeeze=False)
        for ax, c in zip(axes.ravel(), order):
            d = draws[draws['country'] == c]
            paths = pd.DataFrame({i: pd.Series(p) for i, p in zip(d['id'], d['debt_path'])}).sort_index()
            q = paths.quantile([0.1, 0.25, 0.75, 0.9], axis=1).T
            ax.fill_between(q.index, q[0.1], q[0.9], color=BLUE_PALE, lw=0)
            ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE_TINT, lw=0)
            b = pd.Series(base.loc[c, 'debt_path']).sort_index()
            ax.plot(b.index, b.values, color=INK, lw=1.0)
            ax.set_title(NAMES[c])
            ax.set_xticks([2025, 2030, 2035])
        for ax in axes.ravel()[len(order):]:
            ax.set_visible(False)
        handles = [Patch(color=BLUE_PALE, label='10th to 90th percentile of joint draws'),
                   Patch(color=BLUE_TINT, label='25th to 75th percentile'),
                   Line2D([], [], color=INK, lw=1.2, label='Baseline')]
        fig.legend(handles=handles, loc='lower center', ncol=3)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
