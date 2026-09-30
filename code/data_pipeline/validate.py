# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Validation               #
# ========================================================================================= #
#
# Compares model results against the European Commission prior guidance calculation sheets:
#   1. Debt path when the model is given the Commission's SPB adjustment path (tests the debt
#      dynamics and input mapping independently of the optimisation).
#   2. SPB at the end of the adjustment period required by the DSA-based criteria and after
#      safeguards, using find_spb_binding(rules='commission') (tests the optimisation) or the
#      default rules.
#   3. compare_rules: contribution of each rule to the difference between the Commission rules
#      and the default rules, switching one rule at a time from the Commission to the default version.
#
# Run with an input workbook built in 'commission' mode to replicate the Commission, or with an
# 'api' workbook to see how updated data change the results.
#
# Author: Lennard Welslau
# ========================================================================================= #

import warnings
import numpy as np
import pandas as pd

from .schema import EU27
from .commission import load_prior_guidance

DSA_CRITERIA = ['main_adjustment', 'adverse_r_g', 'lower_spb', 'financial_stress', 'stochastic']


def compare_with_commission(input_file, countries=EU27, prior_guidance=None, stochastic=True,
                            adjustment_periods=(4, 7), seed=0, verbose=True, out_file=None, rules='commission'):
    """
    Run the model on an input workbook and compare with the Commission prior guidance sheets.

    Parameters:
        input_file (str): Input workbook, normally built with build_inputs(mode='commission').
        prior_guidance (str): '2024' or 'latest'. None: 'latest' if the file name contains 'latest', else '2024'.
        rules (str): Rules of find_spb_binding: 'commission' (prior guidance implementation, default) or 'default'.
        out_file (str/Path): Optional Excel file for the results (sheet spb_targets and one debt_<ISO3> sheet per country),
            e.g. REPO_ROOT / 'output' / 'validation' / 'commission_2024_comparison.xlsx'.

    Returns:
        summary (DataFrame): one row per country and adjustment period with Commission and model SPB targets.
        debt_paths (dict): {country: DataFrame of Commission vs model debt under the Commission SPB path}.
    """
    from classes import StochasticDsaModel as DSA  # local import avoids a circular import

    if prior_guidance is None:
        prior_guidance = 'latest' if 'latest' in str(input_file) else '2024'
    sheets = load_prior_guidance(countries, vintage=prior_guidance)
    rows, debt_paths = [], {}
    for country in countries:
        sheet = sheets[country]
        res = sheet.results()
        T = res['T']
        horizon = sheet.adjustment_end - T

        # 1. Debt path under the Commission adjustment scenario SPB path
        try:
            model = DSA(country=country, start_year=T, adjustment_start_year=T + 1, adjustment_period=horizon,
                        input_file=input_file)
            spb_path = res['spb_adjustment'].loc[T:sheet.adjustment_end].to_numpy()
            model.project(spb_steps=np.diff(spb_path))
            d_model = model.df('d')['d'].droplevel('t')
            d_ec = res['debt_adjustment']
            df = pd.DataFrame({'commission': d_ec, 'model': d_model}).loc[T:max(d_ec.index)]
            df['diff'] = df['model'] - df['commission']
            debt_paths[country] = df
            max_diff_10y = df.loc[:T + 10, 'diff'].abs().max()
        except Exception as e:
            warnings.warn(f'{country}: debt path replication failed ({e})')
            max_diff_10y = np.nan

        # 2. Required SPB at the end of the adjustment period
        for n in adjustment_periods:
            row = {'country': country, 'T': T, 'forecast': res['forecast'], 'guidance': res['guidance'],
                   'adjustment_period': n, 'spb_T': res['spb_T'],
                   'commission_spb_end': res[f'spb_end_{n}y'],
                   'commission_dsa_spb_end': res['spb_T'] + n * res[f'annual_adjustment_dsa_{n}y'],
                   'max_abs_debt_diff_T10': max_diff_10y if n == 4 else np.nan}
            row['commission_annual_adjustment_dsa'] = res[f'annual_adjustment_dsa_{n}y']
            try:
                model = DSA(country=country, start_year=T, adjustment_start_year=T + 1, adjustment_period=n,
                            input_file=input_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    np.random.seed(seed)
                    model.find_spb_binding(rules=rules, stochastic=stochastic, print_results=False)
                ref = model.binding_parameter_dict
                if 'annual_adjustment_dsa' in ref:
                    a_dsa = ref['annual_adjustment_dsa']
                else:  # default rules: DSA-based target from the results summary
                    dsa_target = model.binding_tables['Summary'].iloc[:, 0]['DSA-based SPB target (% of GDP)']
                    a_dsa = (dsa_target - model.spb_bca[model.adjustment_start - 1]) / n
                row['model_annual_adjustment_dsa'] = a_dsa
                row['model_annual_adjustment_dsa_rounded'] = round(a_dsa, 2)  # Commission rules: already on the 0.01 grid
                row['model_dsa_spb_end'] = res['spb_T'] + n * a_dsa
                row['model_spb_end'] = model.spb_target_dict['binding']
                row['model_binding_criterion'] = ref['criterion']
            except Exception as e:
                warnings.warn(f'{country} ({n}y): optimisation failed ({e})')
            rows.append(row)
        if verbose:
            r4 = rows[-len(adjustment_periods)]
            print(f"{country}: debt diff (max to T+10) {max_diff_10y:.2f} | 4y SPB end: "
                  f"Commission {r4['commission_spb_end']:.2f}, model {r4.get('model_spb_end', np.nan):.2f}")

    summary = pd.DataFrame(rows)
    summary['diff_spb_end'] = summary['model_spb_end'] - summary['commission_spb_end']
    summary['diff_dsa_spb_end'] = summary['model_dsa_spb_end'] - summary['commission_dsa_spb_end']
    summary['diff_annual_adjustment_dsa'] = summary['model_annual_adjustment_dsa_rounded'] - summary['commission_annual_adjustment_dsa']

    if out_file is not None:
        with pd.ExcelWriter(out_file) as writer:
            summary.to_excel(writer, sheet_name='spb_targets', index=False)
            for country, df in debt_paths.items():
                df.to_excel(writer, sheet_name=f'debt_{country}')
    return summary, debt_paths


# Rule switches from the Commission rules to the default rules (see StochasticDsaModel.find_spb_binding). Rules that
# interact are switched together: the adjustment path rules (front-loading of minimum steps, EDP, deficit resilience
# and debt safeguard) and the DSA criteria (deterministic and stochastic criteria, technical information and floors)
RULE_SWITCHES = {
    'Commission rules': {'rules': 'commission'},
    'Adjustment path rules': {'rules': 'commission', 'frontloading': True, 'edp': 'default', 'deficit_resilience': 'default',
                              'debt_safeguard': 'default'},
    'DSA criteria': {'rules': 'commission', 'dsa_criteria': 'default', 'stochastic_criteria': ['debt_declines', 'debt_below_60']},
    'Default rules': {'rules': 'default'},
}


def _rules_task(args):
    """
    Binding SPB at the end of the adjustment period for one country, adjustment period and rule configuration.
    """
    from classes import StochasticDsaModel as DSA  # local import avoids a circular import
    country, n, label, options, input_file, stochastic, seed = args
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = DSA(country=country, adjustment_period=n, input_file=input_file)
        np.random.seed(seed)
        model.find_spb_binding(stochastic=stochastic, print_results=False, **options)
    return {'country': country, 'adjustment_period': n, 'rules': label, 'spb_end': model.spb_target_dict['binding'],
            'binding_criterion': model.binding_criterion}


def compare_rules(input_file, countries=EU27, switches=RULE_SWITCHES, stochastic=True, adjustment_periods=(4, 7),
                  seed=0, parallel=True, max_workers=None, out_file=None):
    """
    SPB at the end of the adjustment period under the Commission rules, with one rule at a time switched to the
    default version, and under the default rules. Differences to the Commission rules show which rules drive the
    deviations between the two implementations (effects need not add up, as rules interact).

    Returns a DataFrame (country, adjustment period) x rule configuration, optionally saved to out_file.
    """
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(c, n, label, options, input_file, stochastic, seed)
             for c in countries for n in adjustment_periods for label, options in switches.items()]
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(_rules_task, tasks))
    else:
        rows = [_rules_task(task) for task in tasks]
    df = pd.DataFrame(rows)
    table = df.pivot_table(index=['country', 'adjustment_period'], columns='rules', values='spb_end', sort=False)
    table = table[list(switches)]
    criteria = df.pivot_table(index=['country', 'adjustment_period'], columns='rules', values='binding_criterion',
                              aggfunc='first', sort=False)[list(switches)]
    if out_file is not None:
        with pd.ExcelWriter(out_file) as writer:
            table.to_excel(writer, sheet_name='spb_end')
            criteria.to_excel(writer, sheet_name='binding_criterion')
    return table, criteria
