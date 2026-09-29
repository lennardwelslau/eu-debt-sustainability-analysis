# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Validation               #
# ========================================================================================= #
#
# Compares model results against the European Commission prior guidance calculation sheets:
#   1. Debt path when the model is given the Commission's SPB adjustment path (tests the debt
#      dynamics and input mapping independently of the optimisation).
#   2. SPB at the end of the adjustment period required by the DSA-based criteria and after
#      safeguards, using find_spb_binding(rules='commission') (tests the optimisation).
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
                            adjustment_periods=(4, 7), seed=0, verbose=True, out_file=None):
    """
    Run the model on an input workbook and compare with the Commission prior guidance sheets.

    Parameters:
        input_file (str): Input workbook, normally built with build_inputs(mode='commission').
        prior_guidance (str): '2024' or 'latest'. None: 'latest' if the file name contains 'latest', else '2024'.
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
                    model.find_spb_binding(rules='commission', stochastic=stochastic, print_results=False)
                ref = model.binding_parameter_dict
                a_dsa = ref['annual_adjustment_dsa']
                row['model_annual_adjustment_dsa'] = a_dsa
                row['model_annual_adjustment_dsa_rounded'] = round(a_dsa, 2)  # already on the 0.01 grid
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
