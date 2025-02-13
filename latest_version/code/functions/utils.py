# NOTE: Since introduction of the GroupDsaModel, many of the functions here are redundant.
# A more efficent way to run DSA is to use the GroupDsaModel class and its methods.

# Import libraries and modules
import os
base_dir = '../' * (os.getcwd().split(os.sep)[::-1].index('code')+1)
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format

# Import DSA model class and stochastic subclass
from classes import StochasticDsaModel as DSA

def create_results_dict(country_codes, adjustment_periods=[4, 7]):
    """
    Create results dictionary for analysis.
    """
    results_dict = {}
    for country in country_codes:
        results_dict[country] = {}
        for adjustment_period in adjustment_periods:
            results_dict[country][adjustment_period] = {}
    return results_dict

def add_output_folder(folder_name):
    """
    Create output folder and results dictionary for analysis.
    If folder already exists, load results dictionary.
    """
    output_path = f'{base_dir}output/{folder_name}'
    results_charts_path = f'{output_path}/charts'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(results_charts_path)

def load_results_dict(folder_name):
    """
    Load results dictionary from output folder.
    """ 
    return pd.read_pickle(f'{base_dir}output/{folder_name}/results_dict.pkl')

def save_results(
        results_dict,
        folder_name, 
        save_dfs=True,
        ):
    """
    Saves DSA results to excel.
    """
    _save_spb_table(results_dict, folder_name)
    if save_dfs == True: _save_dfs(results_dict, folder_name)

def _save_spb_table(results_dict, folder_name):
    """
    Saves spb_targets for each instance in output folder.
    """    
    # Create df_spb from dsa_dict
    df_spb = pd.DataFrame()
    for country in results_dict.keys():
        for adjustment_period in results_dict[country].keys():
            spb_target_dict = results_dict[country][adjustment_period]['spb_target_dict']
            for scenario in spb_target_dict.keys():
                df = pd.DataFrame(columns=['country', 'adjustment_period', 'scenario', 'spbstar'])
                df.loc[0] = [country, adjustment_period, scenario, spb_target_dict[scenario]]
                df_spb = pd.concat([df_spb, df])
    df_spb = df_spb.pivot(index=['country', 'adjustment_period'], columns='scenario', values='spbstar').reset_index()
    
    # Get binding DSA scenario
    dsa_col_list = ['main_adjustment', 'adverse_r_g', 'lower_spb', 'financial_stress', 'stochastic']
    dsa_col_list = [col for col in dsa_col_list if col in df_spb.columns]
    df_spb['binding_dsa'] = df_spb[dsa_col_list].max(axis=1)
    
    # Get binding safeguard scenario
    safeguard_col_list = ['deficit_reduction', 'debt_safeguard', 'deficit_resilience']
    safeguard_col_list = [col for col in safeguard_col_list if col in df_spb.columns]
    df_spb['binding_safeguard'] = df_spb[safeguard_col_list].max(axis=1)
    df_spb.rename(columns={'main_adjustment_deficit_reduction': 'deficit_reduction'}, inplace=True)
    
    # add country names
    df_spb['iso'] = df_spb['country']
    country_code_dict = {
        'AUT': 'Austria',
        'BEL': 'Belgium',
        'BGR': 'Bulgaria',
        'HRV': 'Croatia',
        'CYP': 'Cyprus',
        'CZE': 'Czechia',
        'DNK': 'Denmark',
        'EST': 'Estonia',
        'FIN': 'Finland',
        'FRA': 'France',
        'DEU': 'Germany',
        'GRC': 'Greece',
        'HUN': 'Hungary',
        'IRL': 'Ireland',
        'ITA': 'Italy',
        'LVA': 'Latvia',
        'LTU': 'Lithuania',
        'LUX': 'Luxembourg',
        'MLT': 'Malta',
        'NLD': 'Netherlands',
        'POL': 'Poland',
        'PRT': 'Portugal',
        'ROU': 'Romania',
        'SVK': 'Slovakia',
        'SVN': 'Slovenia',
        'ESP': 'Spain',
        'SWE': 'Sweden',
        }
    df_spb['country'] = df_spb['country'].map(country_code_dict)

    # Sort columns
    col_order = ['country',
                'iso',
                'adjustment_period', 
                'main_adjustment',
                'adverse_r_g',            
                'lower_spb', 
                'financial_stress',
                'stochastic', 
                'binding_dsa',
                'deficit_reduction', 
                'edp',
                'debt_safeguard',
                'deficit_resilience',
                'binding_safeguard',
                'binding'
                ]
    
    for col in col_order:
        if col not in df_spb.columns: df_spb[col] = np.nan
    df_spb = df_spb[col_order].sort_values(['adjustment_period', 'country']).round(3)

    # Save to excel
    df_spb.to_excel(f'{base_dir}output/{folder_name}/results_spb.xlsx', index=False)

def _save_dfs(results_dict, folder_name):
    """
    Saves dfs for each country, adjustment period and scenario in dsa_dict.
    """
    with pd.ExcelWriter(f'{base_dir}output/{folder_name}/results_timeseries.xlsx') as writer:
        for country in results_dict.keys():
            for adjustment_period in results_dict[country].keys():
                df_dict = results_dict[country][adjustment_period]['df_dict']
                for scenario in df_dict.keys():
                    sheet_name = f'{country}_{adjustment_period}_{scenario}'[:31] # limit sheet_name length to 31 characters
                    df = df_dict[scenario]
                    df.to_excel(writer, sheet_name=sheet_name)