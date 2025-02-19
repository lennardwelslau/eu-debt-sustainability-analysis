# NOTE: Since introduction of the GroupDsaModel, the run_dsa function and associated utils
# are redundant. The results_dict produced by the GroupDsaModel differs from the run_dsa 
# results_dict, in that in only contains a single adjustment period. 

# Import libraries and modules
import os
base_dir = '../' * (os.getcwd().split(os.sep)[::-1].index('code')+1)
import numpy as np
import pandas as pd
import pickle
import time
pd.options.display.float_format = "{:,.3f}".format

# Import DSA model class and stochastic subclass
from classes import StochasticDsaModel as DSA

def run_dsa(
        countries, 
        adjustment_periods, 
        results_dict,
        folder_name, 
        file_name='results_dict',
        edp_countries=None, # countries to apply EDP to
        edp=True, # apply EDP 
        debt_safeguard=True, # apply debt safeguard
        deficit_resilience=True, # apply deficit resilience safeguard
        stochastic_only=False, # use only stochastic projection
        start_year=2023, # start year of projection, first year is baseline value
        end_year=2070, # end year of projection
        adjustment_start_year=2025, # start year of linear spb_bca adjustment
        ageing_cost_period=10, # number of years for ageing cost adjustment
        shock_sample_start=2000, # start year of shock sample
        stochastic_start_year=None, # start year of stochastic projection
        stochastic_period=5, # number of years for stochastic projection
        stochastic_criteria=['debt_explodes', 'debt_above_60'],   
        prob_target=None, # target probability of debt sustainability     
        shock_frequency='quarterly', # frequency of shocks, 'quarterly' or 'annual'
        estimation='normal', # estimation method for covariance matrix, 'normal' or 'var_cholesky', or 'var_bootstrap'
        fiscal_multiplier=0.75, # size of the fiscal multiplier
        fiscal_multiplier_type='ec', # type of fiscal multiplier
        bond_data=False, # Use bond level data for repayment profile,
        ):
    """
    Runs DSA for all EU countries and saves results individually.
    """
    start_time = time.time()
    total_countries = len(countries)
    for counter, country in enumerate(countries):
        counter += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = round((elapsed_time / counter) * (total_countries - counter) / 60, 1)
        print(f'\n... optimising {country}, {counter} of {total_countries}, estimated remaining time: {estimated_remaining_time} minutes\n')
        
        if edp_countries:
            edp = True if country in edp_countries else False
        for adjustment_period in adjustment_periods:
            model = DSA(
                country=country, 
                adjustment_period=adjustment_period,
                start_year=start_year, # start year of projection, first year is baseline value
                end_year=end_year, # end year of projection
                adjustment_start_year=adjustment_start_year, # start year of linear spb_bca adjustment
                ageing_cost_period=ageing_cost_period, # number of years for ageing cost adjustment
                shock_sample_start=shock_sample_start, # start year of shock sample
                stochastic_start_year=stochastic_start_year, # start year of stochastic projection
                stochastic_period=stochastic_period, # number of years for stochastic projection
                shock_frequency=shock_frequency, # frequency of shocks, 'quarterly' or 'annual'
                estimation=estimation, # estimation method for covariance matrix, 'normal' or 'var'
                fiscal_multiplier=fiscal_multiplier, # size of the fiscal multiplier
                fiscal_multiplier_type=fiscal_multiplier_type, # type of fiscal multiplier
                bond_data=bond_data, # Use bond level data for repayment profile
                )
                        
            # If stochastic_only is True, only run stochastic projection
            if stochastic_only:
                if prob_target:
                    model.prob_target = prob_target
                model.find_spb_stochastic()
                results_dict[country]['spb_target_dict'] = {'stochastic': model.spb_target}
                results_dict[country]['df_dict'] = {'stochastic': model.df(all=True)}

            # If stochastic_only is False, run full DSA with deterministic and stochastic projection
            else:
                model.find_spb_binding(
                    save_df=True, 
                    edp=edp, 
                    debt_safeguard=debt_safeguard, 
                    deficit_resilience=deficit_resilience,
                    stochastic_criteria=stochastic_criteria
                    )
                results_dict[country]['spb_target_dict'] = model.spb_target_dict
                results_dict[country]['df_dict'] = model.df_dict
                results_dict[country]['binding_parameter_dict'] = model.binding_parameter_dict 

            # Project no policy change scenario
            model = DSA(
                country=country, 
                adjustment_period=0,
                start_year=start_year, # start year of projection, first year is baseline value
                end_year=end_year, # end year of projection
                adjustment_start_year=adjustment_start_year, # start year of linear spb_bca adjustment
                ageing_cost_period=ageing_cost_period, # number of years for ageing cost adjustment
                shock_sample_start=shock_sample_start, # start year of shock sample
                stochastic_start_year=stochastic_start_year, # start year of stochastic projection
                stochastic_period=stochastic_period, # number of years for stochastic projection
                shock_frequency=shock_frequency, # frequency of shocks, 'quarterly' or 'annual'
                estimation=estimation, # estimation method for covariance matrix, 'normal' or 'var'
                fiscal_multiplier=fiscal_multiplier, # size of the fiscal multiplier
                fiscal_multiplier_type=fiscal_multiplier_type, # type of fiscal multiplier
                bond_data=bond_data, # Use bond level data for repayment profile
                )
            model.project()
            results_dict[country]['df_dict']['no_policy_change'] = model.df(all=True)

    with open(f'../output/{folder_name}/{file_name}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print('DSA run completed and saved.')


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