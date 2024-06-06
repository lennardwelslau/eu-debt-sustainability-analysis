# Functions accompanying the Jupyter Notebook main for the reproduction of the 
# results of the Bruegel Working Paper "A Quantitative Evaluation of the European 
# Commission´s Fiscal Governance Proposal" by Zsolt Darvas, Lennard Welslau, and 
# Jeromin Zettelmeyer (2023). Updated: 2024-04-16 Preliminary version not for distribution.

# Import libraries and modules
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import pickle

# Import DSA model class and stochastic subclass
from classes import *

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
    output_path = f'../output/{folder_name}'
    results_charts_path = f'{output_path}/charts'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(results_charts_path)

def load_dsa_dict(folder_name):
    """
    Load results dictionary from output folder.
    """ 
    return pd.read_pickle(f'../output/{folder_name}/results_dict.pkl')

def run_dsa(
        country_codes, 
        adjustment_periods, 
        results_dict,
        folder_name, 
        edp=True, 
        debt_safeguard=True, 
        deficit_resilience=True,
        deficit_resilience_post_adjustment=True
        ):
    """
    Runs DSA for all EU countries and saves results individually.
    """
    start_time = time.time()
    total_countries = len(country_codes)
    for counter, country in enumerate(country_codes):
        counter += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = round((elapsed_time / counter) * (total_countries - counter) / 60, 1)
        print(f'\n--> {counter} of {total_countries}, estimated remaining time: {estimated_remaining_time} minutes')
        
        for adjustment_period in adjustment_periods:
            dsa = StochasticDsaModel(
                country=country, 
                adjustment_period=adjustment_period
                )
            dsa.find_spb_binding(
                save_df=True, 
                edp=edp, 
                debt_safeguard=debt_safeguard, 
                deficit_resilience=deficit_resilience,
                deficit_resilience_post_adjustment=deficit_resilience_post_adjustment
                )
            results_dict[country][adjustment_period]['spb_target_dict'] = dsa.spb_target_dict
            results_dict[country][adjustment_period]['df_dict'] = dsa.df_dict
            results_dict[country][adjustment_period]['binding_parameter_dict'] = dsa.binding_parameter_dict  
            dsa.project()
            results_dict[country][adjustment_period]['df_dict']['no_policy_change'] = dsa.df(all=True)

    with open(f'../output/{folder_name}/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print('DSA run completed and saved.')

def save_results(
        results_dict,
        folder_name, 
        ):
    """
    Saves DSA results to excel.
    """

    _save_spbs(results_dict, folder_name)
    _save_dfs(results_dict, folder_name)

def _save_spbs(results_dict, folder_name):
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
    col_list = ['main_adjustment', 'adverse_r_g', 'lower_spb', 'financial_stress', 'stochastic']
    df_spb['binding_dsa'] = df_spb[col_list].max(axis=1)
    
    # Get binding safeguard scenario
    safeguard_col_list = ['deficit_reduction', 'debt_safeguard', 'deficit_resilience']
    safeguard_col_list = [col for col in safeguard_col_list if col in df_spb.columns]
    df_spb['binding_safeguard_council'] = df_spb[safeguard_col_list].max(axis=1)
    df_spb.rename(columns={'main_adjustment_deficit_reduction': 'deficit_reduction'}, inplace=True)
    
    # Sort columns
    col_order = ['country',
                'adjustment_period', 
                'main_adjustment',
                'adverse_r_g',            
                'lower_spb', 
                'financial_stress',
                'stochastic', 
                'binding_dsa',
                'deficit_reduction', 
                'debt_safeguard',
                'deficit_resilience',
                'binding_safeguard',
                'binding',
                'post_adjustment'
                ]
    col_order = [col for col in col_order if col in df_spb.columns]
    df_spb = df_spb[col_order].sort_values(['adjustment_period', 'country']).round(3)
    
    # Save to excel
    df_spb.to_excel(f'../output/{folder_name}/results_spb.xlsx', index=False)

def _save_dfs(results_dict, folder_name):
    """
    Saves dfs for each country, adjustment period and scenario in dsa_dict.
    """
    with pd.ExcelWriter(f'../output/{folder_name}/results_timeseries.xlsx') as writer:
        for country in results_dict.keys():
            for adjustment_period in results_dict[country].keys():
                df_dict = results_dict[country][adjustment_period]['df_dict']
                for scenario in df_dict.keys():
                    sheet_name = f'{country}_{adjustment_period}_{scenario}'[:31] # limit sheet_name length to 31 characters
                    df = df_dict[scenario]
                    df.to_excel(writer, sheet_name=sheet_name)

def run_inv_scenario(
        country_codes, 
        results_dict,
        folder_name,
        adjustment_periods=[7], 
    ):
    """
    Run DSA with temporaryinvestment shock scenario and save results in results_dict
    """
    # Loop over countries and adjustment periods
    for country in country_codes:
        for adjustment_period in adjustment_periods:

            # Load baseline adjustment steps
            adjustment_steps = np.copy(results_dict[country][adjustment_period]['binding_parameter_dict']['adjustment_steps'])

            # First adjustment step is lowered by size of investment shock
            investment_shock = 0.5
            adjustment_steps[0] -= investment_shock

            # Final adjustmnt step is set to nan because we will reoptimize
            adjustment_steps[-1] = np.nan

            # Create new instance of DSA model
            dsa = StochasticDsaModel(
                country=country, 
                adjustment_period=adjustment_period, 
                shock_frequency='quarterly'
                )
            
            # Set predefined adjustment steps
            dsa.predefined_adjustment_steps = adjustment_steps

            # Find binding spb, without safeguards since those are already included in the adjustment steps
            dsa.find_spb_binding(edp=False, 
                                 debt_safeguard=False, 
                                 deficit_resilience=False,
                                 deficit_resilience_post_adjustment=True)

            # if dsa.spb_binding < spb_binding baseline, increase by 0.5 
            if dsa.spb_target < results_dict[country][adjustment_period]['binding_parameter_dict']['spb_target']:
                dsa.adjustment_steps[-1] += investment_shock
                dsa.project(adjustment_steps=dsa.adjustment_steps)
            
            # if binding debt_safeguard, simply use old adjustment steps
            if results_dict[country][adjustment_period]['binding_parameter_dict']['criterion'] == 'debt_safeguard':
                dsa.spb_target = results_dict[country][adjustment_period]['binding_parameter_dict']['spb_target']
                dsa.project(spb_target=dsa.spb_target)

            # Safe df to results dict
            results_dict[country][adjustment_period]['df_dict']['inv'] = dsa.df(all=True)

    # Save results dict    
    with open(f'../output/{folder_name}/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)