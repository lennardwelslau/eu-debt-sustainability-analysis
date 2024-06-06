# Import libraries and modules
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import pickle

# Import DSA model class and stochastic subclass
from classes import *

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