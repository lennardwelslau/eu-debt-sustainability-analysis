# Import libraries and modules
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Import DSA model class and stochastic subclass
from classes import *

def run_dsa(
        country_codes, 
        adjustment_periods, 
        results_dict,
        folder_name, 
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
        shock_frequency='quarterly', # frequency of shocks, 'quarterly' or 'annual'
        estimation='normal', # estimation method for covariance matrix, 'normal' or 'var'
        var_method='cholesky', # method for drawing shocks if estimation is 'var', 'cholesky' or 'bootstrap'
        fiscal_multiplier=0.75, # size of the fiscal multiplier
        growth_policy=False, # Growth policy counterfactual 
        growth_policy_effect=0, # Total effect of growth policy on GDP growth
        growth_policy_cost=0, # Total cost of growth policy as percent of GDP
        growth_policy_period=1, # Period of growth policy counterfactual
        bond_data=False, # Use bond level data for repayment profile,
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
        print(f'\n... optimising {country}, {counter} of {total_countries}, estimated remaining time: {estimated_remaining_time} minutes\n')
        
        if edp_countries:
            edp = True if country in edp_countries else False
        for adjustment_period in adjustment_periods:
            dsa = StochasticDsaModel(
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
                var_method=var_method, # method for drawing shocks if estimation is 'var', 'cholesky' or 'bootstrap'
                fiscal_multiplier=fiscal_multiplier, # size of the fiscal multiplier
                growth_policy=growth_policy, # Growth policy counterfactual 
                growth_policy_effect=growth_policy_effect, # Total effect of growth policy on GDP growth
                growth_policy_cost=growth_policy_cost, # Total cost of growth policy as percent of GDP
                growth_policy_period=growth_policy_period, # Period of growth policy counterfactual
                bond_data=bond_data, # Use bond level data for repayment profile
                )
            
            # If stochastic_only is True, only run stochastic projection
            if stochastic_only:
                dsa.find_spb_stochastic()
                results_dict[country][adjustment_period]['spb_target_dict'] = {'stochastic': dsa.spb_target}
                results_dict[country][adjustment_period]['df_dict'] = {'stochastic': dsa.df(all=True)}

            # If stochastic_only is False, run full DSA with deterministic and stochastic projection
            else:
                dsa.find_spb_binding(
                    save_df=True, 
                    edp=edp, 
                    debt_safeguard=debt_safeguard, 
                    deficit_resilience=deficit_resilience,
                    deficit_resilience_post_adjustment=False
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
                                 deficit_resilience_post_adjustment=False,
                                 print_results=False)

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

def run_consecutive_dsa(
        country,
        start_year=2024, 
        initial_adjustment_period=7, 
        consecutive_adjustment_period=4, 
        number_of_adjustment_periods=3, 
        plot_results=False):
    """
    Performs DSA for consecutive adjustment periods and returns results in a DataFrame.
    """
    results = {}
    for i in range(number_of_adjustment_periods):
        adjustment_period = initial_adjustment_period + consecutive_adjustment_period * i
        dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period)
        if i == 0:
            dsa.find_spb_binding()
            adjustment_steps = dsa.adjustment_steps
        else:
            dsa.predefined_adjustment_steps = np.concatenate([adjustment_steps, np.nan * np.ones(consecutive_adjustment_period)])
            dsa.find_spb_binding(deficit_resilience=False, deficit_resilience_post_adjustment=False)
            adjustment_steps = np.concatenate([adjustment_steps, dsa.adjustment_steps[len(adjustment_steps):]])
        results[f'adjustment_period_{i+1}'] = dsa.spb_target_dict

    results_df = pd.DataFrame(results).T

    if plot_results:
        df = dsa.df().loc[:30].reset_index().set_index('y')
        ax = df[['ob', 'sb', 'spb_bca']].plot(legend=False, lw=2)
        ax2 = df['d'].plot(secondary_y=True, legend=False, lw=2)

        # Adding vertical spans
        colors = sns.color_palette('tab10')
        for i in range(number_of_adjustment_periods):
            start = start_year + initial_adjustment_period + consecutive_adjustment_period * i
            end = start_year + initial_adjustment_period + consecutive_adjustment_period * (i + 1)
            plt.axvspan(start - consecutive_adjustment_period, end, color=colors[i], alpha=0.1, label=f'adj. period {i+1}')

        # hline for 3 and 1.5
        ax.axhline(-3, color='black', linestyle='--', label='3%', alpha=0.5)
        ax.axhline(-1.5, color='black', linestyle='-.', label='1.5%', alpha=0.5)

        # Collecting handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

        plt.title(f'{country}: {initial_adjustment_period}-year, followed by {number_of_adjustment_periods-1}x {consecutive_adjustment_period}-year adjustment')
        plt.legend(handles, labels)
        plt.show()

    return results_df