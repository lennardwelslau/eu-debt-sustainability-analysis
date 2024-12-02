# Import libraries and modules
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.grid':True,'grid.color':'black','grid.alpha':'0.25','grid.linestyle':'--'})
plt.rcParams.update({'font.size': 14})

# Import DSA model class and stochastic subclass
from classes import *

def run_dsa(
        country_codes, 
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
        fiscal_multiplier_type='com', # type of fiscal multiplier
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
            debt_safeguard_used = debt_safeguard if country != 'AUT' else False
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
                fiscal_multiplier=fiscal_multiplier, # size of the fiscal multiplier
                fiscal_multiplier_type=fiscal_multiplier_type, # type of fiscal multiplier
                bond_data=bond_data, # Use bond level data for repayment profile
                )
                        
            # If stochastic_only is True, only run stochastic projection
            if stochastic_only:
                if prob_target:
                    dsa.prob_target = prob_target
                dsa.find_spb_stochastic()
                results_dict[country][adjustment_period]['spb_target_dict'] = {'stochastic': dsa.spb_target}
                results_dict[country][adjustment_period]['df_dict'] = {'stochastic': dsa.df(all=True)}

            # If stochastic_only is False, run full DSA with deterministic and stochastic projection
            else:
                dsa.find_spb_binding(
                    save_df=True, 
                    edp=edp, 
                    debt_safeguard=debt_safeguard_used, 
                    deficit_resilience=deficit_resilience,
                    stochastic_criteria=stochastic_criteria
                    )
                results_dict[country][adjustment_period]['spb_target_dict'] = dsa.spb_target_dict
                results_dict[country][adjustment_period]['df_dict'] = dsa.df_dict
                results_dict[country][adjustment_period]['binding_parameter_dict'] = dsa.binding_parameter_dict 

            dsa.project()
            results_dict[country][adjustment_period]['df_dict']['no_policy_change'] = dsa.df(all=True)

    with open(f'../output/{folder_name}/{file_name}.pkl', 'wb') as f:
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
            spb_steps = np.copy(results_dict[country][adjustment_period]['binding_parameter_dict']['spb_steps'])

            # First adjustment step is lowered by size of investment shock
            investment_shock = 0.5
            spb_steps[0] -= investment_shock

            # Create new instance of DSA model
            dsa = StochasticDsaModel(
                country=country, 
                adjustment_period=adjustment_period, 
                shock_frequency='quarterly'
                )
            
            # Set predefined adjustment steps, drop final one for reoptimization
            dsa.predefined_spb_steps = spb_steps[:-1]

            # Find binding spb, without safeguards since those are already included in the adjustment steps
            dsa.find_spb_binding(edp=False, 
                                 debt_safeguard=False, 
                                 deficit_resilience=False,
                                 print_results=False)

            # if dsa.spb_binding < spb_binding baseline, increase by 0.5 
            if dsa.spb_target < results_dict[country][adjustment_period]['binding_parameter_dict']['spb_target']:
                dsa.spb_steps[-1] += investment_shock
                dsa.project(spb_steps=dsa.spb_steps)
            
            # if binding debt_safeguard, simply use old adjustment steps
            if results_dict[country][adjustment_period]['binding_parameter_dict']['criterion'] == 'debt_safeguard':
                dsa.spb_target = results_dict[country][adjustment_period]['binding_parameter_dict']['spb_target']
                dsa.project(spb_target=dsa.spb_target)

            # Safe df to results dict
            results_dict[country][adjustment_period]['df_dict']['inv'] = dsa.df(all=True)

    # Save results dict    
    with open(f'../../output/{folder_name}/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

def run_consecutive_dsa(
        country,
        adjustment_start=2025, 
        initial_adjustment_period=7, 
        consecutive_adjustment_period=4, 
        number_of_adjustment_periods=3, 
        scenario_data=None,
        scenario=None,
        debt_safeguard=True,
        deficit_resilience=True,
        edp=True,
        print_results=False,
        plot_results=False):
    """
    Performs DSA for consecutive adjustment periods and returns results in a DataFrame.
    """
    results = {}

    # Loop over consecutive adjustment periods
    for i in range(number_of_adjustment_periods):
        print_results = False if i < number_of_adjustment_periods else True

        adjustment_period = initial_adjustment_period + consecutive_adjustment_period * i
        dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period)

        # If demographic scenario data is provided, set ageing cost and potential GDP growth
        if scenario_data and scenario != 'baseline':
            dsa.ageing_cost = scenario_data[scenario]['cost'].loc[country].to_numpy()
            rg_pot_scenario = scenario_data[scenario]['gdp'].loc[country].to_numpy()
            for t, y in enumerate(range(dsa.adjustment_start, dsa.end_year + 1)):
                if t > 5:
                    dsa.rg_pot[t] = rg_pot_scenario[t]
                    dsa.rgdp_pot[t] = dsa.rgdp_pot[t - 1] * (1 + dsa.rg_pot[t] / 100) 
            dsa.rgdp_pot_bl = dsa.rgdp_pot.copy()
            dsa._project_gdp()

        # For first adjustment period, use initial adjustment period
        if i == 0:
            dsa.find_spb_binding(debt_safeguard=debt_safeguard, deficit_resilience=deficit_resilience, edp=edp, print_results=print_results, save_df=True)

        # For consecutive adjustment periods, use previous spb steps as pedefined steps
        else:
            dsa.predefined_spb_steps = spb_steps # np.concatenate([spb_steps, np.nan * np.ones(consecutive_adjustment_period)])
            dsa.find_spb_binding(debt_safeguard=debt_safeguard, deficit_resilience=deficit_resilience, edp=edp, print_results=print_results, save_df=True)
        
        spb_steps = dsa.spb_steps
        results[f'adjustment_period_{i+1}'] = dsa.spb_target_dict
        print(dsa.spb_target_dict)

    # Plot results if requested
    if plot_results:
        df = dsa.df().loc[:30].reset_index().set_index('y')
        ax = df[['ob', 'sb', 'spb_bca']].plot(legend=False, lw=2)
        ax2 = df['d'].plot(secondary_y=True, legend=False, lw=2)

        # Adding vertical spans
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plt.axvspan(adjustment_start, adjustment_start + initial_adjustment_period, color=colors[0], alpha=0.1, label='adj. 1')
        for i in range(number_of_adjustment_periods-1):
            start = adjustment_start + initial_adjustment_period + consecutive_adjustment_period * i
            end = adjustment_start + initial_adjustment_period + consecutive_adjustment_period * (i + 1)
            plt.axvspan(start, end, color=colors[i+1], alpha=0.1, label=f'adj. {i+2}')

        # hline for 3 and 1.5
        ax.axhline(-3, color='black', linestyle='--', label='3%', alpha=0.5)
        ax.axhline(-1.5, color='black', linestyle='-.', label='1.5%', alpha=0.5)

        # Collecting handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

        plt.title(f'{country}: {initial_adjustment_period}-year, followed by {number_of_adjustment_periods-1}x {consecutive_adjustment_period}-year adjustment')
        # legend underneath plot
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

        dsa.plot_consecutive_dsa = plt.gcf()
        plt.show()

    return dsa