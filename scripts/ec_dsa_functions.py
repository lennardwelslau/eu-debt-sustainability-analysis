# Functions accompanying the Jupyter Notebook ec_dsa_main for the reproduction of the 
# results of the Bruegel Working Paper "A Quantitative Evaluation of the European 
# Commission´s Fiscal Governance Proposal" by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). 

# Import libraries and modules
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style('whitegrid')

# Import DSA model class and stochastic subclass
from EcDsaModelClass import EcDsaModel
from EcStochasticModelClass import EcStochasticModel

def run_deterministic_scenario_loop(eu_codes, results_dict, var_list):
    """
    Runs all deterministic scenarios for a given list of countries and stores the results in a dictionary.
    """
    ## Deterministic scenario loop
    start_time = time.time()
    total_countries = len(eu_codes)
    for counter, country in enumerate(eu_codes):
        counter += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = round((elapsed_time / counter) * (total_countries - counter) / 60, 1)
        print(f'{counter} of {total_countries} - Estimated remaining time: {estimated_remaining_time} minutes', end='\r')

        for adjustment_period in [4, 7]:
            
            # NFPC scenario
            if adjustment_period == 4:
                run_deterministic_scenario(
                    var_list=var_list,
                    results_dict=results_dict,
                    country=country, 
                    adjustment_period=4, 
                    scenario=None, 
                    criterion=None,
                    scenario_name='nfpc'
                    )

            # Main_adjustment and adverse scenarios
            for scenario in ['main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb']:
                run_deterministic_scenario(
                    var_list=var_list,
                    results_dict=results_dict,
                    country=country, 
                    adjustment_period=adjustment_period, 
                    scenario=scenario, 
                    criterion='debt_decline', 
                    scenario_name=scenario
                    )

            # Main_adjustment with deficit/expenditure/debt criteria
            for criterion in ['deficit_reduction', 'expenditure_safeguard', 'debt_safeguard']:
                if criterion == 'debt_safeguard' and adjustment_period == 7:
                    continue
                else:
                    run_deterministic_scenario(
                        var_list=var_list,
                        results_dict=results_dict,
                        country=country, 
                        adjustment_period=adjustment_period, 
                        scenario='main_adjustment', 
                        criterion=criterion, 
                        scenario_name=criterion
                        )

        # Set nfpc and debt safeguard 7 year results equal to 4 year results
        for scenario in ['nfpc', 'debt_safeguard']:
            if scenario in results_dict[country][4].keys():
                results_dict[country][7][scenario] = {}
                for key in results_dict[country][4][scenario].keys():
                    results_dict[country][7][scenario][key] = results_dict[country][4][scenario][key]

def run_deterministic_scenario(var_list, results_dict, country, adjustment_period, scenario=None, criterion=None, scenario_name='scenario'):
    """
    Runs a deterministic scenario for a given country, adjustment period, scenario and criterion.   
    """
    try:
        dsa = EcDsaModel(country=country, adjustment_period=adjustment_period)
        if scenario_name == 'nfpc':
            dsa.project() # No optimization needed if no fiscal policy change
        else:
            dsa.find_spb_deterministic(scenario=scenario, criterion=criterion)
        df = dsa.df(var_list)
        results_dict[country][adjustment_period][scenario_name] = {}
        results_dict[country][adjustment_period][scenario_name]['df'] = df
        results_dict[country][adjustment_period][scenario_name]['spbstar'] = dsa.spb_bcoa[dsa.adjustment_end]
        results_dict[country][adjustment_period][scenario_name]['ob'] = dsa.ob[dsa.adjustment_end]
        results_dict[country][adjustment_period][scenario_name]['d_T+A+10'] = dsa.d[dsa.adjustment_end+10]
        results_dict[country][adjustment_period][scenario_name]['high_deficit'] = np.any(dsa.ob[dsa.adjustment_start:dsa.adjustment_end+1] < -3)
        results_dict[country][adjustment_period][scenario_name]['annual_adjustment'] = dsa.spb_bcoa[dsa.adjustment_start] - dsa.spb_bcoa[dsa.adjustment_start-1]
        results_dict[country][adjustment_period][scenario_name]['debt_safeguard_satisfied'] = dsa.d[dsa.adjustment_start-1] > dsa.d[dsa.adjustment_start+3]

    except:
        pass

def run_stochastic_scenario_loop(eu_codes, results_dict, output_path, var_list, today):
    """
    Runs all stochastic scenarios for a given list of countries and stores the results in a dictionary.
    """
    start_time = time.time()
    total_countries = len(eu_codes)
    
    for counter, country in enumerate(eu_codes):
        counter += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = round((elapsed_time / counter) * (total_countries - counter) / 60, 1)
        print(f'{counter} of {total_countries} - Estimated remaining time: {estimated_remaining_time} minutes', end='\r')

        for adjustment_period in [4, 7]:
            try:
                dsa = EcStochasticModel(country=country, adjustment_period=adjustment_period)
                dsa.find_spb_stochastic()
                df = dsa.df(var_list)
                dsa.fanchart(save_as=f'{output_path}/fancharts/fanchart_{country}_{adjustment_period}_{today}.jpeg', save_df=True, show=False)
                results_dict[country][adjustment_period]['stochastic'] = {}
                results_dict[country][adjustment_period]['stochastic']['df'] = df
                results_dict[country][adjustment_period]['stochastic']['df_fanchart'] = dsa.df_fanchart
                results_dict[country][adjustment_period]['stochastic']['spbstar'] = dsa.spb_bcoa[dsa.adjustment_end]
                results_dict[country][adjustment_period]['stochastic']['ob'] = dsa.ob[dsa.adjustment_end]
                results_dict[country][adjustment_period]['stochastic']['d_T+A+10'] = dsa.d[dsa.adjustment_end+10]
                results_dict[country][adjustment_period]['stochastic']['high_deficit'] = np.any(dsa.ob[dsa.adjustment_start:dsa.adjustment_end+1] < -3)
                results_dict[country][adjustment_period]['stochastic']['annual_adjustment'] = dsa.spb_bcoa[dsa.adjustment_start] - dsa.spb_bcoa[dsa.adjustment_start-1]
                results_dict[country][adjustment_period]['stochastic']['debt_safeguard_satisfied'] = dsa.d[dsa.adjustment_start-1] > dsa.d[dsa.adjustment_start+3]
            except:
                pass

def run_post_debt_safeguard_loop(results_dict, var_list):
    """
    Runs scenarios after a binding debt safeguard in the 7-year scenario.
    """
    for country, adjustment_period_dict in results_dict.items():
        for adjustment_period, scenario_dict in adjustment_period_dict.items():
            if adjustment_period == 4:
                continue

            # Find max spbstar for adjustment period
            max_spbstar = - np.inf
            max_spbstar_key = None
            for scenario, variable_dict in scenario_dict.items():
                if scenario in [
                    'main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb',  'deficit_reduction', 'debt_safeguard', 'expenditure_safeguard', 'stochastic'
                    ] and 'spbstar' in variable_dict.keys():
                    spbstar_value = variable_dict['spbstar']
                    if spbstar_value > max_spbstar:
                        max_spbstar = spbstar_value
                        max_spbstar_key = (country, adjustment_period, max_spbstar, scenario)
            
            # Check if debt_safeguard satisfied in toughest scenario
            if max_spbstar_key is not None:
                country, adjustment_period, max_spbstar, scenario = max_spbstar_key
                debt_safeguard_satisfied = results_dict[country][adjustment_period][scenario]['debt_safeguard_satisfied']
                if not debt_safeguard_satisfied or scenario == 'debt_safeguard':

                    # Find optimal spbstar with deficit procedure
                    print('Calculating optimal spb for', country, adjustment_period, max_spbstar, 'from', scenario)
                    dsa = EcStochasticModel(country=country, adjustment_period=adjustment_period)
                    try:
                        dsa.find_spb_post_debt_safeguard()
                    except:
                        continue
                    df = dsa.df(var_list)
                    results_dict[country][adjustment_period]['post_debt_safeguard'] = {}
                    results_dict[country][adjustment_period]['post_debt_safeguard']['df'] = df
                    results_dict[country][adjustment_period]['post_debt_safeguard']['spbstar'] = dsa.spb_bcoa[dsa.adjustment_end]
                    results_dict[country][adjustment_period]['post_debt_safeguard']['ob'] = dsa.ob[dsa.adjustment_end]
                    results_dict[country][adjustment_period]['post_debt_safeguard']['d_T+A+10'] = dsa.d[dsa.adjustment_end+10]
                    results_dict[country][adjustment_period]['post_debt_safeguard']['high_deficit'] = np.any(dsa.ob[dsa.adjustment_start:dsa.adjustment_end+1] < -3)
                    results_dict[country][adjustment_period]['post_debt_safeguard']['annual_adjustment'] = dsa.spb_bcoa[dsa.adjustment_start] - dsa.spb_bcoa[dsa.adjustment_start-1]
                    results_dict[country][adjustment_period]['post_debt_safeguard']['spb_initial_adjustment_period'] = dsa.spb_initial_adjustment_period
                    results_dict[country][adjustment_period]['post_debt_safeguard']['spb_initial_adjustment_step'] = dsa.spb_initial_adjustment_step

def run_deficit_safeguard_loop(results_dict, var_list):
    """
    Runs scenarios after a binding deficit safeguard.
    """
    for country, adjustment_period_dict in results_dict.items():
        for adjustment_period, scenario_dict in adjustment_period_dict.items():

            # Find max spbstar for adjustment period
            max_spbstar = - np.inf
            max_spbstar_key = None
            for scenario, variable_dict in scenario_dict.items():
                if scenario in [
                    'main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb',  'deficit_reduction', 'debt_safeguard', 'expenditure_safeguard', 'stochastic'
                    ] and 'spbstar' in variable_dict.keys():
                    spbstar_value = variable_dict['spbstar']
                    if spbstar_value > max_spbstar:
                        max_spbstar = spbstar_value
                        max_spbstar_key = (country, adjustment_period, max_spbstar, scenario)
            
            # Check if toughest scenario has excessive deficit and low adjustment
            if max_spbstar_key is not None:
                country, adjustment_period, max_spbstar, scenario = max_spbstar_key
                high_deficit = results_dict[country][adjustment_period][scenario]['high_deficit']
                annual_adjustment = results_dict[country][adjustment_period][scenario]['annual_adjustment']
                if high_deficit and (annual_adjustment < 0.5):

                    # Find optimal spbstar with deficit procedure
                    print('Calculating optimal spb for', country, adjustment_period, max_spbstar, 'from', scenario)
                    dsa = EcStochasticModel(country=country, adjustment_period=adjustment_period)
                    dsa.find_spb_deficit()
                    df = dsa.df(var_list)

                    # Save results
                    results_dict[country][adjustment_period]['deficit_safeguard'] = {}
                    results_dict[country][adjustment_period]['deficit_safeguard']['df'] = df
                    results_dict[country][adjustment_period]['deficit_safeguard']['spbstar'] = dsa.spb_bcoa[dsa.adjustment_end]
                    results_dict[country][adjustment_period]['deficit_safeguard']['ob'] = dsa.ob[dsa.adjustment_end]
                    results_dict[country][adjustment_period]['deficit_safeguard']['d_T+A+10'] = dsa.d[dsa.adjustment_end+10]
                    results_dict[country][adjustment_period]['deficit_safeguard']['high_deficit'] = np.any(dsa.ob[dsa.adjustment_start:dsa.adjustment_end+11] < -3)
                    results_dict[country][adjustment_period]['deficit_safeguard']['annual_adjustment'] = dsa.spb_bcoa[dsa.adjustment_start] - dsa.spb_bcoa[dsa.adjustment_start-1]
                    results_dict[country][adjustment_period]['deficit_safeguard']['spb_initial_adjustment_period'] = dsa.spb_initial_adjustment_period
                    results_dict[country][adjustment_period]['deficit_safeguard']['spb_initial_adjustment_step'] = dsa.spb_initial_adjustment_step

def run_binding_baseline_loop(results_dict, var_list):
    """
    Runs loop with baseline assumptions and binding structural primary balance targets.
    """
    for country, adjustment_period_dict in results_dict.items():
        for adjustment_period, scenario_dict in adjustment_period_dict.items():
            max_spbstar = - np.inf
            max_spbstar_key = None
            spbstar_frontloaded_key = None
            for scenario, variable_dict in scenario_dict.items():
                if scenario in [
                    'main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb', 'stochastic', 'deficit_reduction', 'debt_safeguard', 'expenditure_safeguard'
                    ] and 'spbstar' in variable_dict.keys():
                    spbstar_value = variable_dict['spbstar']
                    if spbstar_value > max_spbstar:
                        max_spbstar = spbstar_value
                        spb_initial_adjustment_period = 0
                        spb_initial_adjustment_step = 0
                        max_spbstar_key = (country, adjustment_period, spb_initial_adjustment_period, spb_initial_adjustment_step, max_spbstar, scenario)
                elif scenario in [
                    'post_debt_safeguard', 'deficit_safeguard'
                    ] and 'spbstar' in variable_dict.keys():
                    spbstar_frontloaded = variable_dict['spbstar']
                    spb_initial_adjustment_period = variable_dict['spb_initial_adjustment_period']
                    spb_initial_adjustment_step = variable_dict['spb_initial_adjustment_step']
                    spbstar_frontloaded_key = (country, adjustment_period, spb_initial_adjustment_period, spb_initial_adjustment_step, spbstar_frontloaded, scenario)

            
            # Run baseline scenario with max spbstar or frontloaded spbstar if defined
            if max_spbstar_key is not None and spbstar_frontloaded_key is None:
                country, adjustment_period, spb_initial_adjustment_period, spb_initial_adjustment_step, spbstar, scenario = max_spbstar_key
            
            elif spbstar_frontloaded_key is not None:
                country, adjustment_period, spb_initial_adjustment_period, spb_initial_adjustment_step, spbstar, scenario = spbstar_frontloaded_key

            dsa = EcDsaModel(country=country, adjustment_period=adjustment_period)
            dsa.project(spb_target=spbstar, spb_initial_adjustment_period=spb_initial_adjustment_period, spb_initial_adjustment_step=spb_initial_adjustment_step)
            df = dsa.df(var_list)

            results_dict[country][adjustment_period]['baseline_binding_spbstar'] = {}
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['df'] = df
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['spbstar'] = dsa.spb_bcoa[dsa.adjustment_end]
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['ob'] = dsa.ob[dsa.adjustment_end]
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['d_T+A+10'] = dsa.d[dsa.adjustment_end+10]
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['high_deficit'] = np.any(dsa.ob[dsa.adjustment_start:dsa.adjustment_end+10] < -3)
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['annual_adjustment'] = dsa.spb_bcoa[dsa.adjustment_start] - dsa.spb_bcoa[dsa.adjustment_start-1]
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['spb_initial_adjustment_period'] = dsa.spb_initial_adjustment_period
            results_dict[country][adjustment_period]['baseline_binding_spbstar']['spb_initial_adjustment_step'] = dsa.spb_initial_adjustment_step

def run_deficit_prob_loop(results_dict):
    """
    Runs loop with baseline assumptions and binding structural primary balance targets.
    """

    # Initialize deficit_prob_dict
    deficit_prob_dict = {}

    # Get df of binding scenario
    for country, adjustment_period_dict in results_dict.items():
        deficit_prob_dict[country] = {}
        for adjustment_period, scenario_dict in adjustment_period_dict.items():
            deficit_prob_dict[country][adjustment_period] = {}
            
            # Set deficit parameters to baseline
            spbstar = scenario_dict['baseline_binding_spbstar']['spbstar']
            spb_initial_adjustment_period = scenario_dict['baseline_binding_spbstar']['spb_initial_adjustment_period']
            spb_initial_adjustment_step = scenario_dict['baseline_binding_spbstar']['spb_initial_adjustment_step']

            # Find probability of excessive deficit
            try:
                dsa = EcStochasticModel(country=country, adjustment_period=adjustment_period)
                dsa.find_deficit_prob(spb_target=spbstar, spb_initial_adjustment_period=spb_initial_adjustment_period, spb_initial_adjustment_step=spb_initial_adjustment_step)
                deficit_prob_dict[country][adjustment_period]['deficit_prob'] = dsa.prob_deficit
            except:
                deficit_prob_dict[country][adjustment_period]['deficit_prob'] = np.full(adjustment_period, np.nan)
    
    return deficit_prob_dict

def save_deficit_prob(deficit_prob_dict, output_path, today):
    """
    Saves deficit probabilities to Excel file.
    """
    columns_4 = ['Country'] + [str(2025 + i) for i in range(4)]
    columns_7 = ['Country'] + [str(2025 + i) for i in range(7)]

    data_4 = []
    data_7 = []

    for country, values in deficit_prob_dict.items():
        row_4 = [country] + list(values.get(4, {'deficit_prob': []})['deficit_prob'])
        row_7 = [country] + list(values.get(7, {'deficit_prob': []})['deficit_prob'])
        data_4.append(row_4)
        data_7.append(row_7)

    df_4 = pd.DataFrame(data_4, columns=columns_4)
    df_7 = pd.DataFrame(data_7, columns=columns_7)

    # Create an Excel writer
    with pd.ExcelWriter(f'{output_path}/ec_dsa_results_prob_deficit_{today}.xlsx') as writer:
        df_4.to_excel(writer, sheet_name='4-year', index=False)
        df_7.to_excel(writer, sheet_name='7-year', index=False)

def save_timeseries(results_dict, output_path, today):
    """
    Saves timeseries from results dictionary to Excel file.
    """
    # Convert results_dict dataframes to an Excel file, name sheets according to country-adjustment_period-scenario
    with pd.ExcelWriter(f'{output_path}/ec_dsa_results_timeseries_{today}.xlsx') as writer:
            for country in results_dict.keys():
                try:
                    for adjustment_period in results_dict[country].keys():
                        for scenario in results_dict[country][adjustment_period].keys():
                            sheet_name = f'{country}_{adjustment_period}_{scenario}'[:31] # limit sheet_name length to 31 characters
                            df = results_dict[country][adjustment_period][scenario]['df']
                            df.to_excel(writer, sheet_name=sheet_name)
                except:
                    continue

def save_spbstar(results_dict, output_path, today):
    """
    Saves spbstar values from results dictionary to Excel file. Notes binding scenario.
    """
        
    # Save end of adjustment period 'spbstar'
    try:
        df_spb = pd.DataFrame()
        for country in results_dict.keys():
                for adjustment_period in results_dict[country].keys():
                    for scenario in results_dict[country][adjustment_period].keys():
                        spbstar_value = results_dict[country][adjustment_period][scenario]['spbstar']
                        df = pd.DataFrame(columns=['country', 'adjustment_period', 'scenario', 'spbstar'])
                        df.loc[0] = [country, adjustment_period, scenario, spbstar_value]
                        df_spb = pd.concat([df_spb, df])
    except:
        raise
        
    df_spb.reset_index(drop=True, inplace=True)
    df_spb = df_spb.pivot(index=['country', 'adjustment_period'], columns='scenario', values='spbstar').reset_index()
    df_spb['binding_scenario'] = df_spb[[
        'main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb', 'stochastic', 'deficit_reduction', 'debt_safeguard', 'post_debt_safeguard', 'deficit_safeguard', 'expenditure_safeguard'
        ]].idxmax(axis=1)

    df_spb.loc[df_spb['deficit_safeguard'].notnull(), 'binding_scenario'] = 'deficit_safeguard'
    df_spb.loc[df_spb['post_debt_safeguard'].notnull(), 'binding_scenario'] = 'post_debt_safeguard'

    df_spb['binding_spb'] = df_spb['baseline_binding_spbstar']

    df_spb['binding_spb'] = df_spb['baseline_binding_spbstar']
    df_spb['frontloaded_safeguard'] = df_spb['deficit_safeguard'].fillna(df_spb['post_debt_safeguard'])
    df_spb.drop(columns=['deficit_safeguard', 'post_debt_safeguard'], inplace=True)


    var_order = ['country', 'adjustment_period', 'nfpc', 'main_adjustment', 'adverse_r_g', 'financial_stress', 'lower_spb', 'deficit_reduction', 'stochastic', 'debt_safeguard', 'frontloaded_safeguard', 'expenditure_safeguard', 'binding_scenario', 'binding_spb']
    df_spb = df_spb[var_order]
    df_spb.to_excel(f'{output_path}/ec_dsa_results_spbstar_{today}.xlsx', index=False)

def save_fancharts(results_dict, output_path, today):
    """
    Saves fancharts from stochastic scenario to Excel file.
    """

    # Convert results_dict fanchart dataframes to an Excel file, name sheets according to country-adjustment_period
    with pd.ExcelWriter(f'{output_path}/ec_dsa_results_fancharts_{today}.xlsx') as writer:
            for country in results_dict.keys():
                try:
                    for adjustment_period in results_dict[country].keys():
                        sheet_name = f'{country}_{adjustment_period}_fanchart'[:31] # limit sheet name length to 31 characters
                        df_fanchart = results_dict[country][adjustment_period]['stochastic']['df_fanchart']
                        df_fanchart.to_excel(writer, sheet_name=sheet_name)
                except:
                    pass

def calc_fig2(results_dict):
    """
    Calculates data for figure 2.
    """

    # Initialize fig2_chart_dict
    fig2_chart_dict = {}

    for scenario in ['main_adjustment', 'baseline_binding_spbstar']:
        fig2_chart_dict[scenario] = {}
        df = results_dict['BEL'][4][scenario]['df'].reset_index()
        df_rg = results_dict['BEL'][4]['adverse_r_g']['df'].reset_index()

        # Set deficit parameters to baseline
        spbstar = results_dict['BEL'][4][scenario]['spbstar']

        # Create df for chart a)
        df_interest_ageing_growth = df[['y', 'iir', 'ageing_cost', 'ng']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
        df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Ageing costs', 'Implicit interest rate', 'Nominal GDP growth']]
        if scenario == 'baseline_binding_spbstar':
            df_rg = results_dict['BEL'][4]['adverse_r_g']['df'][['iir', 'ng']].reset_index().rename(columns={'iir': 'Implicit interest rate - adverse r-g', 'ng': 'Nominal GDP growth - adverse r-g'})
            df_interest_ageing_growth['Implicit interest rate - adverse r-g'] = df_rg['Implicit interest rate - adverse r-g']
            df_interest_ageing_growth['Nominal GDP growth - adverse r-g'] = df_rg['Nominal GDP growth - adverse r-g']
            df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Ageing costs', 'Implicit interest rate', 'Nominal GDP growth', 'Implicit interest rate - adverse r-g', 'Nominal GDP growth - adverse r-g']]
        fig2_chart_dict[scenario]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2050]

        # Create df for chart b)
        df_budget_balance = df[['y', 'spb_bcoa', 'pb', 'ob', 'ageing_component']].rename(columns={'y': 'year', 'spb_bcoa': 'Age-adjusted structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
        df_budget_balance['Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'] + df_budget_balance['ageing_component'] 
        df_budget_balance.drop('ageing_component', axis=1, inplace=True)
        
        # Use .loc to modify the DataFrame without SettingWithCopyWarning
        df_budget_balance.loc[:1+4, 'Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'].iloc[:2+4]

        df_budget_balance = df_budget_balance[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
        fig2_chart_dict[scenario]['df_budget_balance'] = df_budget_balance.set_index('year').loc[:2050]

        # Run fanchart for chart c)
        dsa = EcStochasticModel(country='BEL', adjustment_period=4)
        dsa.project(spb_target=spbstar)
        dsa.simulate()
        dsa.fanchart(save_df=True, show=False)
        df_fanchart = dsa.df_fanchart
        df_fanchart.loc[2+4+6:, 'p10':'p90'] = np.nan
        fig2_chart_dict[scenario]['df_fanchart'] = df_fanchart.set_index('year').loc[:2050]

        return fig2_chart_dict
    
def plot_save_fig2(fig2_chart_dict, output_path, today):
    """
    Plots and saves figure 2.
    """

    # Set color pallette
    sns.set_palette(sns.color_palette('tab10'))
    subtitle_size = 14 
    for scenario in fig2_chart_dict.keys():
        if scenario == 'main_adjustment':
            figure_title = 'A) Adjustment under baseline'
        elif scenario == 'baseline_binding_spbstar':
            figure_title = 'B) SPB* corresponding to adverse r-g conditions with baseline growth and interest rate assumptions'
        
        # Make 1, 3 sublot with df_interest_ageing_growth, df_budget_balance, df_fanchart
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(figure_title, fontsize=16)

        ## Plot df_interest_ageing_growth
        axs[0].set_title('Ageing costs, interest rate, growth', fontsize=subtitle_size)
        df_interest_ageing_growth = fig2_chart_dict[scenario]['df_interest_ageing_growth']
        colors = sns.color_palette("tab10")
        axs[0].plot(df_interest_ageing_growth.index, df_interest_ageing_growth['Ageing costs'], label='Ageing costs', color=colors[0])
        axs1 = axs[0].twinx()  # Create a twin Axes for the secondary y-axis
        axs1.plot(df_interest_ageing_growth.index, df_interest_ageing_growth['Implicit interest rate'], label='Implicit interest rate (right)', color=colors[1])
        axs1.plot(df_interest_ageing_growth.index, df_interest_ageing_growth['Nominal GDP growth'], label='Nominal GDP growth (right)', color=colors[2])

        if scenario == 'baseline_binding_spbstar':
            axs1.plot(df_interest_ageing_growth.index, df_interest_ageing_growth['Implicit interest rate - adverse r-g'], label='Implicit interest rate - adverse r-g (right)', ls=(0,(1,1)), lw=2, color=colors[1])
            axs1.plot(df_interest_ageing_growth.index, df_interest_ageing_growth['Nominal GDP growth - adverse r-g'], label='Nominal GDP growth - adverse r-g (right)', ls=(0,(1,1)), lw=2, color=colors[2])
        axs1.grid(None)

        lines, labels = axs[0].get_legend_handles_labels()
        lines1, labels1 = axs1.get_legend_handles_labels()
        axs[0].legend(lines + lines1, labels + labels1, loc='upper left')
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))



        # Plot df_budget_balance
        axs[1].set_title('Budget balance', fontsize = subtitle_size)
        df_budget_balance = fig2_chart_dict[scenario]['df_budget_balance']
        df_budget_balance.plot(ax=axs[1])
        axs[1].legend(loc='best')

        # Plot df_fanchart
        df_fanchart = fig2_chart_dict[scenario]['df_fanchart']

        # plot, fill between p10 & p90, p20 & p80, p40 & p60
        # plot p50
        # plot baseline
        axs[2].set_title('Debt simulations', fontsize = subtitle_size)
        axs[2].plot(df_fanchart.index, df_fanchart['baseline'], color='blue', marker='o', markerfacecolor='none', markersize=3, label='Deterministic scenario')
        try:
            axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th percentile', alpha=0.8)
            axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th percentile', alpha=0.8)
            axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th percentile', alpha=0.8)
            axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th percentile', alpha=0.8)
            axs[2].plot(df_fanchart.index, df_fanchart['p50'], color='black', alpha=0.8, label='Median')
        except:
            pass
        axs[2].legend(loc='upper right')

        # add grey fill for 10 year post adustment and delete x-axis label
        for i in range(3):
            axs[i].axvspan(df_fanchart.index[2+4], df_fanchart.index[12+4], alpha=0.2, color='grey')
            axs[i].axvline(df_fanchart.index[2], color='grey', linestyle='--', lw=1.5)
            axs[i].set_xlabel('')

        fig.subplots_adjust(top=0.87)

        # Export to jpeg
        plt.savefig(f'{output_path}/results_charts/fig2_BEL_4_{scenario}_{today}.jpeg', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}/results_charts/fig2_BEL_4_{scenario}_{today}.svg', format='svg', bbox_inches='tight')

def plot_save_fig3(results_dict, output_path, today):
    """
    Calculates data, plots and saves figure 3.
    """

    df = results_dict['FRA'][7]['baseline_binding_spbstar']['df'].reset_index()

    # Create df for chart b)
    df_budget_balance = df[['y', 'spb_bcoa', 'pb', 'ob', 'ageing_component']].rename(columns={'y': 'year', 'spb_bcoa': 'Age-adjusted structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
    df_budget_balance['Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'] + df_budget_balance['ageing_component'] 
    df_budget_balance.drop('ageing_component', axis=1, inplace=True)

    # Use .loc to modify the DataFrame without SettingWithCopyWarning
    df_budget_balance.loc[:1+7, 'Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'].iloc[:2+7]
    df_budget_balance = df_budget_balance[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']].set_index('year').loc[:2050]

    # Initiate plot
    sns.set_palette(sns.color_palette('tab10'))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot df_budget_balance
    df_budget_balance.plot(ax=ax)
    ax.legend(loc='best')

    # Layout
    ax.axvspan(df_budget_balance.index[2+4], df_budget_balance.index[12+4], alpha=0.2, color='grey')
    ax.axvline(df_budget_balance.index[2], color='grey', linestyle='--', lw=1.5)
    ax.set_xlabel('')
    fig.subplots_adjust(top=0.87)

    # Export to jpeg
    plt.savefig(f'{output_path}/results_charts/fig3_FRA_7_{today}.jpeg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}/results_charts/fig3_FRA_7_{today}.svg', format='svg', bbox_inches='tight')


def calc_annex_charts(results_dict):
    """
    Calculates data for annex charts.
    """

    # Initialize fig2_chart_dict
    annex_chart_dict = {}

    # 1) get df of binding scenario
    for country, adjustment_period_dict in results_dict.items():
        annex_chart_dict[country] = {}
        for adjustment_period, scenario_dict in adjustment_period_dict.items():
            annex_chart_dict[country][adjustment_period] = {}
            
            try:
                # Safe df for baseline binding scenario
                df = results_dict[country][adjustment_period]['baseline_binding_spbstar']['df'].reset_index()
            except:
                continue

            # Set deficit parameters to baseline
            spbstar = results_dict[country][adjustment_period]['baseline_binding_spbstar']['spbstar']
            spb_initial_adjustment_period = results_dict[country][adjustment_period]['baseline_binding_spbstar']['spb_initial_adjustment_period']
            spb_initial_adjustment_step = results_dict[country][adjustment_period]['baseline_binding_spbstar']['spb_initial_adjustment_step']

            # Create df for chart a)
            df_interest_ageing_growth = df[['y', 'iir', 'ageing_cost', 'ng']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
            df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Ageing costs', 'Implicit interest rate', 'Nominal GDP growth']]
            annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2050]

            # Create df for chart b)
            df_budget_balance = df[['y', 'spb_bcoa', 'pb', 'ob', 'ageing_component']].rename(columns={'y': 'year', 'spb_bcoa': 'Age-adjusted structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
            df_budget_balance['Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'] + df_budget_balance['ageing_component'] 
            df_budget_balance.drop('ageing_component', axis=1, inplace=True)
            df_budget_balance.loc[:1+adjustment_period, 'Structural primary balance'] = df_budget_balance['Age-adjusted structural primary balance'].iloc[:2+adjustment_period]
            
            df_budget_balance = df_budget_balance[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
            annex_chart_dict[country][adjustment_period]['df_budget_balance'] = df_budget_balance.set_index('year').loc[:2050]

            # Run fanchart for chart c)
            try:
                if country == 'BGR':
                    raise Exception('BGR has no viable fanchart because of restricted sample period')
                dsa = EcStochasticModel(country=country, adjustment_period=adjustment_period)
                dsa.project(spb_target=spbstar, spb_initial_adjustment_period=spb_initial_adjustment_period, spb_initial_adjustment_step=spb_initial_adjustment_step)
                dsa.simulate()
                dsa.fanchart(save_df=True, show=False)
                df_fanchart = dsa.df_fanchart
                df_fanchart.loc[2+adjustment_period+6:, 'p10':'p90'] = np.nan
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2050]
            except:
                df_fanchart = pd.DataFrame(columns=['year', 'baseline', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90'])
                df_fanchart['year'] = df['y']
                df_fanchart['baseline'] = df['d']
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2050]
    
    return annex_chart_dict
    
def plot_save_annex_charts(annex_chart_dict, output_path, eu_code_dict, today):
    """
    Plots and saves annex charts.
    """
    # Set color pallette
    sns.set_palette(sns.color_palette('tab10'))
    subtitle_size = 14 
    for country in annex_chart_dict.keys():
        for adjustment_period in annex_chart_dict[country].keys():
            
            try:
                # Make 1, 3 sublot with df_interest_ageing_growth, df_budget_balance, df_fanchart
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'{eu_code_dict[country]}: {adjustment_period}-year scenario', fontsize=16)

                # Plot df_interest_ageing_growth
                axs[0].set_title('Ageing costs, interest rate, growth', fontsize = subtitle_size)
                df_interest_ageing_growth = annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth']
                df_interest_ageing_growth.plot(ax=axs[0], secondary_y=['Implicit interest rate', 'Nominal GDP growth'])
                axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                lines = axs[0].get_lines() + axs[0].right_ax.get_lines()
                axs[0].legend(lines, [l.get_label() for l in lines], loc='best')

                # Plot df_budget_balance
                axs[1].set_title('Budget balance', fontsize = subtitle_size)
                df_budget_balance = annex_chart_dict[country][adjustment_period]['df_budget_balance']
                df_budget_balance.plot(ax=axs[1])
                axs[1].legend(loc='best')

                # Plot df_fanchart
                df_fanchart = annex_chart_dict[country][adjustment_period]['df_fanchart']
                axs[2].set_title('Debt simulations', fontsize = subtitle_size)
                axs[2].plot(df_fanchart.index, df_fanchart['baseline'], color='blue', marker='o', markerfacecolor='none', markersize=3, label='Deterministic scenario')
                try:
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th percentile', alpha=0.8)
                    axs[2].plot(df_fanchart.index, df_fanchart['p50'], color='black', alpha=0.8, label='Median')
                except:
                    pass
                axs[2].legend(loc='best')

                # add grey fill for 10 year post adustment and delete x-axis label
                for i in range(3):
                    axs[i].axvspan(df_fanchart.index[2+adjustment_period], df_fanchart.index[12+adjustment_period], alpha=0.2, color='grey')
                    axs[i].axvline(df_fanchart.index[2], color='grey', linestyle='--', lw=1.5)
                    axs[i].set_xlabel('')

                # Increase space between subplots and heading
                fig.subplots_adjust(top=0.87)

                # Export to jpeg
                plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{today}.jpeg', dpi=300, bbox_inches='tight')
                plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{today}.svg', format='svg', bbox_inches='tight')

                plt.close()
            except:
                print(f'Error: {country}_{adjustment_period}')
                plt.close()