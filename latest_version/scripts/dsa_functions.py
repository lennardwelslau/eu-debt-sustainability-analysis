# Functions accompanying the Jupyter Notebook dsa_main for the reproduction of the 
# results of the Bruegel Working Paper "A Quantitative Evaluation of the European 
# Commission´s Fiscal Governance Proposal" by Zsolt Darvas, Lennard Welslau, and Jeromin Zettelmeyer (2023). 
# Updated: 2023-12-22

# Import libraries and modules
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style('whitegrid')

# Import DSA model class and stochastic subclass
from DsaModelClass import DsaModel
from StochasticDsaModelClass import StochasticDsaModel

################ Main loop for DSA analysis ################

def run_dsa(country_codes, results_dict, output_path, today, inv_shock=False):
    """
    Runs DSA for all EU countries and saves results in a dictionary.
    """
    start_time = time.time()
    total_countries = len(country_codes)
    for counter, country in enumerate(country_codes):
        counter += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = round((elapsed_time / counter) * (total_countries - counter) / 60, 1)
        print(f'\n--> {counter} of {total_countries}, estimated remaining time: {estimated_remaining_time} minutes')
        
        for adjustment_period in [4,7]:
            dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period, inv_shock=inv_shock)
            dsa.find_spb_binding(save_df=True)
            results_dict[country][adjustment_period]['spb_targets'] = dsa.spb_target_dict
            results_dict[country][adjustment_period]['dfs'] = dsa.df_dict
            results_dict[country][adjustment_period]['binding_parameters'] = dsa.binding_parameter_dict

    with open(f'{output_path}/dsa_results_dict_{today}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

################ Functions to save results to excel ################

def save_results(results_dict, output_path, today):
    """
    Saves DSA results to excel.
    """
    _save_spbs(results_dict, output_path, today)
    _save_dfs(results_dict, output_path, today)
    _save_ameco(results_dict, output_path, today)

def _save_spbs(results_dict, output_path, today):
    """
    Saves spb_targets for each country and adjustment period in results_dict.
    """    
    df_spb = pd.DataFrame()
    for country in results_dict.keys():
        for adjustment_period in results_dict[country].keys():
            spb_target_dict = results_dict[country][adjustment_period]['spb_targets']
            for scenario in spb_target_dict.keys():
                df = pd.DataFrame(columns=['country', 'adjustment_period', 'scenario', 'spbstar'])
                df.loc[0] = [country, adjustment_period, scenario, spb_target_dict[scenario]]
                df_spb = pd.concat([df_spb, df])

    df_spb = df_spb.pivot(index=['country', 'adjustment_period'], columns='scenario', values='spbstar').reset_index()
    df_spb['binding_dsa'] = df_spb[['main_adjustment','adverse_r_g','lower_spb','financial_stress','stochastic']].max(axis=1)
    df_spb['binding_safeguard'] = df_spb[['debt_safeguard', 'deficit_resilience']].max(axis=1)
    df_spb.rename(columns={'main_adjustment_deficit_reduction': 'deficit_reduction'}, inplace=True)
    var_order = ['country',
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
    df_spb = df_spb[var_order].sort_values(['adjustment_period', 'country']).round(3)
    df_spb.to_excel(f'{output_path}/dsa_results_spb_{today}.xlsx', index=False)

def _save_dfs(results_dict, output_path, today):
    """
    Saves dfs for each country, adjustment period and scenario in results_dict.
    """
    with pd.ExcelWriter(f'{output_path}/dsa_results_timeseries_{today}.xlsx') as writer:
        for country in results_dict.keys():
            for adjustment_period in results_dict[country].keys():
                for scenario in results_dict[country][adjustment_period]['dfs'].keys():
                    sheet_name = f'{country}_{adjustment_period}_{scenario}'[:31] # limit sheet_name length to 31 characters
                    df = results_dict[country][adjustment_period]['dfs'][scenario]
                    df.to_excel(writer, sheet_name=sheet_name)

def _save_ameco(results_dict, output_path, today):
    """
    Saves ameco 2024 data for easy access.
    """
    df_ameco = pd.DataFrame()
    for country in results_dict.keys():
        df = pd.read_excel(f'../data/InputData/ameco_projections.xlsx', sheet_name=country)
        df['country'] = country
        df = df.loc[(df['year'] == 2024), ['country', 'd', 'ngdp', 'fb', 'spb']] 
        df_ameco = pd.concat([df_ameco, df])
    df_ameco.to_excel(f'{output_path}/dsa_ameco_2024_{today}.xlsx', index=False)


################ Functions to plot charts ################

def plot_charts(country_code_dict, results_dict, output_path, today):
    """
    Plot charts for Annex
    a) Ageing costs, interest rate, growth
    b) Budget balance
    c) Debt simulations
    """
    annex_chart_dict = _calc_charts(results_dict)
    _create_charts(annex_chart_dict, output_path, country_code_dict, today)

def _calc_charts(results_dict):
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
                # Safe df for baseline binding path
                df = results_dict[country][adjustment_period]['dfs']['binding'].reset_index()
            except:
                continue

            # Set adjustment path parameters to binding path
            binding_parameter_dict = results_dict[country][adjustment_period]['binding_parameters']
            binding_spb_target = binding_parameter_dict['binding_spb_target']
            initial_adjustment_period = binding_parameter_dict['initial_adjustment_period']
            initial_adjustment_step = binding_parameter_dict['initial_adjustment_step']
            intermediate_adjustment_period = binding_parameter_dict['intermediate_adjustment_period']
            intermediate_adjustment_step = binding_parameter_dict['intermediate_adjustment_step']
            deficit_resilience_periods = binding_parameter_dict['deficit_resilience_periods']
            deficit_resilience_step = binding_parameter_dict['deficit_resilience_step']
            post_adjustment_periods = binding_parameter_dict['post_adjustment_periods']
            
            # Create df for chart a)
            df_interest_ageing_growth = df[['y', 'iir', 'ageing_cost', 'ng']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
            df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Ageing costs', 'Implicit interest rate', 'Nominal GDP growth']]
            annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2050]

            # Create df for chart b)
            df_debt_chart = df[['y', 'spb_bca', 'spb', 'pb', 'ob']].rename(columns={'y': 'year', 'spb_bca': 'Age-adjusted structural primary balance', 'spb': 'Structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
            df_debt_chart = df_debt_chart[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
            annex_chart_dict[country][adjustment_period]['df_debt_chart'] = df_debt_chart.set_index('year').loc[:2050]

            # Run fanchart for chart c)
            try:
                if country == 'BGR':
                    raise Exception('BGR has no viable fanchart because of restricted sample period')
                dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period)
                dsa.project(spb_target=binding_spb_target, 
                            initial_adjustment_period=initial_adjustment_period, 
                            initial_adjustment_step=initial_adjustment_step, 
                            intermediate_adjustment_period=intermediate_adjustment_period,
                            intermediate_adjustment_step=intermediate_adjustment_step,
                            deficit_resilience_periods=deficit_resilience_periods,
                            deficit_resilience_step=deficit_resilience_step
                            )                
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
    
def _create_charts(annex_chart_dict, output_path, eu_code_dict, today):
    """
    Plots and saves annex charts.
    """
    # Set color pallette
    sns.set_palette(sns.color_palette('tab10'))
    subtitle_size = 14 
    for country in annex_chart_dict.keys():
        for adjustment_period in annex_chart_dict[country].keys():
            
            try:
                # Make 1, 3 sublot with df_interest_ageing_growth, df_debt_chart, df_fanchart
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'{eu_code_dict[country]}: {adjustment_period}-year scenario', fontsize=16)

                # Plot df_interest_ageing_growth
                axs[0].set_title('Ageing costs, interest rate, growth', fontsize = subtitle_size)
                df_interest_ageing_growth = annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth']
                df_interest_ageing_growth.plot(ax=axs[0], secondary_y=['Implicit interest rate', 'Nominal GDP growth'], lw=3)
                axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                lines = axs[0].get_lines() + axs[0].right_ax.get_lines()
                axs[0].legend(lines, [l.get_label() for l in lines], loc='best') 

                # Plot df_debt_chart
                axs[1].set_title('Budget balance', fontsize = subtitle_size)
                df_debt_chart = annex_chart_dict[country][adjustment_period]['df_debt_chart']
                df_debt_chart.plot(ax=axs[1], lw=3)
                axs[1].legend(loc='best') 

                # Plot df_fanchart
                df_fanchart = annex_chart_dict[country][adjustment_period]['df_fanchart']
                axs[2].set_title('Debt simulations', fontsize = subtitle_size)
                axs[2].plot(df_fanchart.index, df_fanchart['baseline'], color='blue', marker='o', markerfacecolor='none', markersize=3, label='Deterministic scenario', lw=2)
                try:
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th percentile', alpha=0.8)
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th percentile', alpha=0.8)
                    axs[2].plot(df_fanchart.index, df_fanchart['p50'], color='black', alpha=0.8, label='Median', lw=2)
                except:
                    raise
                axs[2].legend(loc='best') 

                # Add grey fill for 10 year post adustment and delete x-axis label
                for i in range(3):
                    axs[i].axvspan(df_fanchart.index[2+adjustment_period], df_fanchart.index[12+adjustment_period], alpha=0.2, color='grey')
                    axs[i].axvline(df_fanchart.index[2], color='grey', linestyle='--', lw=1.5)
                    axs[i].set_xlabel('')
                        
                # Increase space between subplots and heading
                fig.subplots_adjust(top=0.87)

                # Export to jpeg
                plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{today}.jpeg', dpi=300, bbox_inches='tight')
                plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{today}.svg', format='svg', bbox_inches='tight')

                # If plots should not show up in notebook
                #plt.close()

            except:
                print(f'Error: {country}_{adjustment_period}')
                plt.close()