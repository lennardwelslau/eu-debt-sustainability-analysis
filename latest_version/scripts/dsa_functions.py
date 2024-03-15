# Functions accompanying the Jupyter Notebook main for the reproduction of the 

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

# # Load Conduit font and set as default
# import matplotlib.font_manager as fm
# prop = fm.FontProperties(fname='Conduit ITC Regular.otf', size=12)

# Import DSA model class and stochastic subclass
from DsaModelClass import DsaModel
from StochasticDsaModelClass import StochasticDsaModel

# Define main loop for DSA
def run_dsa(
        country_codes, 
        adjustment_periods, 
        results_dict, 
        output_path, 
        today,
        file_note, 
        edp=True, 
        debt_safeguard=True, 
        deficit_resilience=True, 
        inv_shock=False, 
        inv_period=None,
        inv_exception=False):
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
        
        for adjustment_period in adjustment_periods:
            dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period, inv_shock=inv_shock, inv_period=inv_period, inv_exception=inv_exception)
            # if country == 'FIN' and adjustment_period == 7:
            #     dsa.find_spb_binding(save_df=True, edp=edp, debt_safeguard=False,  deficit_resilience=deficit_resilience)
            # else:
            dsa.find_spb_binding(save_df=True, edp=edp, debt_safeguard=debt_safeguard,  deficit_resilience=deficit_resilience)
            results_dict[country][adjustment_period]['spb_targets'] = dsa.spb_target_dict
            results_dict[country][adjustment_period]['dfs'] = dsa.df_dict
            results_dict[country][adjustment_period]['binding_parameters'] = dsa.binding_parameter_dict
    
    with open(f'{output_path}/results_dict_{today}_{file_note}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

# Save results to excel
def save_results(results_dict, output_path, file_note=''):
    """
    Saves DSA results to excel.
    """
    _save_spbs(results_dict, output_path, file_note)
    _save_dfs(results_dict, output_path, file_note)

def _save_spbs(results_dict, output_path, file_note):
    """
    Saves spb_targets for each country and adjustment period in results_dict.
    """    
    # Create df_spb from results_dict
    df_spb = pd.DataFrame()
    for country in results_dict.keys():
        for adjustment_period in results_dict[country].keys():
            spb_target_dict = results_dict[country][adjustment_period]['spb_targets']
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
    df_spb.to_excel(f'{output_path}/results_spb_{file_note}.xlsx', index=False)

def _save_dfs(results_dict, output_path, file_note):
    """
    Saves dfs for each country, adjustment period and scenario in results_dict.
    """
    with pd.ExcelWriter(f'{output_path}/results_timeseries_{file_note}.xlsx') as writer:
        for country in results_dict.keys():
            for adjustment_period in results_dict[country].keys():
                for scenario in results_dict[country][adjustment_period]['dfs'].keys():
                    sheet_name = f'{country}_{adjustment_period}_{scenario}'[:31] # limit sheet_name length to 31 characters
                    df = results_dict[country][adjustment_period]['dfs'][scenario]
                    df.to_excel(writer, sheet_name=sheet_name)

def plot_annex_charts(country_code_dict, results_dict, output_path, file_note):
    """
    Plot charts for Annex
    a) Ageing costs, interest rate, growth
    b) Budget balance
    c) Debt simulations
    """
    annex_chart_dict = _calc_annex_charts(results_dict)
    _create_annex_charts(annex_chart_dict, output_path, country_code_dict, file_note)

def _calc_annex_charts(results_dict):
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
            binding_spb_target = binding_parameter_dict['spb_target']
            binding_adjustment_steps = binding_parameter_dict['adjustment_steps']
            binding_post_adjustment = binding_parameter_dict['post_adjustment_steps']

            # Create df for chart a)
            df_interest_ageing_growth = df[['y', 'iir', 'ageing_cost', 'ng']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
            df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Ageing costs', 'Implicit interest rate', 'Nominal GDP growth']]
            annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2050]

            # Create df for chart b)
            df_debt_chart = df[['y', 'spb_bca', 'spb', 'pb', 'ob']].rename(columns={'y': 'year', 'spb_bca': 'Age-adjusted structural primary balance', 'spb':'Structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
            df_debt_chart = df_debt_chart[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
            annex_chart_dict[country][adjustment_period]['df_debt_chart'] = df_debt_chart.set_index('year').loc[:2050]

            # Run fanchart for chart c)
            try:
                if country == 'BGR':
                    raise Exception('BGR has no viable fanchart because of restricted sample period')
                dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period)
                dsa.project(
                    adjustment_steps=binding_adjustment_steps, 
                    post_adjustment_steps=binding_post_adjustment
                    )            
                dsa.simulate()
                dsa.fanchart(show=False)
                df_fanchart = dsa.df_fanchart
                df_fanchart.loc[2+adjustment_period+6:, 'p10':'p90'] = np.nan
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2050]
            except:
                df_fanchart = pd.DataFrame(columns=['year', 'baseline', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90'])
                df_fanchart['year'] = df['y']
                df_fanchart['baseline'] = df['d']
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2050]
    
    return annex_chart_dict
    
def _create_annex_charts(annex_chart_dict, output_path, eu_code_dict, file_note):
    """
    Plots and saves annex charts.
    """
    # Set color pallette
    sns.set_palette(sns.color_palette('tab10'))
    tab10_palette = sns.color_palette('tab10')
    fanchart_palette = sns.color_palette('Blues')
    subtitle_size = 14 
    for country in annex_chart_dict.keys():
        for adjustment_period in annex_chart_dict[country].keys():
            
            try:
                # Make 1, 3 sublot with df_interest_ageing_growth, df_debt_chart, df_fanchart
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'{eu_code_dict[country]}: {adjustment_period}-year scenario', fontsize=16) # , fontproperties=prop)

                # Plot df_interest_ageing_growth
                axs[0].set_title('Ageing costs, interest rate, growth', fontsize = subtitle_size) # , fontproperties=prop)
                df_interest_ageing_growth = annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth']
                df_interest_ageing_growth.plot(ax=axs[0], lw=2.5, alpha=0.9, secondary_y=['Implicit interest rate', 'Nominal GDP growth'])
                axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                lines = axs[0].get_lines() + axs[0].right_ax.get_lines()
                axs[0].legend(lines, [l.get_label() for l in lines], loc='best') # , prop=prop)

                # Plot df_debt_chart
                axs[1].set_title('Budget balance', fontsize = subtitle_size) # , fontproperties=prop)
                df_debt_chart = annex_chart_dict[country][adjustment_period]['df_debt_chart']
                df_debt_chart.plot(lw=2.5, ax=axs[1])
                axs[1].legend(loc='best') #, prop=prop)

                # Plot df_fanchart
                df_fanchart = annex_chart_dict[country][adjustment_period]['df_fanchart']
                axs[2].set_title('Debt simulations', fontsize = subtitle_size) # , fontproperties=prop)
                
                # add grey fill for 10 year post adustment and delete x-axis label
                for i in range(3):
                    if i < 2: axs[i].axvspan(df_fanchart.index[2+adjustment_period], df_fanchart.index[12+adjustment_period], alpha=0.2, color='darkgrey')
                    axs[i].axvline(df_fanchart.index[2], color='grey', linestyle='--', lw=1)
                    axs[i].axvline(df_fanchart.index[2]+adjustment_period, color='grey', linestyle='--', lw=1)

                    axs[i].set_xlabel('')
                try:
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th percentile', color=fanchart_palette[0], edgecolor='white')
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th percentile', color=fanchart_palette[1], edgecolor='white')
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th percentile', color=fanchart_palette[2], edgecolor='white')
                    axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th percentile', color=fanchart_palette[3], edgecolor='white')
                    axs[2].plot(df_fanchart.index, df_fanchart['p50'], label='Median', color='black', alpha=0.9, lw=2.5)
                except:
                    pass
                axs[2].plot(df_fanchart.index, df_fanchart['baseline'], color=tab10_palette[3], ls='dashed', lw=2.5, alpha=0.9, label='Deterministic scenario')

                axs[2].legend(loc='best') # , prop=prop)

  
                        
                # for i in range(3):
                #     tick_labels = axs[i].get_xticklabels()
                #     tick_labels.extend(axs[i].get_yticklabels())
                #     for tick in tick_labels:
                #         tick.set_fontproperties(prop)

                # Increase space between subplots and heading
                fig.subplots_adjust(top=0.87)

                # Export to jpeg
                plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{file_note}.jpeg', dpi=300, bbox_inches='tight')
                # plt.savefig(f'{output_path}/results_charts/{country}_{adjustment_period}_{file_note}.svg', format='svg', bbox_inches='tight')

            except:
                print(f'Error: {country}_{adjustment_period}')
