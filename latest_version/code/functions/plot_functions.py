# Import libraries and modules
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style('white')

# Import DSA model class and stochastic subclass
from classes import *

def get_country_name(iso):
    """
    Convert ISO country code to country name.
    """
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
    return country_code_dict[iso]

def plot_annex_charts(
        results_dict,
        folder_name, 
        save_svg=False, 
        save_png=True
        ):
    """
    Plot charts for Annex
    a) Ageing costs, interest rate, growth
    b) Budget balance
    c) Debt simulations
    """
    annex_chart_dict = _calc_annex_charts(results_dict)
    _create_annex_charts(annex_chart_dict, folder_name, save_svg, save_png)

def _calc_annex_charts(results_dict):
    """
    Calculates data for annex charts.
    """
    # Initialize fig2_chart_dict
    annex_chart_dict = {}

    # 1) get df of binding scenario
    for country in results_dict.keys():
        annex_chart_dict[country] = {}
        for adjustment_period in results_dict[country].keys():
            annex_chart_dict[country][adjustment_period] = {}
            df_dict = results_dict[country][adjustment_period]['df_dict']
            
            try:
                # Safe df for baseline binding path
                df = df_dict['binding'].reset_index()
            except:
                continue

            # Set adjustment path parameters to binding path
            binding_parameter_dict = results_dict[country][adjustment_period]['binding_parameter_dict']
            binding_spb_target = binding_parameter_dict['spb_target']
            binding_adjustment_steps = binding_parameter_dict['adjustment_steps']
            # binding_post_adjustment = binding_parameter_dict['post_adjustment_steps']

            # Create df for chart a)
            df_interest_ageing_growth = df[['y','ageing_cost', 'ng', 'iir']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
            df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Implicit interest rate', 'Nominal GDP growth', 'Ageing costs']]
            annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2053]

            # Create df for chart b)
            df_debt_chart = df[['y', 'spb_bca', 'spb', 'pb', 'ob']].rename(columns={'y': 'year', 'spb_bca': 'Age-adjusted structural primary balance', 'spb':'Structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
            df_debt_chart = df_debt_chart[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
            annex_chart_dict[country][adjustment_period]['df_debt_chart'] = df_debt_chart.set_index('year').loc[:2053]

            # Run fanchart for chart c)
            try:
                dsa = StochasticDsaModel(country=country, adjustment_period=adjustment_period)
                dsa.project(
                    adjustment_steps=binding_adjustment_steps, 
                    )            
                dsa.simulate()
                dsa.fanchart(show=False)
                df_fanchart = dsa.df_fanchart
                df_fanchart.loc[2+adjustment_period+6:, 'p10':'p90'] = np.nan
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2053]
            except:
                df_fanchart = pd.DataFrame(columns=['year', 'baseline', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90'])
                df_fanchart['year'] = df['y']
                df_fanchart['baseline'] = df['d']
                annex_chart_dict[country][adjustment_period]['df_fanchart'] = df_fanchart.set_index('year').loc[:2053]
        
    return annex_chart_dict
    
def _create_annex_charts(annex_chart_dict, folder_name, save_svg=False, save_png=True):
    """
    Plots and saves annex charts.
    """
    # Set color palette
    sns.set_palette(sns.color_palette('tab10'))
    tab10_palette = sns.color_palette('tab10')
    fanchart_palette = sns.color_palette('Blues')

    # Loop over countries
    for country in annex_chart_dict.keys():
        # Create a figure with two rows, one for each adjustment period
        fig, axs = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f'{get_country_name(country)}', fontsize=18)

        # Loop over adjustment periods
        for row, adjustment_period in enumerate(annex_chart_dict[country].keys()):
            print(f'Creating chart for {country}_{adjustment_period}')

            try:
                # Set subplot titles based on the adjustment period
                period_title_suffix = f'({adjustment_period}-year)'
                titles = [
                    f'Ageing costs, interest rate, growth {period_title_suffix}',
                    f'Budget balance {period_title_suffix}',
                    f'Debt simulations {period_title_suffix}'
                ]

                for col in range(3):
                    axs[row, col].set_title(titles[col], fontsize=14)

                # Plot df_interest_ageing_growth
                df_interest_ageing_growth = annex_chart_dict[country][adjustment_period]['df_interest_ageing_growth']
                df_interest_ageing_growth.plot(ax=axs[row, 0], lw=2.5, alpha=0.9, secondary_y=['Implicit interest rate', 'Nominal GDP growth'])
                lines = axs[row, 0].get_lines() + axs[row, 0].right_ax.get_lines()
                axs[row, 0].legend(lines, [l.get_label() for l in lines], loc='best', fontsize=10)
                

                # Plot df_debt_chart
                df_debt_chart = annex_chart_dict[country][adjustment_period]['df_debt_chart']
                df_debt_chart.plot(lw=2.5, ax=axs[row, 1])
                axs[row, 1].legend(loc='best', fontsize=10)

                # Plot df_fanchart
                df_fanchart = annex_chart_dict[country][adjustment_period]['df_fanchart']
                
                for i in range(3):
                    # Add grey fill for adjustment period
                    axs[row, i].axvspan(df_fanchart.index[1], df_fanchart.index[1+adjustment_period], alpha=0.3, color='grey')
                    axs[row, i].axvline(df_fanchart.index[1]+adjustment_period+10, color='black', ls='--', alpha=0.8, lw=1.5)

                    # Set labels and ticks
                    axs[row, i].set_xlabel('')
                    axs[row, i].tick_params(axis='both', which='major', labelsize=12)
                    axs[row, i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    # Check if there are duplicates in the first digits
                    first_digits = [np.floor(tick) for tick in axs[row, i].get_yticks()]
                    if len(first_digits) != len(set(first_digits)):
                        axs[row, i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    else:
                        axs[row, i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    if i == 0:
                        axs[row, i].right_ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                        axs[row, i].right_ax.tick_params(axis='y', labelsize=12)
                    if i == 2:
                        axs[row, i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                        
                # Add fanchart to plot
                try:
                    axs[row, 2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th pct', color=fanchart_palette[0], edgecolor='white')
                    axs[row, 2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th pct', color=fanchart_palette[1], edgecolor='white')
                    axs[row, 2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th pct', color=fanchart_palette[2], edgecolor='white')
                    axs[row, 2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th pct', color=fanchart_palette[3], edgecolor='white')
                    axs[row, 2].plot(df_fanchart.index, df_fanchart['p50'], label='Median', color='black', alpha=0.9, lw=2.5)
                except:
                    pass

                axs[row, 2].plot(df_fanchart.index, df_fanchart['baseline'], color=tab10_palette[3], ls='dashed', lw=2.5, alpha=0.9, label='Deterministic')
                axs[row, 2].legend(loc='best', fontsize=10)

            except Exception as e:
                print(f'Error: {country}_{adjustment_period}: {e}')

        # Increase space between subplots and heading
        fig.subplots_adjust(top=0.92)

        # Export charts
        if save_svg == True: plt.savefig(f'../output/{folder_name}/charts/{get_country_name(country)}.svg', format='svg', bbox_inches='tight')
        if save_png == True: plt.savefig(f'../output/{folder_name}/charts/{get_country_name(country)}.png', dpi=300, bbox_inches='tight')

def plot_inv(country_codes, results_dict, folder_name, nrows=4, ncols=3, save_svg=False, save_png=True):
    """
    Plots investment shock counterfactual scenario plots given paths of baseline and investment case
    """
    # Define countries to plot
    tab10_palette = sns.color_palette('tab10')

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    for i, country in enumerate(country_codes):
        
        # Load dataframes
        df_bl = results_dict[country][7]['df_dict']['binding'].loc[1:19][['spb', 'ob', 'd', 'ngdp']]
        df_inv = results_dict[country][7]['df_dict']['inv'].loc[1:19][['spb', 'ob', 'd', 'ngdp']]
        df = df_bl.merge(df_inv, left_index=True, right_index=True, suffixes=('_bl', '_inv')).reset_index()
        df['inv_space'] = df['spb_bl'] - df['spb_inv']
        df['inv_space_abs'] = df['inv_space']/100 * df['ngdp_inv']
    
        # Plot corresponding subplot
        row, col = divmod(i, ncols)
        
        # Plot the SPB variable
        axs[row,col].plot(df['y'], df['spb_bl'], color=tab10_palette[2], label='SPB (Baseline scenario)', markersize=4, fillstyle='none', lw=2.5)
        axs[row,col].plot(df['y'], df['spb_inv'], color=tab10_palette[2], ls='--', marker=None, markersize=4, fillstyle='none', label='SPB (Investment scenario)', lw=2.5)


        # Plot overall balance
        axs[row,col].plot(df['y'], df['ob_bl'], color=tab10_palette[1], label='Overall balance (Baseline scenario)', lw=2.5)
        axs[row,col].plot(df['y'], df['ob_inv'], color=tab10_palette[1], ls='--', marker=None, markersize=4, fillstyle='none', label='Overall balance (Investment scenario)', lw=2.5)

        # Plot debt variables on secondary axis
        axs2 = axs[row, col].twinx()
        axs2.plot(df['y'], df['d_bl'], color=tab10_palette[0], alpha=0.8, label='Debt (Baseline scenario, RHS)', lw=2.5)
        axs2.plot(df['y'], df['d_inv'], color=tab10_palette[0], alpha=0.8, ls='--', marker=None, markersize=4, fillstyle='none', label='Debt (Investment scenario, RHS)', lw=2.5)

        # plt vertical line at start and end of investment program and add text
        axs[row,col].axvline(x=2025, color='black', lw=1, alpha=0.8, label='Adjustment period')
        axs[row,col].axvline(x=2031, color='black', lw=1, alpha=0.8)
        axs[row, col].axvspan(2025, 2030, facecolor='grey', edgecolor='none', alpha=0.3, label='Investment period')
        axs[row,col].axvline(x=2041, color='black', lw=1, ls='--', alpha=0.8, label='10-year post adjustment')

        # Set labels
        axs[row,col].set_ylabel('Balance', fontsize=14)
        axs2.set_ylabel('Debt', fontsize=14)
        axs[row,col].set_title(get_country_name(country), fontsize=18)

        # turn of grid for axs2
        axs2.grid(False)

        # Use FormatStrFormatter to format tick labels with no decimal places
        axs[row,col].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[row,col].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        # set label size to 12
        axs[row,col].tick_params(axis='both', which='major', labelsize=14)
        axs2.tick_params(axis='both', which='major', labelsize=14)

        # Extract legend handles and labels from subplots
        handles, labels = axs[0, 0].get_legend_handles_labels()
        handles2, labels2 = axs2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

        # Remove duplicate legend entries
        handles_unique, labels_unique = [], []
        for handle, label in zip(handles, labels):
            if label not in labels_unique:
                handles_unique.append(handle)
                labels_unique.append(label)

    # Create a separate legend below the figure
    fig.legend(handles_unique, labels_unique, 
               loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, 
               frameon=False, fontsize=14)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_png == True: plt.savefig(f'../output/{folder_name}/charts/inv_all.png', dpi=300, bbox_inches='tight')
    if save_svg == True: plt.savefig(f'../output/{folder_name}/charts/inv_all.svg', format='svg', bbox_inches='tight')