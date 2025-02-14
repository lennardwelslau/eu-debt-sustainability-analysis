# Import libraries and modules
import os
base_dir = '../' * (os.getcwd().split(os.sep)[::-1].index('code')+1)
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams.update({'axes.grid':True,'grid.color':'black','grid.alpha':'0.25','grid.linestyle':'--'})
plt.rcParams.update({'font.size': 14})
import seaborn as sns

# Import DSA model class and stochastic subclass
from classes import StochasticDsaModel as DSA

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
        folder, 
        adjustment_period=4,
        save_svg=False, 
        save_png=False,
        save_jpg=True
        ):
    """
    Plot charts for Annex
    a) Ageing costs, interest rate, growth
    b) Budget balance
    c) Debt simulations
    """
    # Initialize fig2_chart_dict
    annex_chart_dict = {}

    # 1) get df of binding scenario
    for country in results_dict.keys():
        annex_chart_dict[country] = {}
        annex_chart_dict[country] = {}
        df_dict = results_dict[country]['df_dict']
        
        # Safe df for baseline binding path
        try:
            df = df_dict['binding'].reset_index()
        except:
            continue

        # Create df for chart a)
        df_interest_ageing_growth = df[['y','ageing_cost', 'ng', 'iir']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
        df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Implicit interest rate', 'Nominal GDP growth', 'Ageing costs']]
        annex_chart_dict[country]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:2053]

        # Create df for chart b)
        df_debt_chart = df[['y', 'spb_bca', 'spb', 'pb', 'ob']].rename(columns={'y': 'year', 'spb_bca': 'Age-adjusted structural primary balance', 'spb':'Structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
        df_debt_chart = df_debt_chart[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
        annex_chart_dict[country]['df_debt_chart'] = df_debt_chart.set_index('year').loc[:2053]

        # Run fanchart for chart c)
        try:
            df_fanchart = results_dict[country]['df_fanchart']
            annex_chart_dict[country]['df_fanchart'] = df_fanchart.set_index('year').loc[:2053]
        except:
            df_fanchart = pd.DataFrame(columns=['year', 'baseline', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90'])
            df_fanchart['year'] = df['y']
            df_fanchart['baseline'] = df['d']
            annex_chart_dict[country]['df_fanchart'] = df_fanchart.set_index('year').loc[:2053]
    
    # 2) Plot charts
    sns.set_palette(sns.color_palette('tab10'))
    tab10_palette = sns.color_palette('tab10')
    fanchart_palette = sns.color_palette('Blues')

    # Loop over countries
    for country in annex_chart_dict.keys():
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'{get_country_name(country)}', fontsize=18)

        try:
            # Set subplot titles based on the adjustment period
            titles = [
                f'Ageing costs, interest rate, growth',
                f'Budget balance',
                f'Debt simulations'
            ]
            for col in range(3):
                axs[col].set_title(titles[col], fontsize=14)

            # Plot df_interest_ageing_growth
            df_interest_ageing_growth = annex_chart_dict[country]['df_interest_ageing_growth']
            df_interest_ageing_growth.plot(ax=axs[0], lw=2.5, alpha=0.9, secondary_y=['Implicit interest rate', 'Nominal GDP growth'])
            lines = axs[0].get_lines() + axs[0].right_ax.get_lines()
            axs[0].legend(lines, [l.get_label() for l in lines], loc='best', fontsize=10)
            
            # Plot df_debt_chart
            df_debt_chart = annex_chart_dict[country]['df_debt_chart']
            df_debt_chart.plot(lw=2.5, ax=axs[1])
            axs[1].legend(loc='best', fontsize=10)

            # Plot df_fanchart
            df_fanchart = annex_chart_dict[country]['df_fanchart']
            
            for i in range(3):
                # Add grey fill for adjustment period
                axs[i].axvspan(df_fanchart.index[1], df_fanchart.index[1+adjustment_period], alpha=0.3, color='grey')
                axs[i].axvline(df_fanchart.index[1]+adjustment_period+10, color='black', ls='--', alpha=0.8, lw=1.5)

                # Set labels and ticks
                axs[i].set_xlabel('')
                axs[i].tick_params(axis='both', which='major', labelsize=12)
                axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                # Check if there are duplicates in the first digits
                first_digits = [np.floor(tick) for tick in axs[i].get_yticks()]
                if len(first_digits) != len(set(first_digits)):
                    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                else:
                    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                if i == 0:
                    axs[i].right_ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    axs[i].right_ax.tick_params(axis='y', labelsize=12)
                if i == 2:
                    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    
            # Add fanchart to plot
            try:
                axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th pct', color=fanchart_palette[0], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th pct', color=fanchart_palette[1], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th pct', color=fanchart_palette[2], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th pct', color=fanchart_palette[3], edgecolor='white')
                axs[2].plot(df_fanchart.index, df_fanchart['p50'], label='Median', color='black', alpha=0.9, lw=2.5)
            except:
                pass

            axs[2].plot(df_fanchart.index, df_fanchart['baseline'], color=tab10_palette[3], ls='dashed', lw=2.5, alpha=0.9, label='Deterministic')
            axs[2].legend(loc='best', fontsize=10)

        except Exception as e:
            print(f'Error: {country}: {e}')
            raise

        # Increase space between subplots and heading
        fig.subplots_adjust(top=0.85)

        # Export charts
        if not os.path.exists(f'{base_dir}output/{folder}/charts'):
            os.makedirs(f'{base_dir}output/{folder}/charts')
        if save_svg == True: plt.savefig(f'{base_dir}output/{folder}/charts/{get_country_name(country)}.svg', format='svg', bbox_inches='tight')
        if save_png == True: plt.savefig(f'{base_dir}output/{folder}/charts/{get_country_name(country)}.png', dpi=300, bbox_inches='tight')
        if save_jpg == True: plt.savefig(f'{base_dir}output/{folder}/charts/{get_country_name(country)}.jpeg', dpi=300, bbox_inches='tight')


def plot_inv(countries, results_dict, folder, nrows=4, ncols=3, save_svg=False, save_png=True):
    """
    Plots investment shock counterfactual scenario plots given paths of baseline and investment case
    """
    # Define countries to plot
    tab10_palette = sns.color_palette('tab10')

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    for i, country in enumerate(countries):
        
        # Load dataframes
        df_bl = results_dict[country]['df_dict']['binding'].loc[1:19][['spb', 'ob', 'd', 'ngdp']]
        df_inv = results_dict[country]['df_dict']['inv'].loc[1:19][['spb', 'ob', 'd', 'ngdp']]
        df = df_bl.merge(df_inv, left_index=True, right_index=True, suffixes=('_bl', '_inv')).reset_index()
        df['inv_space'] = df['spb_bl'] - df['spb_inv']
        df['inv_space_abs'] = df['inv_space']/100 * df['ngdp_inv']
    
        # Plot corresponding subplot
        row, col = divmod(i, ncols)
        
        # Plot the SPB variable
        axs[0,col].plot(df['y'], df['spb_bl'], color=tab10_palette[2], label='SPB (Baseline scenario)', markersize=4, fillstyle='none', lw=2.5)
        axs[0,col].plot(df['y'], df['spb_inv'], color=tab10_palette[2], ls='--', marker=None, markersize=4, fillstyle='none', label='SPB (Investment scenario)', lw=2.5)


        # Plot overall balance
        axs[0,col].plot(df['y'], df['ob_bl'], color=tab10_palette[1], label='Overall balance (Baseline scenario)', lw=2.5)
        axs[0,col].plot(df['y'], df['ob_inv'], color=tab10_palette[1], ls='--', marker=None, markersize=4, fillstyle='none', label='Overall balance (Investment scenario)', lw=2.5)

        # Plot debt variables on secondary axis
        axs2 = axs[col].twinx()
        axs2.plot(df['y'], df['d_bl'], color=tab10_palette[0], alpha=0.8, label='Debt (Baseline scenario, RHS)', lw=2.5)
        axs2.plot(df['y'], df['d_inv'], color=tab10_palette[0], alpha=0.8, ls='--', marker=None, markersize=4, fillstyle='none', label='Debt (Investment scenario, RHS)', lw=2.5)

        # plt vertical line at start and end of investment program and add text
        axs[0,col].axvline(x=2025, color='black', lw=1, alpha=0.8, label='Adjustment period')
        axs[0,col].axvline(x=2031, color='black', lw=1, alpha=0.8)
        axs[col].axvspan(2025, 2030, facecolor='grey', edgecolor='none', alpha=0.3, label='Investment period')
        axs[0,col].axvline(x=2041, color='black', lw=1, ls='--', alpha=0.8, label='10-year post adjustment')

        # Set labels
        axs[0,col].set_ylabel('Balance', fontsize=14)
        axs2.set_ylabel('Debt', fontsize=14)
        axs[0,col].set_title(get_country_name(country), fontsize=18)

        # turn of grid for axs2
        axs2.grid(False)

        # Use FormatStrFormatter to format tick labels with no decimal places
        axs[0,col].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[0,col].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        # set label size to 12
        axs[0,col].tick_params(axis='both', which='major', labelsize=14)
        axs2.tick_params(axis='both', which='major', labelsize=14)

        # Extract legend handles and labels from subplots
        handles, labels = axs[0].get_legend_handles_labels()
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
            
    plt.tight_layout()

    if save_png == True: plt.savefig(f'{base_dir}output/{folder}/charts/inv_all.png', dpi=300, bbox_inches='tight')
    if save_svg == True: plt.savefig(f'{base_dir}output/{folder}/charts/inv_all.svg', format='svg', bbox_inches='tight')