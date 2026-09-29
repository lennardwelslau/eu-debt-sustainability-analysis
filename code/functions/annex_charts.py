# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Annex Charts             #
# ========================================================================================= #
#
# plot_annex_charts plots, for each country, (a) ageing costs, interest rate and growth, (b) budget
# balances and (c) the debt fanchart for the binding scenario, using the results of
# GroupDsaModel.find_spb_binding. Charts are saved in output/<folder>/charts.
#
# Author: Lennard Welslau
# Updated: 2026-09-29
# ========================================================================================= #

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from data_pipeline import REPO_ROOT
from classes.ResultsTables import COUNTRY_NAMES

base_dir = REPO_ROOT.as_posix() + '/'  # repository root, independent of the working directory

def get_country_name(iso):
    """
    Convert ISO country code to country name.
    """
    return COUNTRY_NAMES[iso]


def plot_annex_charts(
        results_dict,
        folder,
        adjustment_period=4,
        save_svg=False,
        save_png=False,
        save_jpg=True,
        show=True,
        horizon=29
        ):
    """
    Plot charts for each country:
    a) Ageing costs, interest rate, growth
    b) Budget balance
    c) Debt simulations (fanchart if stochastic results are available)

    Parameters:
        results_dict: GroupDsaModel.results after find_spb_binding.
        folder (str): Charts are saved in output/<folder>/charts.
        show (bool): Show the charts (False only saves them, which keeps notebooks small).
        horizon (int): Number of years after the start year shown.
    """
    annex_chart_dict = {}

    # 1) Data of the binding scenario
    for country in results_dict.keys():
        df_dict = results_dict[country].get('df_dict', {})
        if 'binding' not in df_dict:
            print(f'{country}: no binding scenario, skipped')
            continue
        annex_chart_dict[country] = {}
        df = df_dict['binding'].reset_index()
        last_year = df['y'].iloc[0] + horizon

        # Create df for chart a)
        df_interest_ageing_growth = df[['y','ageing_cost', 'ng', 'iir']].rename(columns={'y': 'year', 'iir': 'Implicit interest rate', 'ng': 'Nominal GDP growth', 'ageing_cost':'Ageing costs'})
        df_interest_ageing_growth = df_interest_ageing_growth[['year', 'Implicit interest rate', 'Nominal GDP growth', 'Ageing costs']]
        annex_chart_dict[country]['df_interest_ageing_growth'] = df_interest_ageing_growth.set_index('year').loc[:last_year]

        # Create df for chart b)
        df_debt_chart = df[['y', 'spb_bca', 'spb', 'pb', 'ob']].rename(columns={'y': 'year', 'spb_bca': 'Age-adjusted structural primary balance', 'spb':'Structural primary balance', 'pb': 'Primary balance', 'ob': 'Overall balance'})
        df_debt_chart = df_debt_chart[['year', 'Age-adjusted structural primary balance', 'Structural primary balance', 'Primary balance', 'Overall balance']]
        annex_chart_dict[country]['df_debt_chart'] = df_debt_chart.set_index('year').loc[:last_year]

        # Fanchart for chart c), deterministic debt path only if there are no stochastic results
        if results_dict[country].get('df_fanchart') is not None:
            df_fanchart = results_dict[country]['df_fanchart']
        else:
            df_fanchart = pd.DataFrame(columns=['year', 'baseline', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90'])
            df_fanchart['year'] = df['y']
            df_fanchart['baseline'] = df['d']
        annex_chart_dict[country]['df_fanchart'] = df_fanchart.set_index('year').loc[:last_year]

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
                'Ageing costs, interest rate, growth',
                'Budget balance',
                'Debt simulations'
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
                axs[i].axvspan(df_fanchart.index[0], df_fanchart.index[adjustment_period], alpha=0.3, color='grey')
                axs[i].axvline(df_fanchart.index[0]+adjustment_period+10, color='black', ls='--', alpha=0.8, lw=1.5)

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

            # Add fanchart to plot if percentiles are available
            if df_fanchart['p50'].notna().any():
                axs[2].fill_between(df_fanchart.index, df_fanchart['p10'], df_fanchart['p90'], label='10th-90th pct', color=fanchart_palette[0], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p20'], df_fanchart['p80'], label='20th-80th pct', color=fanchart_palette[1], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p30'], df_fanchart['p70'], label='30th-70th pct', color=fanchart_palette[2], edgecolor='white')
                axs[2].fill_between(df_fanchart.index, df_fanchart['p40'], df_fanchart['p60'], label='40th-60th pct', color=fanchart_palette[3], edgecolor='white')
                axs[2].plot(df_fanchart.index, df_fanchart['p50'], label='Median', color='black', alpha=0.9, lw=2.5)

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
        if show:
            plt.show()
        else:
            plt.close(fig)

