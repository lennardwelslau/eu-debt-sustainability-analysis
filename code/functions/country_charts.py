# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Country Charts           #
# ========================================================================================= #
#
# plot_country_charts plots, for each country, (a) change in ageing costs, interest rate and growth, (b) budget
# balances and (c) the debt fanchart for the binding scenario, using the results of
# GroupDsaModel.find_spb_binding. Charts are saved in output/<folder>/charts.
#
# Author: Lennard Welslau
# Updated: 2026-10-01
# ========================================================================================= #

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd
from data_pipeline import REPO_ROOT
from classes.ResultsTables import COUNTRY_NAMES
from classes.StochasticDsaModelClass import plot_fan

base_dir = REPO_ROOT.as_posix() + '/'  # repository root, independent of the working directory

# Plain matplotlib style, as in the exercise notebooks
CHART_STYLE = {'axes.grid': True, 'grid.color': 'black', 'grid.alpha': 0.15, 'grid.linestyle': '--', 'font.size': 11,
               'axes.titlesize': 12, 'legend.fontsize': 10, 'legend.frameon': False}

MIN_AGEING_SPAN = 2  # minimum range of the ageing cost axis (pp. of GDP)


def get_country_name(iso):
    """
    Convert ISO country code to country name.
    """
    return COUNTRY_NAMES[iso]


def country_chart_data(results, horizon=29):
    """
    Data of the binding scenario for the country charts: dict of DataFrames indexed by year (growth, balances, debt).
    Returns None if the results have no binding scenario.
    """
    df_dict = results.get('df_dict', {})
    if 'binding' not in df_dict:
        return None
    df = df_dict['binding'].reset_index().set_index('y')
    last_year = df.index[0] + horizon
    growth = df[['ageing_cost', 'iir', 'ng']].rename(columns={
        'ageing_cost': 'Change in ageing costs since T (pp. of GDP, left axis)',
        'iir': 'Implicit interest rate (%, right axis)', 'ng': 'Nominal GDP growth (%, right axis)'})
    growth.iloc[:, 0] -= growth.iloc[0, 0]
    balances = df[['spb_bca', 'spb', 'pb', 'ob']].rename(columns={
        'spb_bca': 'Age-adjusted structural primary balance', 'spb': 'Structural primary balance',
        'pb': 'Primary balance', 'ob': 'Overall balance'})

    # Fanchart, deterministic debt path only if there are no stochastic results
    if results.get('df_fanchart') is not None:
        debt = results['df_fanchart'].set_index('year')
    else:
        debt = pd.DataFrame({'baseline': df['d']})
    return {'growth': growth.loc[:last_year], 'balances': balances.loc[:last_year], 'debt': debt.loc[:last_year]}


def _legend_below(ax, handles=None, labels=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)


def plot_country_chart(data, country, adjustment_period=4):
    """
    One country chart (three panels) from country_chart_data. Returns the figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(get_country_name(country), fontsize=16, fontweight='bold')
    start = data['debt'].index[0]

    # a) Ageing costs (left axis), interest rate and growth (right axis)
    ax, growth = axs[0], data['growth']
    ax.set_title('Ageing costs, interest rate, growth')
    ageing = growth.columns[0]
    ax.plot(growth.index, growth[ageing], color='C0', lw=2, label=ageing)
    ax.set_ylabel('pp. of GDP')
    low, high = min(growth[ageing].min(), 0), max(growth[ageing].max(), 0)
    pad = max(0, MIN_AGEING_SPAN - (high - low)) / 2  # minimum span, so that small changes look small
    ax.set_ylim(low - pad - 0.05 * MIN_AGEING_SPAN, high + pad + 0.05 * MIN_AGEING_SPAN)
    right = ax.twinx()
    for col, color in zip(growth.columns[1:], ['C1', 'C2']):
        right.plot(growth.index, growth[col], color=color, lw=2, label=col)
    right.set_ylabel('%')
    right.grid(False)
    handles = ax.get_lines() + right.get_lines()
    _legend_below(ax, handles, [h.get_label() for h in handles])

    # b) Budget balances
    ax = axs[1]
    ax.set_title('Budget balances')
    for col in data['balances'].columns:
        ax.plot(data['balances'].index, data['balances'][col], lw=2, label=col)
    ax.set_ylabel('% of GDP')
    _legend_below(ax)

    # c) Debt fanchart (style shared with StochasticDsaModel.fanchart)
    ax = axs[2]
    ax.set_title('Debt simulations')
    plot_fan(ax, data['debt'])
    ax.set_ylabel('% of GDP')
    _legend_below(ax)

    # Adjustment period and end of the 10 years after it (DSA criteria horizon)
    for ax in axs:
        ax.axvspan(start, start + adjustment_period, color='grey', alpha=0.2, lw=0)
        ax.axvline(start + adjustment_period + 10, color='black', ls=':', lw=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(handles=[Patch(color='grey', alpha=0.2, label='Adjustment period'),
                        Line2D([], [], color='black', ls=':', lw=1.5, label='10 years after the adjustment period')],
               loc='upper right', bbox_to_anchor=(1, 1), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_country_charts(
        results_dict,
        folder,
        adjustment_period=4,
        countries=None,
        save_svg=False,
        save_png=False,
        save_jpg=True,
        show=True,
        horizon=29
        ):
    """
    Plot and save one chart per country:
    a) Ageing costs, interest rate, growth
    b) Budget balances
    c) Debt simulations (fanchart if stochastic results are available)

    Parameters:
        results_dict: GroupDsaModel.results after find_spb_binding.
        folder (str): Charts are saved in output/<folder>/charts.
        countries (list): Countries to plot, None plots all countries in results_dict.
        show (bool): Show the charts (False only saves them, which keeps notebooks small).
        horizon (int): Number of years after the start year shown.
    """
    path = f'{base_dir}output/{folder}/charts'
    os.makedirs(path, exist_ok=True)
    with plt.rc_context(CHART_STYLE):
        for country in countries or results_dict.keys():
            data = country_chart_data(results_dict[country], horizon=horizon)
            if data is None:
                print(f'{country}: no binding scenario, skipped')
                continue
            fig = plot_country_chart(data, country, adjustment_period=adjustment_period)
            name = f'{path}/{get_country_name(country)}'
            if save_svg:
                fig.savefig(f'{name}.svg', format='svg', bbox_inches='tight')
            if save_png:
                fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
            if save_jpg:
                fig.savefig(f'{name}.jpeg', dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)
