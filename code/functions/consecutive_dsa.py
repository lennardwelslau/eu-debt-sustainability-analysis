# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Consecutive DSA          #
# ========================================================================================= #
#
# run_consecutive_dsa applies the EU fiscal rules to several consecutive adjustment periods: the
# SPB steps found for one period are kept as predefined steps when the rules are applied again at
# the end of that period, with the next adjustment period added.
#
# Author: Lennard Welslau
# Updated: 2026-09-29
# ========================================================================================= #

import matplotlib.pyplot as plt
from classes import StochasticDsaModel as DSA


def run_consecutive_dsa(
        country,
        initial_adjustment_period=4,
        consecutive_adjustment_period=4,
        number_of_adjustment_periods=3,
        scenario_data=None,
        scenario=None,
        debt_safeguard=True,
        deficit_resilience=True,
        edp=True,
        print_results=False,
        plot_results=False,
        **model_params
        ):
    """
    Apply the fiscal rules for consecutive adjustment periods and return the model of the last period.

    Parameters:
        country (str): ISO3 country code.
        initial_adjustment_period (int): Length of the first adjustment period (4 or 7 years).
        consecutive_adjustment_period (int): Length of each following adjustment period.
        number_of_adjustment_periods (int): Number of adjustment periods, including the first.
        scenario_data (dict): Optional demographic scenarios {scenario: {'cost': DataFrame, 'gdp': DataFrame}} with
            ageing costs and potential growth by country (rows) and year (columns).
        scenario (str): Scenario in scenario_data to apply ('baseline' uses the input data).
        debt_safeguard, deficit_resilience, edp (bool): Rules applied by find_spb_binding.
        print_results (bool): Display the results tables of the last adjustment period.
        plot_results (bool): Plot balances and debt over all adjustment periods.
        **model_params: Further arguments for StochasticDsaModel (e.g. input_file).

    The SPB targets of each adjustment period are stored in model.consecutive_spb_targets.
    """
    spb_targets = {}
    spb_steps = None

    for i in range(number_of_adjustment_periods):
        last = i == number_of_adjustment_periods - 1
        adjustment_period = initial_adjustment_period + consecutive_adjustment_period * i
        model = DSA(country=country, adjustment_period=adjustment_period, **model_params)

        # If demographic scenario data is provided, set ageing cost and potential GDP growth
        if scenario_data and scenario != 'baseline':
            model.ageing_cost = scenario_data[scenario]['cost'].loc[country].to_numpy()
            rg_pot_scenario = scenario_data[scenario]['gdp'].loc[country].to_numpy()
            for t in range(6, model.projection_period):
                model.rg_pot[t] = rg_pot_scenario[t]
                model.rgdp_pot[t] = model.rgdp_pot[t - 1] * (1 + model.rg_pot[t] / 100)
            model.rg_pot_bl = model.rg_pot.copy()
            model.rgdp_pot_bl = model.rgdp_pot.copy()
            model._project_gdp()

        # From the second adjustment period, the steps of the previous periods are kept
        if spb_steps is not None:
            model.predefined_spb_steps = spb_steps
        model.find_spb_binding(debt_safeguard=debt_safeguard, deficit_resilience=deficit_resilience, edp=edp,
                               print_results=print_results and last, save_df=True)

        # SPB target at the end of the adjustment period added in this application of the rules
        spb_steps = model.spb_steps
        period_start = model.adjustment_end_year - (initial_adjustment_period if i == 0 else consecutive_adjustment_period) + 1
        spb_targets[f'{period_start}-{model.adjustment_end_year}'] = model.spb_target_dict['binding']

    model.consecutive_spb_targets = spb_targets

    # Plot results if requested
    if plot_results:
        adjustment_start = model.adjustment_start_year
        df = model.df().loc[:30].reset_index().set_index('y')
        ax = df[['ob', 'sb', 'spb_bca']].plot(legend=False, lw=2)
        ax2 = df['d'].plot(secondary_y=True, legend=False, lw=2)

        # Shade the adjustment periods
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plt.axvspan(adjustment_start, adjustment_start + initial_adjustment_period, color=colors[0], alpha=0.1, label='adj. 1')
        for i in range(number_of_adjustment_periods - 1):
            start = adjustment_start + initial_adjustment_period + consecutive_adjustment_period * i
            end = start + consecutive_adjustment_period
            plt.axvspan(start, end, color=colors[(i + 1) % len(colors)], alpha=0.1, label=f'adj. {i + 2}')

        # Deficit thresholds of 3% and 1.5% of GDP
        ax.axhline(-3, color='black', linestyle='--', label='3%', alpha=0.5)
        ax.axhline(-1.5, color='black', linestyle='-.', label='1.5%', alpha=0.5)

        # Combined legend of both axes, underneath the plot
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.title(f'{country}: {initial_adjustment_period}-year, followed by '
                  f'{number_of_adjustment_periods - 1}x {consecutive_adjustment_period}-year adjustment')
        plt.legend(handles + handles2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

        model.plot_consecutive_model = plt.gcf()
        plt.show()

    return model
