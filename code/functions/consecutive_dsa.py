# Import libraries and modules
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.grid': True,
    'grid.color': 'black',
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'font.size': 14
})

# Import DSA model class and stochastic subclass
from classes import StochasticDsaModel as DSA

def run_consecutive_dsa(
        country,
        adjustment_start=2025, 
        initial_adjustment_period=4, 
        consecutive_adjustment_period=4, 
        number_of_adjustment_periods=3, 
        scenario_data=None,
        scenario=None,
        debt_safeguard=True,
        deficit_resilience=True,
        edp=True,
        print_results=False,
        plot_results=False
        ):
    """
    Performs DSA for consecutive adjustment periods and returns results in a DataFrame.
    """
    results = {}

    # Loop over consecutive adjustment periods
    for i in range(number_of_adjustment_periods):
        print_results = False if i < number_of_adjustment_periods else True

        adjustment_period = initial_adjustment_period + consecutive_adjustment_period * i
        model = DSA(country=country, adjustment_period=adjustment_period)

        # If demographic scenario data is provided, set ageing cost and potential GDP growth
        if scenario_data and scenario != 'baseline':
            model.ageing_cost = scenario_data[scenario]['cost'].loc[country].to_numpy()
            rg_pot_scenario = scenario_data[scenario]['gdp'].loc[country].to_numpy()
            for t, y in enumerate(range(model.adjustment_start, model.end_year + 1)):
                if t > 5:
                    model.rg_pot[t] = rg_pot_scenario[t]
                    model.rgdp_pot[t] = model.rgdp_pot[t - 1] * (1 + model.rg_pot[t] / 100) 
            model.rgdp_pot_bl = model.rgdp_pot.copy()
            model._project_gdp()

        # For first adjustment period, use initial adjustment period
        if i == 0:
            model.find_spb_binding(debt_safeguard=debt_safeguard, deficit_resilience=deficit_resilience, edp=edp, print_results=print_results, save_df=True)

        # For consecutive adjustment periods, use previous spb steps as pedefined steps
        else:
            model.predefined_spb_steps = spb_steps # np.concatenate([spb_steps, np.nan * np.ones(consecutive_adjustment_period)])
            model.find_spb_binding(debt_safeguard=debt_safeguard, deficit_resilience=deficit_resilience, edp=edp, print_results=print_results, save_df=True)
        
        spb_steps = model.spb_steps
        results[f'adjustment_period_{i+1}'] = model.spb_target_dict
        print(model.spb_target_dict)

    # Plot results if requested
    if plot_results:
        df = model.df().loc[:30].reset_index().set_index('y')
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

        model.plot_consecutive_model = plt.gcf()
        plt.show()

    return model