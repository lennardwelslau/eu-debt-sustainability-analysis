# Import libraries and modules
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams.update({
    'axes.grid': True,
    'grid.color': 'black',
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'font.size': 14
})
base_dir = '../' * (os.getcwd().split(os.sep)[::-1].index('code')+1)

# Import DSA model class and stochastic subclass
from classes import StochasticDsaModel as DSA

def run_inv_scenario(
        countries, 
        results_dict,
        folder=None,
        adjustment_period=7,
        ):
    """
    Run DSA with temporaryinvestment shock scenario and save results in results_dict
    """
    # Loop over countries and adjustment periods
    for country in countries:

        # Load baseline adjustment steps
        spb_steps = np.copy(results_dict[country]['binding_parameter_dict']['spb_steps'])

        # First adjustment step is lowered by size of investment shock
        investment_shock = 0.5
        spb_steps[0] -= investment_shock

        # Create new instance of DSA model
        model = DSA(
            country=country, 
            adjustment_period=adjustment_period, 
            shock_frequency='quarterly'
            )
        
        # Set predefined adjustment steps, drop final one for reoptimization
        model.predefined_spb_steps = spb_steps[:-1]

        # Find binding spb, without safeguards since those are already included in the adjustment steps
        model.find_spb_binding(
            edp=False, 
            debt_safeguard=False, 
            deficit_resilience=False,
            print_results=False
        )

        # if model.spb_binding < spb_binding baseline, increase by 0.5 
        if model.spb_target < results_dict[country]['binding_parameter_dict']['spb_target']:
            model.spb_steps[-1] += investment_shock
            model.project(spb_steps=model.spb_steps)
        
        # if binding debt_safeguard, simply use old adjustment steps
        if results_dict[country]['binding_parameter_dict']['criterion'] == 'debt_safeguard':
            model.spb_target = results_dict[country]['binding_parameter_dict']['spb_target']
            model.project(spb_target=model.spb_target)

        # Safe df to results dict
        results_dict[country]['df_dict']['inv'] = model.df(all=True)

    # Save results dict    
    if folder:
        with open(f'../../output/{folder}/results_dict.pkl', 'wb') as f:
            pickle.dump(results_dict, f)



def plot_inv(countries, results_dict, folder, nrows=4, ncols=3, save_svg=False, save_png=False, save_jpg=False):
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
        axs[row,col].plot(df['y'], df['spb_bl'], color=tab10_palette[2], label='SPB (Baseline scenario)', markersize=4, fillstyle='none', lw=2.5)
        axs[row,col].plot(df['y'], df['spb_inv'], color=tab10_palette[2], ls='--', marker=None, markersize=4, fillstyle='none', label='SPB (Investment scenario)', lw=2.5)


        # Plot overall balance
        axs[row,col].plot(df['y'], df['ob_bl'], color=tab10_palette[1], label='Overall balance (Baseline scenario)', lw=2.5)
        axs[row,col].plot(df['y'], df['ob_inv'], color=tab10_palette[1], ls='--', marker=None, markersize=4, fillstyle='none', label='Overall balance (Investment scenario)', lw=2.5)

        # Plot debt variables on secondary axis
        axs2 = axs[row,col].twinx()
        axs2.plot(df['y'], df['d_bl'], color=tab10_palette[0], alpha=0.8, label='Debt (Baseline scenario, RHS)', lw=2.5)
        axs2.plot(df['y'], df['d_inv'], color=tab10_palette[0], alpha=0.8, ls='--', marker=None, markersize=4, fillstyle='none', label='Debt (Investment scenario, RHS)', lw=2.5)

        # plt vertical line at start and end of investment program and add text
        axs[row,col].axvline(x=2025, color='black', lw=1, alpha=0.8, label='Adjustment period')
        axs[row,col].axvline(x=2031, color='black', lw=1, alpha=0.8)
        axs[row,col].axvspan(2025, 2030, facecolor='grey', edgecolor='none', alpha=0.3, label='Investment period')
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
        handles, labels = axs[row,col].get_legend_handles_labels()
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
    if save_jpg == True: plt.savefig(f'{base_dir}output/{folder}/charts/inv_all.jpeg', dpi=300, bbox_inches='tight')