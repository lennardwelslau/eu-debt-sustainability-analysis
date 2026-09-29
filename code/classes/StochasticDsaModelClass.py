# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Stochastic Sub Class     #
# ========================================================================================= #
#
# The StochasticDsaModel is a subclass of DsaModel for stochastic projections around the
# deterministic debt path, with shocks to exchange rates, interest rates, nominal GDP growth and the
# primary balance. It has five parts:
#
# 1. Simulation methods: draw quarterly or annual shocks (joint normal or VAR), aggregate them to
#    annual shocks, combine them with the baseline and simulate the debt ratio.
# 2. Auxiliary methods: fan charts and shock plots.
# 3. Stochastic optimization: SPB target such that debt declines (or stays below 60%) with a given
#    probability.
# 4. Integrated optimizer: find_spb_binding combines the DSA criteria, the EDP and the safeguards.
#    Default rules follow Darvas, Welslau and Zettelmeyer (2024); each rule can be switched to the
#    Commission prior guidance version (see BINDING_RULES and COMMISSION_RULES).
# 5. Numba functions that speed up the simulations.
#
# For comments and suggestions please contact lennard.welslau[at]gmail[dot]com
#
# Author: Lennard Welslau
# Updated: 2026-09-29
# ========================================================================================= #

# Import libraries and modules
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from statsmodels.tsa.api import VAR
from numba import jit
from classes import DsaModel
from classes.DsaModelClass import DEFAULT_INPUT_FILE
from data_pipeline import read_shocks
from classes.ResultsTables import display_tables, criterion_label, COUNTRY_NAMES

# Rules for find_spb_binding: defaults (Darvas, Welslau and Zettelmeyer, 2024) and Commission prior guidance
BINDING_RULES = {
    'dsa_criteria': 'default',
    'stochastic_criteria': ['debt_declines', 'debt_below_60'],
    'edp': 'default',
    'debt_safeguard': 'default',
    'deficit_resilience': 'default',
    'frontloading': True,
    'grid': None,
}
COMMISSION_RULES = {
    'dsa_criteria': 'commission',
    'stochastic_criteria': ['debt_declines'],
    'edp': 'commission',
    'debt_safeguard': 'commission',
    'deficit_resilience': 'commission',
    'frontloading': False,
    'grid': 0.01,
}


class StochasticDsaModel(DsaModel):

# ========================================================================================= #
#                               INITIALIZE SUBCLASS                                         #
# ========================================================================================= #
    def __init__(self,
                country, # ISO code of country
                start_year=None, # start year of projection, first year is baseline value; None uses input file reference year
                end_year=2070, # end year of projection
                adjustment_period=4, # number of years for linear spb_bca adjustment
                adjustment_start_year=None, # start year of linear spb_bca adjustment; None uses start_year + 1
                ageing_cost_period=10, # number of years for ageing cost adjustment
                shock_sample_start=2000, # start year of shock sample
                stochastic_start_year=None, # start year of stochastic projection
                stochastic_period=5, # number of years for stochastic projection
                shock_frequency='quarterly', # frequency of shock data
                winsorize_sample=True, # clip shocks to the 5th and 95th percentiles
                estimation='normal', # 'normal', 'var_cholesky' or 'var_bootstrap'
                fiscal_multiplier=None, # None uses input file value
                fiscal_multiplier_persistence=3, # years over which the multiplier effect on the output gap fades
                fiscal_multiplier_type=None, # 'pers' or 'ec', None uses input file
                bond_data=False, # Use bond level data for repayment profile
                input_file=DEFAULT_INPUT_FILE, # input workbook, file name in data/InputData or full path
                overrides=None, # user inputs on top of the input workbook: rows with CODE, VALUE and YEAR (None for parameters)
                ):

        # Initialize base class
        super().__init__(
            country=country,
            start_year=start_year,
            end_year=end_year,
            adjustment_period=adjustment_period,
            adjustment_start_year=adjustment_start_year,
            ageing_cost_period=ageing_cost_period,
            fiscal_multiplier=fiscal_multiplier,
            fiscal_multiplier_persistence=fiscal_multiplier_persistence,
            fiscal_multiplier_type=fiscal_multiplier_type,
            bond_data=bond_data,
            input_file=input_file,
            overrides=overrides,
            )

        # Set stochastic parameters
        self.shock_frequency = shock_frequency
        self.shock_sample_start = shock_sample_start
        self.estimation = estimation
        if stochastic_start_year is None:
            stochastic_start_year = self.adjustment_start_year + adjustment_period
        self.stochastic_start_year = stochastic_start_year
        self.stochastic_period = stochastic_period
        self.stochastic_start = stochastic_start_year - self.start_year
        self.stochastic_end = self.stochastic_start + stochastic_period - 1
        if shock_frequency == 'quarterly':
            self.draw_period = stochastic_period * 4
        elif shock_frequency == 'annual':
            self.draw_period = stochastic_period
        self.winsorize_sample = winsorize_sample

        # Get shock data
        self._get_shock_data()

    def _get_shock_data(self):
        """
        Get shock data from the input workbook (sheets Shocks_Q and Shocks_A) and adjust outliers.
        """
        # Read country shock data and get number of variables for quarterly data
        if self.shock_frequency == 'quarterly':
            self.df_shocks = read_shocks(self.input_file, self.country, 'Q')

            # If quarterly shock data is not available, set parameters to annual
            if self.df_shocks.empty:
                print(f'No quarterly shock data available for {self.country}, using annual data instead.')
                self.shock_frequency = 'annual'
                self.draw_period = self.stochastic_period

        # Read country shock data for annual data
        if self.shock_frequency == 'annual':
            self.df_shocks = read_shocks(self.input_file, self.country, 'A')

        # Subset shock period and order variables
        self.df_shocks = self.df_shocks.loc[self.df_shocks.index.astype(str).str[:4].astype(int) >= self.shock_sample_start]
        self.df_shocks = self.df_shocks[
            ['EXR_EUR', 'EXR_USD', 'INTEREST_RATE_ST', 'INTEREST_RATE_LT', 'NOMINAL_GDP_GROWTH', 'PRIMARY_BALANCE']
            ]

        # Get number of variables
        self.num_variables = self.df_shocks.shape[1]
        assert self.num_variables == 6, 'Unexpected number of shock variables!'

        # Adjust outliers by keeping only 95th to 5th percentile
        if self.winsorize_sample:
            self.df_shocks = self.df_shocks.clip(
                lower=self.df_shocks.quantile(0.05, axis=0),
                upper=self.df_shocks.quantile(0.95, axis=0),
                axis=1
                )

# ========================================================================================= #
#                               SIMULATION METHODS                                          #
# ========================================================================================= #

    def simulate(self, N=100000):
        """
        Simulate the stochastic model.
        """
        # Set number of simulations
        self.N = N

        # Draw shocks from a multivariate normal distribution or VAR model
        if self.estimation == 'normal':
            self._draw_shocks_normal()
        elif self.estimation in ['var_cholesky', 'var_bootstrap']:
            self._draw_shocks_var()

        # Aggregate quarterly shocks to annual shocks
        if self.shock_frequency == 'quarterly':
            self._aggregate_shocks_quarterly()
        elif self.shock_frequency == 'annual':
            self._aggregate_shocks_annual()

        # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()

    def _draw_shocks_normal(self):
        """
        Draw quarterly or annual shocks from a multivariate normal distribution.

        This method calculates the covariance matrix of the shock DataFrame and then draws N samples of quarterly shocks
        from a multivariate normal distribution with mean 0 and the calculated covariance matrix.

        It reshapes the shocks into a 4-dimensional array of shape (N, draw_period, num_variables), where N is the number of simulations,
        draw_period is the number of consecutive years or quarters drawn, and num_variables is the number of shock variables.
        """

        # Calculate the covariance matrix of the shock DataFrame
        self.cov_matrix = self.df_shocks.cov()

        # Draw samples of quarterly shocks from a multivariate normal distribution
        self.shocks_sim_draws = np.random.multivariate_normal(
            mean=np.zeros(self.cov_matrix.shape[0]),
            cov=self.cov_matrix,
            size=(self.N, self.draw_period)
            )

        # Set PB shocks during adjustment period to zero
        if not hasattr(self, 'stochastic_pb_adjustment'): # attribute can be set if adjustment is already included
            stochastic_within_adjustment = self.adjustment_end - self.stochastic_start + 1
            if self.shock_frequency == 'quarterly': stochastic_within_adjustment *= 4
            if stochastic_within_adjustment > 0:
                self.shocks_sim_draws[:, :stochastic_within_adjustment, -1] = 0

    def _draw_shocks_var(self):
        """
        Draw quarterly or annual shocks from a VAR model.

        This method estimates a VAR model on the shock DataFrame and then draws N samples of quarterly shocks from the residuals of the VAR
        model using a bootstrap method. It reshapes the shocks into a 4-dimensional array of shape (N, draw_period, num_variables) where N is the
        number of simulations, draw_period is the number of consecutive years or quarters drawn, and num_variables is the number of shock variables.
        """
        # Define sample for VAR model, exclude EUR exchange rate shocks for euro area countries and DNK
        ea_countries = ['AUT', 'BEL', 'BGR', 'DNK', 'HRV', 'CYP', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN', 'ESP']
        var_sample = self.df_shocks.copy()
        if self.country in ea_countries: var_sample.drop(columns=['EXR_EUR'], inplace=True)
        elif self.country == 'USA': var_sample.drop(columns=['EXR_USD'], inplace=True)

        # Estimate VAR(1) model
        self.var = VAR(var_sample).fit(1)

        # Extract parameters from the VAR results
        lags = self.var.k_ar
        intercept = self.var.params.iloc[0].values
        coefs = self.var.coefs
        residuals = self.var.resid.values

        # Use bootstrap sampling from the residuals or Cholesky decomposition of the covariance matrix
        if self.estimation == 'var_bootstrap':
            residual_draws = residuals[np.random.choice(len(residuals), size=(self.N, self.draw_period), replace=True)]
        if self.estimation == 'var_cholesky':
            cov_matrix = np.cov(residuals.T)
            chol_matrix = np.linalg.cholesky(cov_matrix)
            residual_draws = np.random.randn(self.N, self.draw_period, residuals.shape[1]) @ chol_matrix.T

        # Set PB shocks during adjustment period to zero
        if not hasattr(self, 'stochastic_pb_adjustment'): # attribute can be set if adjustment is already included
            stochastic_within_adjustment = self.adjustment_end - self.stochastic_start + 1
            if self.shock_frequency == 'quarterly': stochastic_within_adjustment *= 4
            if stochastic_within_adjustment > 0:
                residual_draws[:, :stochastic_within_adjustment, -1] = 0
        else:
            stochastic_within_adjustment = 0 # set to zero if adjustment is already included

        # Simulate shocks using numba
        self.shocks_sim_draws = np.zeros_like(residual_draws)
        construct_var_shocks(
            N=self.N,
            draw_period=self.draw_period,
            shocks_sim_draws=self.shocks_sim_draws,
            lags=lags,
            intercept=intercept,
            coefs=coefs,
            residual_draws=residual_draws,
            stochastic_within_adjustment=stochastic_within_adjustment
            )

        # Add zero exchange rate shock if it was removed before
        if self.country in ea_countries:
            exr_eur_shock = np.zeros((self.N, self.draw_period, 1))
            self.shocks_sim_draws = np.concatenate((exr_eur_shock, self.shocks_sim_draws), axis=2)
        elif self.country == 'USA':
            exr_usd_shock = np.zeros((self.N, self.draw_period, 1))
            self.shocks_sim_draws = np.concatenate((self.shocks_sim_draws[:,:,:1], exr_usd_shock, self.shocks_sim_draws[:,:,1:]), axis=2)

    def _aggregate_shocks_quarterly(self):
        """
        Aggregate quarterly shocks to annual shocks for specific variables.

        This method aggregates the shocks for exchange rate, short-term interest rate, nominal GDP growth, and primary balance
        from quarterly to annual shocks as the sum over four quarters. For long-term interest rate, it aggregates shocks over all past quarters up to the current year and avg_res_mat.
        """
        # Reshape the shocks to sum over the four quarters
        self.shocks_sim_grouped = self.shocks_sim_draws.reshape((self.N, self.stochastic_period, 4, self.num_variables))

        # Aggregate shocks for exchange rates, short-term interest rate, nominal GDP growth, and primary balance
        exr_eur_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -6], axis=2)
        exr_usd_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -5], axis=2)
        short_term_interest_rate_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -4], axis=2)
        nominal_gdp_growth_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -2], axis=2)
        primary_balance_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -1], axis=2)

        # Aggregate shocks for long-term interest rate over the average residual maturity (in quarters)
        maturity_quarters = int(np.round(self.avg_res_mat * 4))

        # Initialize an array to store the aggregated shocks for the long-term interest rate
        self.long_term_interest_rate_shocks = np.zeros((self.N, self.stochastic_period))

        # Iterate over each year
        for t in range(1, self.stochastic_period+1):
            q = t * 4

            # Calculate the weight for each quarter based on the current year and average residual maturity
            weight = np.min([self.avg_res_mat, t]) / self.avg_res_mat

            # Determine the number of quarters to sum based on the current year and average residual maturity
            q_to_sum = np.min([q, maturity_quarters])

            # Sum the shocks (N, T, num_quarters, num_variables) across the selected quarters
            aggregated_shocks = weight * np.sum(
                self.shocks_sim_draws[:, q - q_to_sum : q, -3], axis=(1)
                )

            # Assign the aggregated shocks to the corresponding year
            self.long_term_interest_rate_shocks[:, t-1] = aggregated_shocks

        # Calculate the weighted average of short and long-term interest using D_share_st
        interest_rate_shocks = self.D_share_st * short_term_interest_rate_shocks + self.D_share_lt * self.long_term_interest_rate_shocks

        # Stack all shocks in a matrix
        self.shocks_sim = np.stack([exr_eur_shocks, exr_usd_shocks, interest_rate_shocks, nominal_gdp_growth_shocks, primary_balance_shocks], axis=2)
        self.shocks_sim = np.transpose(self.shocks_sim, (0, 2, 1))

    def _aggregate_shocks_annual(self):
        """
        Save annual into shock matrix, aggregate long term interest rate shocks.
        """
        # Reshape the shocks
        self.shocks_sim_grouped = self.shocks_sim_draws.reshape((self.N, self.stochastic_period, self.num_variables))

        # Shocks for exchange rates, short-term interest rate, nominal GDP growth, and primary balance
        exr_eur_shocks = self.shocks_sim_grouped[:, :, -6]
        exr_usd_shocks = self.shocks_sim_grouped[:, :, -5]
        short_term_interest_rate_shocks = self.shocks_sim_grouped[:, :, -4]
        nominal_gdp_growth_shocks = self.shocks_sim_grouped[:, :, -2]
        primary_balance_shocks = self.shocks_sim_grouped[:, :, -1]

        # Aggregate shocks for long-term interest rate over the average residual maturity (in years)
        maturity_years = int(np.round(self.avg_res_mat))

        # Initialize an array to store the aggregated shocks for the long-term interest rate
        self.long_term_interest_rate_shocks = np.zeros((self.N, self.stochastic_period))

        # Iterate over each year
        for t in range(1, self.stochastic_period+1):

            # Calculate the weight for each year based on the current year and avg_res_mat
            weight = np.min([self.avg_res_mat, t]) / self.avg_res_mat

            # Determine the number of years to sum based on the current year and avg_res_mat
            t_to_sum = np.min([t, maturity_years])

            # Sum the shocks (N, T, num_variables) across the selected years
            aggregated_shocks = weight * np.sum(
                self.shocks_sim_draws[:, t - t_to_sum : t, -3], axis=(1))

            # Assign the aggregated shocks to the corresponding year
            self.long_term_interest_rate_shocks[:, t-1] = aggregated_shocks

        # Calculate the weighted average of short and long-term interest using D_share_st
        interest_rate_shocks = self.D_share_st * short_term_interest_rate_shocks + self.D_share_lt * self.long_term_interest_rate_shocks

        # Stack all shocks in a matrix
        self.shocks_sim = np.stack([exr_eur_shocks, exr_usd_shocks, interest_rate_shocks, nominal_gdp_growth_shocks, primary_balance_shocks], axis=2)
        self.shocks_sim = np.transpose(self.shocks_sim, (0, 2, 1))

    def _combine_shocks_baseline(self):
        """
        Combine shocks with the respective baseline variables and set starting values for simulation.
        """
        # Create arrays to store the simulated variables
        d_sim = np.zeros([self.N, self.stochastic_period+1])  # Debt to GDP ratio
        exr_eur_sim = np.zeros([self.N, self.stochastic_period+1])  # EUR exchange rate
        exr_usd_sim = np.zeros([self.N, self.stochastic_period+1])  # USD exchange rate
        iir_sim = np.zeros([self.N, self.stochastic_period+1])  # Implicit interest rate
        ng_sim = np.zeros([self.N, self.stochastic_period+1])  # Nominal GDP growth
        pb_sim = np.zeros([self.N, self.stochastic_period+1])  # Primary balance
        sf_sim = np.zeros([self.N, self.stochastic_period+1])  # Stock flow adjustment

        # Call the Numba JIT function with converted self variables
        combine_shocks_baseline_jit(
            N=self.N,
            stochastic_start=self.stochastic_start,
            stochastic_end=self.stochastic_end,
            shocks_sim=self.shocks_sim,
            exr_eur=self.exr_eur,
            exr_usd=self.exr_usd,
            iir=self.iir,
            ng=self.ng,
            pb=self.pb,
            sf=self.sf,
            d=self.d,
            d_sim=d_sim,
            exr_eur_sim=exr_eur_sim,
            exr_usd_sim=exr_usd_sim,
            iir_sim=iir_sim,
            ng_sim=ng_sim,
            pb_sim=pb_sim,
            sf_sim=sf_sim
            )

        # Store simulated variables
        self.d_sim = d_sim
        self.exr_eur_sim = exr_eur_sim
        self.exr_usd_sim = exr_usd_sim
        self.iir_sim = iir_sim
        self.ng_sim = ng_sim
        self.pb_sim = pb_sim
        self.sf_sim = sf_sim

    def _simulate_debt(self):
        """
        Simulate the debt-to-GDP ratio using the baseline variables and the shocks.
        """
        # Call the Numba JIT function with converted self variables and d_sim as an argument
        simulate_debt_jit(
            N=self.N,
            stochastic_period=self.stochastic_period,
            D_share_domestic=self.D_share_domestic,
            D_share_eur=self.D_share_eur,
            D_share_usd=self.D_share_usd,
            d_sim=self.d_sim,
            iir_sim=self.iir_sim,
            ng_sim=self.ng_sim,
            exr_eur_sim=self.exr_eur_sim,
            exr_usd_sim=self.exr_usd_sim,
            pb_sim=self.pb_sim,
            sf_sim=self.sf_sim)

        # Set negative debt-to-GDP ratios to zero
        self.d_sim = np.where(self.d_sim < 0, 0, self.d_sim)

# ========================================================================================= #
#                                AUXILIARY METHODS                                          #
# ========================================================================================= #

    def fanchart(self, var='d', plot=True, save_as=None, xlim=None, ylim=None, pct_line=False, figsize=(10, 6)):
        """
        Create a fanchart for the debt-to-GDP ratio or other variables. Percentiles are stored in df_fanchart.
        save_as: optional file path for the plot (e.g. '../output/fanchart.png').
        """
        # Default x-axis from start year to 16 years ahead
        if xlim is None:
            xlim = (self.start_year, self.start_year + 16)

        # Simulate if there is no simulation for the current projection (first values of baseline and simulation differ)
        bl_var = getattr(self, f'{var}')
        sim_var = getattr(self, f'{var}_sim', None)
        if sim_var is None or not np.isclose(sim_var[0, 0], bl_var[self.stochastic_start-1]):
            self.simulate()
            sim_var = getattr(self, f'{var}_sim')

        # Calculate the percentiles
        self.pcts_dict = {}
        for pct in np.arange(10, 100, 10):
            self.pcts_dict[pct] = np.percentile(sim_var, pct, axis=0)[:self.stochastic_period+1]

        # Create array of years and baseline debt-to-GDP ratio
        years = np.arange(self.start_year, self.end_year+1)

        # Plot the results using fill between if plot is True
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(years, bl_var, ls='--', lw=3, color='red', label='Baseline', zorder=3)
            ax.plot(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[50], alpha=1, ls='-', lw=3, color='black', label='Median', zorder=2)
            if pct_line:
                if not hasattr(self, 'prob_target'):
                    self.prob_target = 0.7
                pct = int((self.prob_target) * 100)
                ax.plot(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[pct], color='darkgreen', linestyle=(0,(1,1)), lw=3.5, label=f'{pct} pct', zorder=1)
            ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[40], self.pcts_dict[60], color='dodgerblue', edgecolor='none', lw=1, alpha=0.9, label='40-60 pct', zorder=0)
            ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[30], self.pcts_dict[70], color='dodgerblue', edgecolor='none', lw=1, alpha=0.5, label='30-70 pct', zorder=0)
            ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[20], self.pcts_dict[80], color='dodgerblue', edgecolor='none', lw=1, alpha=0.3, label='20-80 pct', zorder=0)
            ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[10], self.pcts_dict[90], color='dodgerblue', edgecolor='none', lw=1, alpha=0.15, label='10-90 pct', zorder=0)

            # Plot layout
            ax.legend(loc='best')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.xaxis.grid(False)
            ylabel = 'Debt (percent of GDP)' if var == 'd' else var
            ax.set_ylabel(ylabel)
            ax.set_title(f'{self.stochastic_period}-year fanchart for {self.country} (adjustment {self.adjustment_start_year}-{self.adjustment_end_year})')
            ax.set_xlim(xlim[0], xlim[1])
            if ylim:
                ax.set_ylim(ylim[0], ylim[1])
            plt.tight_layout()
            if save_as:
                plt.savefig(save_as, dpi=300, bbox_inches='tight')
            plt.show()

        # Save fanchart data in a dataframe
        self.df_fanchart = pd.DataFrame({'year': years, 'baseline': bl_var})
        for pct in self.pcts_dict:
            df_pct = pd.DataFrame({'year': years[self.stochastic_start-1:self.stochastic_end+1], f'p{pct}':
            self.pcts_dict[pct]})
            self.df_fanchart = self.df_fanchart.merge(df_pct, on='year', how='left')

    def plot_shocks(self, hist=False, percentiles=False, sim=False, figsize=(15, 10)):
        """
        Create a figure with subplots for each shock variable as a line plot or histogram.
        Shocks are based on historical data or simulation. If 'percentiles' is True, the function
        adds quantile lines: vertical for histograms and horizontal for line plots (time-series
        percentiles for simulation data).
        """
        if not hasattr(self, 'prob_target'):
            self.prob_target = 0.7

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        q_list = [50, int((1 - self.prob_target) * 100), int(self.prob_target * 100)]
        ls_list = ['-', '--', '-.']

        # For simulation, plot the mean time series; otherwise use historical data.
        data = (pd.DataFrame(self.shocks_sim_draws.mean(axis=0), columns=self.df_shocks.columns)
                if sim else self.df_shocks)

        for i, col in enumerate(data.columns):
            if hist:
                # For histograms, flatten simulation draws if needed.
                series = (pd.Series(self.shocks_sim_draws[:, :, i].flatten())
                        if sim else data[col])
                if np.all(np.abs(series) < 1e-6):
                    series = series.round(0)

                series.plot(kind='hist', ax=axes[i], title=col, bins=50)
                if percentiles:
                    for q, ls in zip(q_list, ls_list):
                        q_val = np.percentile(series, q)
                        axes[i].axvline(q_val, color='red', linestyle=ls,
                                        label=f'{"median" if q==50 else f"{q} pct"}')
            else:
                # Plot the mean time series.
                data[col].plot(ax=axes[i], title=col, label='')
                if percentiles:
                    for q, ls in zip(q_list, ls_list):
                        if sim:
                            # Compute quantile as a time series across simulation draws.
                            q_val = np.percentile(self.shocks_sim_draws[:, :, i], q, axis=0)
                            axes[i].plot(data[col].index, q_val, color='red', linestyle=ls,
                                        label=f'{"median" if q==50 else f"{q} pct"}')
                        else:
                            q_val = np.percentile(data[col], q)
                            axes[i].axhline(q_val, color='red', linestyle=ls,
                                            label=f'{"median" if q==50 else f"{q} pct"}')
        if percentiles:
            axes[0].legend()

        plt.tight_layout()
        plt.show()


# ========================================================================================= #
#                              STOCHASTIC OPTIMIZATION METHODS                              #
# ========================================================================================= #

    def find_spb_stochastic(self,
                            bounds=(-10, 10),
                            stochastic_criteria=None,
                            stochastic_criterion_start_year=None,
                            print_update=False,
                            prob_target=None):
        """
        Find the SPB target such that the debt ratio declines (or stays below 60%) with probability prob_target (default 0.7).
        """
        # Set parameters
        self.print_update = print_update
        self.stochastic_criteria = stochastic_criteria or ['debt_declines', 'debt_below_60']

        if not hasattr(self, 'edp_steps'):
           self.edp_steps = None

        if not hasattr(self, 'deficit_resilience_steps'):
           self.deficit_resilience_steps = None

        if not hasattr(self, 'post_spb_steps'):
            self.post_spb_steps = None

        if prob_target is not None:
            self.prob_target = prob_target
        elif not hasattr(self, 'prob_target'):
            self.prob_target = 0.7

        if stochastic_criterion_start_year is None:
            self.stochastic_criterion_start = 0
        else:
            self.stochastic_criterion_start = stochastic_criterion_start_year - self.stochastic_start_year

        self.stochastic_optimization_dict = {}

        # Initial projection
        self.project(
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_spb_steps=self.post_spb_steps,
            scenario=None
            )

        # Optimize for both debt decline and debt remaining under 60 and choose the lower SPB
        self.spb_target = self._stochastic_optimization(bounds=bounds)

        # Project with optimal spb
        self.project(
            spb_target=self.spb_target,
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_spb_steps=self.post_spb_steps,
            scenario=None
            )

        return self.spb_target

    def _stochastic_optimization(self, bounds):
        """
        Optimizes for SPB that ensures debt remains below 60% with probability prob_target.
        """
        # Initital simulation
        self.simulate()

        # Set parameters
        self.spb_bounds = bounds
        self.stochastic_optimization_dict = {}

        # Optimize _target_pb to find b_target that ensures prob_debt_below_60 or prob_debt_declines == prob_target
        self.spb_target = minimize_scalar(self._stochastic_target, method='bounded', bounds=self.spb_bounds).x

        # Store results in a dataframe
        self.df_stochastic_optimization = pd.DataFrame(self.stochastic_optimization_dict).T

        return self.spb_target

    def _stochastic_target(self, spb_target):
        """
        Returns zero if primary balance ensures prop_debt_declines == prob_target or prob_debt_below_60 == prob_target.
        """
        # Simulate the debt-to-GDP ratio with the given primary balance target
        self.project(
            spb_target=spb_target,
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_spb_steps=self.post_spb_steps,
            scenario=None
            )

        # Combine shocks with new baseline
        self._combine_shocks_baseline()

        # Simulate debt ratio and calculate probability of debt exploding or exceeding 60
        self._simulate_debt()
        self.stochastic_optimization_dict[spb_target] = {}
        print_msg = f'spb: {spb_target:.2f}'
        if 'debt_declines' in self.stochastic_criteria:
            self.prob_debt_declines()
            self.stochastic_optimization_dict[spb_target]['prob_debt_declines'] = self.prob_declines
            print_msg += f', prob_debt_declines: {self.prob_declines:.2f}'
        if 'debt_stable' in self.stochastic_criteria:
            self.prob_debt_stable()
            self.stochastic_optimization_dict[spb_target]['prob_debt_stable'] = self.prob_stable
            print_msg += f', prob_debt_stable: {self.prob_stable:.2f}'
        if 'debt_below_60' in self.stochastic_criteria:
            self.prob_debt_below_60()
            self.stochastic_optimization_dict[spb_target]['prob_debt_below_60'] = self.prob_below_60
            print_msg += f', prob_debt_below_60: {self.prob_below_60:.2f}'
        if self.print_update:
            print(print_msg, end='\r')

        # Optimize for more probable target
        if 'debt_declines' in self.stochastic_criteria and 'debt_below_60' in self.stochastic_criteria:
            max_prob = np.max([self.prob_declines, self.prob_below_60])
        elif 'debt_stable' in self.stochastic_criteria and 'debt_below_60' in self.stochastic_criteria:
            max_prob = np.max([self.prob_stable, self.prob_below_60])
        elif 'debt_declines' in self.stochastic_criteria:
            max_prob = self.prob_declines
        elif 'debt_stable' in self.stochastic_criteria:
            max_prob = self.prob_stable
        elif 'debt_below_60' in self.stochastic_criteria:
            max_prob = self.prob_below_60
        else:
            raise ValueError('Unknown stochastic criteria or combination!')

        # Penalty term to avoid local minima at probability bounds
        if (np.isclose(max_prob, 0)) or (np.isclose(max_prob, 1)):
            penalty = np.max([0,spb_target/10])
        else:
            penalty = 0

        return np.abs(max_prob - self.prob_target) + penalty

    def prob_debt_declines(self):
        """
        Probability that the debt ratio declines over the stochastic projection period.
        """
        # Call the Numba JIT function with converted self variables
        self.prob_declines = prob_debt_declines_jit(
            N=self.N,
            d_sim=self.d_sim,
            stochastic_criterion_start=self.stochastic_criterion_start
            )
        return self.prob_declines

    def prob_debt_stable(self):
        """
        Probability that the debt ratio is stable (does not increase) at the end of the stochastic projection.
        """
        # Call the Numba JIT function with converted self variables
        self.prob_stable = prob_debt_stable_jit(
            N=self.N,
            d_sim=self.d_sim
            )
        return self.prob_stable

    def prob_debt_below_60(self):
        """
        Probability that the debt ratio is below 60% at the end of the stochastic projection.
        """
        # Call the Numba JIT function with converted self variables
        self.prob_below_60 = prob_debt_below_60_jit(
            N=self.N,
            d_sim=self.d_sim,
            )
        return self.prob_below_60

    def plot_target_func(self):
        """
        Plot the target function for the stochastic optimization.
        """
        results = {}
        for x in np.linspace(self.spb_bounds[0], self.spb_bounds[1], 100):
            y = self._stochastic_target(spb_target=x)
            results[x] = y
        results = pd.Series(results)
        results.plot()

# ========================================================================================= #
#                               INTEGRATED OPTIMIZERS                                       #
# ========================================================================================= #

    def find_spb_binding(self,
                         edp=True,
                         debt_safeguard=True,
                         deficit_resilience=True,
                         stochastic=True,
                         print_results=True,
                         stochastic_criteria=None,
                         save_df=False,
                         rules=None,
                         **rule_options):
        """
        Find the structural primary balance that meets the DSA criteria, the EDP requirements and the safeguards.

        The default rules follow Darvas, Welslau and Zettelmeyer (2024). Each rule can be switched to the version used
        in the Commission prior guidance calculation sheets, individually (keyword arguments) or all at once
        (rules='commission'), see BINDING_RULES and COMMISSION_RULES:

            dsa_criteria        'default': per deterministic scenario, debt declines or stays below 60%, deficit below 3%,
                                stochastic criteria; 'commission': debt declines in all scenarios and stochastically, or
                                debt below 60% in all scenarios, deficit below 3% (tolerance 0.05); technical information
                                (deficit and debt below 60% only) for countries with debt < 60% and deficit < 3% in T;
                                non-negative adjustment for the reference trajectory, at most -1 pp. per year otherwise
            stochastic_criteria list of 'debt_declines', 'debt_stable', 'debt_below_60'
            edp                 'default': triggered by a projected deficit above 3%, 0.5 pp. SPB steps followed by
                                0.5 pp. SB steps; 'commission': EDP status in T from the input file, abrogated after two
                                years with deficit below 3%, min. 0.5 pp. step after a year with deficit above 3% (in SB
                                terms from 2028); None: no EDP
            debt_safeguard      'default': average decline from the year the EDP is projected to be abrogated (the year
                                after the deficit falls below 3%, if it stays below 3%, as in the Commission sheets) or T
                                to adjustment end, 1 pp. if debt in T > 90%, 0.5 pp. otherwise; 'commission': by
                                start-of-year debt band (> 90%, 60-90%), average over adjustment years outside the EDP;
                                None
            deficit_resilience  'default': steps raised (up to 0.4/0.25 pp.) until the structural deficit is below 1.5%
                                in the same year; 'commission': 0.4/0.25 pp. step after a year with a structural deficit
                                above 1.55%; None
            frontloading        True: EDP and deficit resilience steps front-load the adjustment, later steps are reduced
                                to keep the SPB target; False: minimum steps are added on top (Commission)
            grid                None: exact SPB target; 0.01: annual adjustment rounded up to 0.01 pp. (Commission)

        Arguments edp, debt_safeguard and deficit_resilience accept True (rule from rules), False, or a rule name.
        """
        # Set rules: defaults, preset, and individual options
        self.rules = self._set_rules(rules, rule_options, edp, debt_safeguard, deficit_resilience, stochastic_criteria)
        self.stochastic_criteria = self.rules['stochastic_criteria']

        # Remove results of earlier runs
        for attr in ['guidance', 'reference_trajectory', 'edp_binding', 'debt_safeguard_binding',
                     'deficit_resilience_binding', 'edp_min_steps', 'binding_criterion']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Model settings used by project() during the optimization, restored afterwards
        settings = {'frontloading': self.frontloading, 'edp_active': getattr(self, 'edp_active', True)}
        self.frontloading = self.rules['frontloading']
        self.edp_active = self.rules['edp'] is not None

        # Default rules follow the sequential DWZ (2024) implementation, other rules a constant annual adjustment
        default = (all(self.rules[k] in (BINDING_RULES[k], None)
                       for k in ['dsa_criteria', 'edp', 'debt_safeguard', 'deficit_resilience'])
                   and self.rules['frontloading'] and self.rules['grid'] is None)
        try:
            if default:
                self._find_spb_binding_default(stochastic, print_results, save_df)
            else:
                self._find_spb_binding_rules(stochastic, print_results, save_df)
        finally:
            for k, v in settings.items():
                setattr(self, k, v)

    def _set_rules(self, rules, rule_options, edp, debt_safeguard, deficit_resilience, stochastic_criteria):
        """
        Combine default rules, preset and individual options.
        """
        if rules is None or rules == 'default':
            out = dict(BINDING_RULES)
        elif rules == 'commission':
            out = dict(COMMISSION_RULES)
        elif isinstance(rules, dict):
            out = {**BINDING_RULES, **rules}
        else:
            raise ValueError(f"Unknown rules '{rules}', use 'default', 'commission' or a dict")
        unknown = set(rule_options) - set(BINDING_RULES)
        if unknown:
            raise TypeError(f'Unknown rule options: {sorted(unknown)}')
        out.update(rule_options)
        for key, value in [('edp', edp), ('debt_safeguard', debt_safeguard), ('deficit_resilience', deficit_resilience)]:
            if value is False or value is None:
                out[key] = None
            elif isinstance(value, str):
                out[key] = value
        if stochastic_criteria is not None:
            out['stochastic_criteria'] = stochastic_criteria
        return out

    def _find_spb_binding_default(self, stochastic, print_results, save_df):
        """
        Binding SPB target with the default rules: DSA target, then EDP, debt safeguard and deficit resilience.
        """
        edp = self.rules['edp'] is not None
        debt_safeguard = self.rules['debt_safeguard'] is not None
        deficit_resilience = self.rules['deficit_resilience'] is not None

        # Initiate spb_target and dataframe dictionary
        self.save_df = save_df
        self.spb_target_dict = {}
        self.pb_target_dict = {}
        self.binding_parameter_dict = {}
        if self.save_df:
            self.df_dict = {}

        # Run DSA and deficit criteria and project toughest under baseline assumptions
        self.project(spb_target=None, edp_steps=None) # clear projection
        self._run_dsa(stochastic=stochastic)
        self._get_binding()

        # Apply EDP (edp_end = adjustment_start - 2 marks no EDP, as in find_edp)
        if edp:
            self._apply_edp()
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 2
        self.edp_min_steps = np.copy(self.edp_steps)

        # Apply debt safeguard
        if debt_safeguard:
            self._apply_debt_safeguard()

        # Apply deficit resilience
        if deficit_resilience:
            self._apply_deficit_resilience()

        self._save_binding(edp, debt_safeguard, deficit_resilience, print_results)

    def _save_binding(self, edp, debt_safeguard, deficit_resilience, print_results):
        """
        Save binding SPB target, adjustment parameters and results tables.
        """
        # Save binding SPB and PB target
        self.spb_target_dict['binding'] = self.spb_bca[self.adjustment_end]
        self.pb_target_dict['binding'] = self.pb[self.adjustment_end]

        # Save binding parameters to reproduce adjustment path
        self.binding_parameter_dict['spb_steps'] = self.spb_steps
        self.binding_parameter_dict['spb_target'] = self.binding_spb_target
        self.binding_parameter_dict['criterion'] = self.binding_criterion

        # Save EDP and safeguards parameters
        if edp:
            self.binding_parameter_dict['edp_binding'] = self.edp_binding
            self.binding_parameter_dict['edp_steps'] = self.edp_steps
        if debt_safeguard:
            self.binding_parameter_dict['debt_safeguard_binding'] = self.debt_safeguard_binding
        if deficit_resilience:
            self.binding_parameter_dict['deficit_resilience_binding'] = self.deficit_resilience_binding
            self.binding_parameter_dict['deficit_resilience_steps'] = self.deficit_resilience_steps
        self.binding_parameter_dict['net_expenditure_growth'] = self.net_expenditure_growth[self.adjustment_start:self.adjustment_end+1]

        # Results tables, displayed if print_results
        self.binding_results()
        if print_results:
            display_tables(self.binding_tables)

        # Save dataframe
        if self.save_df:
            self.df_dict['binding'] = self.df(all=True)

    # ---------------------------------------------------------------------------------------- #
    #  Binding SPB target with non-default rules (e.g. Commission prior guidance)              #
    # ---------------------------------------------------------------------------------------- #

    def _find_spb_binding_rules(self, stochastic, print_results, save_df, bounds=(-3, 3), tol=1e-4):
        """
        Binding SPB target for a constant annual adjustment: the DSA-based adjustment is the smallest constant annual
        step meeting the DSA criteria; the binding adjustment is the smallest step above it for which the path,
        including EDP and deficit resilience steps, meets the debt safeguard. Rules as set in self.rules.
        """
        rules = self.rules
        n, s, e = self.adjustment_period, self.adjustment_start, self.adjustment_end
        if hasattr(self, 'predefined_spb_steps'):
            raise NotImplementedError('predefined_spb_steps are only supported with the default rules')
        spb_start = self.spb_bca[s - 1]
        edp = rules['edp'] is not None
        debt_safeguard = rules['debt_safeguard'] is not None
        deficit_resilience = rules['deficit_resilience'] is not None
        self.save_df = save_df
        self.spb_target_dict, self.pb_target_dict, self.binding_parameter_dict = {}, {}, {}
        if self.save_df:
            self.df_dict = {}

        # 1. DSA-based annual adjustment
        a = self._dsa_annual_adjustments(stochastic, bounds, tol)
        a_dsa, criterion = self._combine_dsa_criteria(a)
        a_dsa_exact = a_dsa
        if rules['grid']:
            a_dsa = np.ceil(round(a_dsa / rules['grid'], 6)) * rules['grid']
        for k, v in a.items():
            self.spb_target_dict[k] = spb_start + n * v
        self.binding_spb_target = spb_start + n * a_dsa
        self.binding_criterion = criterion

        # 2. EDP with default rule: minimum steps from the EDP optimiser, kept fixed below
        self.edp_default_steps = np.full(n, np.nan)
        self.edp_period, self.edp_end = 0, s - 2
        if rules['edp'] == 'default':
            self.find_edp(spb_target=self.binding_spb_target)
            self.edp_default_steps = np.copy(self.edp_steps)

        # 3. Debt safeguard on the path including EDP and deficit resilience steps. The safeguard is not monotonic in
        # the adjustment (years drop out of debt bands), so search upwards on a grid
        a_binding = a_dsa
        if debt_safeguard and self.reference_trajectory:
            step = rules['grid'] or 0.01
            while not self._debt_safeguard_met(self._rules_path(a_binding)) and a_binding < bounds[1]:
                a_binding = np.round(a_binding + step, 6)
            if not self._debt_safeguard_met(self._rules_path(a_binding)):
                warnings.warn(f'{self.country}: debt safeguard not met at the upper bound of {bounds[1]} pp. per year')
        self.debt_safeguard_binding = bool(a_binding > a_dsa)
        if self.debt_safeguard_binding:
            self.binding_criterion = 'debt_safeguard'
            self.spb_target_dict['debt_safeguard'] = spb_start + n * a_binding
        self._rules_path(a_binding)

        # Record which minimum steps were binding
        self.edp_min_steps = np.copy(self.edp_steps)
        self.edp_binding = bool(np.any(self.edp_steps > a_binding + 1e-8))
        self.deficit_resilience_binding = bool(np.any(self.deficit_resilience_steps > a_binding + 1e-8))
        self.spb_target = self.spb_bca[e]
        self.binding_spb_target = spb_start + n * a_binding
        self.guidance = 'reference trajectory' if self.reference_trajectory else 'technical information'
        self.binding_parameter_dict.update({
            'annual_adjustment_dsa': a_dsa,
            'annual_adjustment_dsa_exact': a_dsa_exact,
            'annual_adjustment': a_binding,
            'guidance': self.guidance,
        })
        self._save_binding(edp, debt_safeguard, deficit_resilience, print_results)

    def _dsa_annual_adjustments(self, stochastic, bounds, tol):
        """
        Smallest constant annual SPB step meeting each DSA criterion.
        """
        n, s, e = self.adjustment_period, self.adjustment_start, self.adjustment_end
        commission = self.rules['dsa_criteria'] == 'commission'
        self.reference_trajectory = bool(self.d[s - 1] > 60 or self.ob[s - 1] < -3) or not commission
        scenarios = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']

        def linear(x, scenario='main_adjustment'):
            self.project(spb_steps=np.full(n, x), scenario=scenario)

        def debt_declines(x, scenario):
            linear(x, scenario)
            return np.all((np.diff(self.d[e:e + 11]) < 0) | (self.d[e + 1:e + 11] < 1e-3))

        def debt_below_60(x, scenario):
            linear(x, scenario)
            return self.d[e + 10] < 60 if commission else self.d[e + 10] <= 60

        def deficit_below_3(x):
            linear(x)
            return np.all(self.ob[e:e + 11] > -3.05) if commission else np.all(self.ob[e:e + 11] >= -3)

        def smallest(condition, low=bounds[0], high=bounds[1]):
            if condition(low):
                return low
            if not condition(high):
                warnings.warn(f'{self.country}: condition not met at upper bound {high}')
                return high
            while high - low > tol:
                mid = (low + high) / 2
                low, high = (low, mid) if condition(mid) else (mid, high)
            return high

        a = {'deficit_reduction': smallest(deficit_below_3)}
        if self.reference_trajectory:
            for sc in scenarios:
                a[f'debt_declines_{sc}'] = smallest(lambda x: debt_declines(x, sc))
                a[f'debt_below_60_{sc}'] = smallest(lambda x: debt_below_60(x, sc))
            if stochastic:
                try:
                    self.find_spb_stochastic(bounds=(self.spb_bca[s - 1] + n * bounds[0], self.spb_bca[s - 1] + n * bounds[1]),
                                             stochastic_criteria=self.rules['stochastic_criteria'])
                    a['stochastic'] = (self.spb_target - self.spb_bca[s - 1]) / n
                except Exception as e:
                    warnings.warn(f'{self.country}: stochastic criterion skipped ({e})')
        else:
            a['debt_below_60_main_adjustment'] = smallest(lambda x: debt_below_60(x, 'main_adjustment'))
        return a

    def _combine_dsa_criteria(self, a):
        """
        DSA-based annual adjustment from the criteria-specific adjustments. Returns (adjustment, binding criterion).
        """
        scenarios = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']
        if not self.reference_trajectory:
            candidates = {k: a[k] for k in ['deficit_reduction', 'debt_below_60_main_adjustment']}
            candidates['floor'] = -1
        elif self.rules['dsa_criteria'] == 'commission':
            option_a = {k: a[k] for k in ['deficit_reduction', 'stochastic'] + [f'debt_declines_{sc}' for sc in scenarios]
                        if k in a}
            option_b = {k: a[k] for k in ['deficit_reduction'] + [f'debt_below_60_{sc}' for sc in scenarios]}
            candidates = option_a if max(option_a.values()) <= max(option_b.values()) else option_b
            candidates = {**candidates, 'floor': 0}
        else:
            # Per scenario, debt declines or stays below 60%
            candidates = {sc: min(a[f'debt_declines_{sc}'], a[f'debt_below_60_{sc}']) for sc in scenarios}
            candidates['deficit_reduction'] = a['deficit_reduction']
            if 'stochastic' in a:
                candidates['stochastic'] = a['stochastic']
        criterion = max(candidates, key=candidates.get)
        return candidates[criterion], criterion

    def _rules_path(self, a):
        """
        Project the path for a constant annual adjustment a including EDP and deficit resilience steps according to
        self.rules. Returns the EDP status by projection year (used by the debt safeguard).
        """
        n, s = self.adjustment_period, self.adjustment_start
        resilience_step = 0.4 if n <= 4 else 0.25
        spb_steps = np.full(n, a, dtype=np.float64)
        edp_steps = np.copy(self.edp_default_steps)
        resilience_steps = np.full(n, np.nan)
        self.project(spb_steps=spb_steps, edp_steps=edp_steps, deficit_resilience_steps=resilience_steps)

        # Minimum steps depending on the previous year (Commission rules), set year by year
        if 'commission' in (self.rules['edp'], self.rules['deficit_resilience']):
            for k in range(n):
                t = s + k
                status = self._edp_status()
                if self.rules['edp'] == 'commission' and status[t - 1] == 1 and self.ob[t - 1] <= -3.05:
                    edp_steps[k] = 0.5 + ((self.interest_ratio[t] - self.interest_ratio[t - 1])
                                          if self.start_year + t >= 2028 else 0)
                if self.rules['deficit_resilience'] == 'commission' and self.sb[t - 1] <= -1.55:
                    resilience_steps[k] = resilience_step
                self.project(spb_steps=spb_steps, edp_steps=edp_steps, deficit_resilience_steps=resilience_steps)

        # Default deficit resilience rule applied on the resulting path
        if (self.rules['deficit_resilience'] == 'default'
                and (np.any(self.d[s - 1:self.adjustment_end + 1] > 60) or self.ob[s - 1] < -3)):
            self.spb_target = self.spb_bca[self.adjustment_end]
            self.deficit_resilience_target = np.full(n, -1.5, dtype=float)
            self.deficit_resilience_step = resilience_step
            self.deficit_resilience_start = s
            self._deficit_resilience_loop_adjustment()
        return self._edp_status()

    def _edp_status(self):
        """
        EDP status by projection year. Commission rule: status in T from the input file, abrogated once the deficit is
        below 3% (tolerance 0.05) in the two preceding years. Default rule: in EDP until the year in which the EDP is
        projected to be abrogated (see DsaModel._debt_safeguard_start).
        """
        status = np.zeros(self.projection_period)
        if self.rules['edp'] == 'commission':
            edp_T = self.params.get('EXCESSIVE_DEFICIT_PROCEDURE', np.nan)
            if np.isnan(edp_T):
                edp_T = int(self.ob[0] <= -3.05)
                warnings.warn(f'{self.country}: EDP status missing in the input file, set from the deficit in T ({edp_T})')
            status[0] = int(edp_T)
            ob_before = self._fiscal_balance_before_start()
            for j in range(1, self.projection_period):
                ob_2 = ob_before if j == 1 else self.ob[j - 2]
                status[j] = 0 if (status[j - 1] == 0 or (ob_2 > -3.05 and self.ob[j - 1] > -3.05)) else 1
        elif self.rules['edp'] == 'default':
            abrogation = self._debt_safeguard_start()
            if abrogation >= self.adjustment_start:
                status[:abrogation + 1] = 1
        return status

    def _debt_safeguard_met(self, edp_status):
        """
        Debt sustainability safeguard for the current projection.

        Commission rule: over the adjustment years outside an EDP, debt declines on average by at least 1 pp. per year
        while above 90% of GDP and 0.5 pp. while between 60% and 90%, with debt bands classified by the debt ratio at
        the start of each year. Default rule: average decline from the year the EDP is projected to be abrogated (or T) to
        the end of the adjustment period by 1 pp. if debt in T exceeds 90% and 0.5 pp. otherwise, for debt above 60% in T
        (see DsaModel._debt_safeguard_start).
        """
        s, e = self.adjustment_start, self.adjustment_end
        if self.rules['debt_safeguard'] == 'default':
            return self.d[s - 1] < 60 or self._debt_safeguard_criterion()

        years = range(s, e + 1)
        change = np.array([self.d[t] - self.d[t - 1] for t in years])
        above90 = np.array([self.d[t - 1] > 90 for t in years])
        between = np.array([60 < self.d[t - 1] <= 90 for t in years])
        no_edp = np.array([edp_status[t] == 0 for t in years])
        if not above90.any() and not between.any():
            return True
        if not no_edp.any():
            return True
        ok = True
        if above90.any():
            sel = no_edp if not between.any() else (above90 & no_edp)
            ok &= change[sel].mean() < -1 if sel.any() else True
        if between.any():
            sel = between & no_edp
            ok &= change[sel].mean() < -0.5 if sel.any() else True
        return bool(ok)

    def _run_dsa(self, stochastic=True, criterion='all'):
        """
        Run the DSA: SPB targets for all deterministic criteria and the stochastic criterion (criterion='all'), or
        re-optimize a single criterion on the EDP path (used by _apply_edp).
        """
        # Define deterministic criteria
        deterministic_criteria_list = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g', 'deficit_reduction']

        # If all criteria, run all deterministic and stochastic
        if criterion == 'all':

            # Run all deterministic scenarios, skip if criterion not applicable
            for deterministic_criterion in deterministic_criteria_list:
                try:
                    self.find_spb_deterministic(criterion=deterministic_criterion)
                    self.spb_target_dict[deterministic_criterion] = self.spb_bca[self.adjustment_end]
                    self.pb_target_dict[deterministic_criterion] = self.pb[self.adjustment_end]
                    if self.save_df:
                        self.df_dict[deterministic_criterion] = self.df(all=True)
                except Exception as e:
                    raise ValueError(f'{deterministic_criterion} did not converge for {self.country}') from e

            # Run stochastic scenario, skip with a warning if not possible (e.g. lack of shock data)
            if stochastic:
                try:
                    self.find_spb_stochastic(stochastic_criteria=self.stochastic_criteria)
                    self.spb_target_dict['stochastic'] = self.spb_bca[self.adjustment_end]
                    self.pb_target_dict['stochastic'] = self.pb[self.adjustment_end]
                    if self.save_df:
                        self.df_dict['stochastic'] = self.df(all=True)
                except Exception as e:
                    warnings.warn(f'{self.country}: stochastic criterion skipped ({e})')

        # If specific criterion given for EDP optimization, run only one optimization
        else:
            if criterion in deterministic_criteria_list:
                self.find_spb_deterministic(criterion=criterion)

            elif criterion == 'stochastic':
                self.find_spb_stochastic(stochastic_criteria=self.stochastic_criteria)

            # Replace binding scenario
            self.binding_spb_target = self.spb_bca[self.adjustment_end]
            self.spb_target_dict['edp'] = self.spb_bca[self.adjustment_end]
            self.pb_target_dict['edp'] = self.pb[self.adjustment_end]

    def _get_binding(self):
        """
        Get binding SPB target and scenario from dictionary with SPB targets.
        """
        # Get binding SPB target and scenario from dictionary with SPB targets
        self.binding_spb_target = np.max(list(self.spb_target_dict.values()))
        self.binding_criterion = list(self.spb_target_dict.keys())[np.argmax(list(self.spb_target_dict.values()))]

        # Project under baseline assumptions
        self.project(spb_target=self.binding_spb_target, scenario=None)

    def _apply_edp(self):
        """
        Check if EDP is binding in binding scenario and apply if it is.
        """
        # Check if EDP binding, run DSA for periods after EDP and project new path under baseline assumptions
        self.find_edp(spb_target=self.binding_spb_target)

        if not np.all([np.isnan(self.edp_steps)]) and np.any([self.edp_steps >= self.spb_steps - 1e-8]):
            self.edp_binding = True
            self._run_dsa(criterion=self.binding_criterion)
            self.project(
                spb_target=self.binding_spb_target,
                edp_steps=self.edp_steps,
                scenario=None
                )
            if self.save_df:
                self.df_dict['edp'] = self.df(all=True)
        else:
            self.edp_binding = False

    def _apply_debt_safeguard(self):
        """
        Check if the debt sustainability safeguard is binding on the current path and apply it if it is.
        """
        # Keep the steps of the EDP period as minimum steps while re-optimizing the SPB target
        self.edp_steps[:self.edp_period] = self.spb_steps[:self.edp_period]

        # Debt safeguard binding for countries with high debt and insufficient average decline after EDP abrogation
        if (self.d[self.adjustment_start-1] >= 60
            and not self._debt_safeguard_criterion()):

            # Find the SPB target for the debt safeguard. Search above the current binding target: the safeguard is not monotonic in the SPB target, as a lower
            # target delays the abrogation of the EDP and hence the start of the safeguard
            self.spb_debt_safeguard_target = self.find_spb_deterministic(
                criterion='debt_safeguard', bounds=(self.binding_spb_target, 10))

            # If debt safeguard SPB target is higher than DSA target, save debt safeguard target
            # 1e-3 tolerance for edp search algo edge case, 1e-8 tolerance for floating point errors
            if self.spb_debt_safeguard_target > self.binding_spb_target + 1e-3 + 1e-8:
                self.debt_safeguard_binding = True
                self.binding_spb_target = self.spb_debt_safeguard_target
                self.spb_target = self.binding_spb_target
                self.binding_criterion = 'debt_safeguard'
                self.spb_target_dict['debt_safeguard'] = self.binding_spb_target
                self.pb_target_dict['debt_safeguard'] = self.pb[self.adjustment_end]
                if self.save_df:
                    self.df_dict['debt_safeguard'] = self.df(all=True)
            else:
                self.debt_safeguard_binding = False
        else:
            self.debt_safeguard_binding = False

    def _apply_deficit_resilience(self):
        """
        Apply deficit resilience safeguard after binding scenario.
        """
        # For countries with high deficit, find SPB target that brings and keeps deficit below 1.5%
        if (np.any(self.d[self.adjustment_start-1:self.adjustment_end+1] > 60)
            or self.ob[self.adjustment_start-1] < -3):
                self.find_spb_deficit_resilience()

        # Save results and print update
        if np.any([~np.isnan(self.deficit_resilience_steps)]):
            self.deficit_resilience_binding = True
            self.spb_target_dict['deficit_resilience'] = self.spb_bca[self.adjustment_end]
            self.pb_target_dict['deficit_resilience'] = self.pb[self.adjustment_end]
            self.binding_spb_target = self.spb_bca[self.adjustment_end]
            if self.save_df:
                self.df_dict['deficit_resilience'] = self.df(all=True)
        else:
            self.deficit_resilience_binding = False

    def binding_results(self, years_after=10):
        """
        Results of find_spb_binding as labelled tables:
            'Summary':          binding SPB target, annual adjustment, binding criterion and safeguards
            'SPB targets':      SPB at the end of the adjustment period required by each criterion
            'Adjustment path':  annual SPB steps, minimum steps, balances, debt and net expenditure growth by year
        """
        n, s, e = self.adjustment_period, self.adjustment_start, self.adjustment_end
        spb_start = self.spb_bca[s - 1]
        binding = self.spb_target_dict['binding']
        rules = getattr(self, 'rules', BINDING_RULES)
        changed = {k: v for k, v in rules.items() if BINDING_RULES.get(k) != v}
        rules_label = ('default' if not changed else 'default (no EDP)' if changed == {'edp': None}
                       else 'commission' if rules == COMMISSION_RULES
                       else ', '.join(f'{k}={v}' for k, v in changed.items()))
        dsa_keys = [k for k in self.spb_target_dict if k not in ['edp', 'debt_safeguard', 'deficit_resilience', 'binding']]
        dsa_target = (spb_start + n * self.binding_parameter_dict['annual_adjustment_dsa']
                      if 'annual_adjustment_dsa' in self.binding_parameter_dict
                      else max(self.spb_target_dict[k] for k in dsa_keys))

        summary = pd.Series({
            'Country': COUNTRY_NAMES.get(self.country, self.country),
            'Adjustment period': f'{self.adjustment_start_year}-{self.adjustment_end_year}',
            'Rules': rules_label,
            'Guidance': getattr(self, 'guidance', ''),
            'Binding criterion': criterion_label(self.binding_criterion),
            'SPB in T (% of GDP)': spb_start,
            'DSA-based SPB target (% of GDP)': dsa_target,
            'SPB at end of adjustment (% of GDP)': binding,
            'Average annual adjustment (pp.)': (binding - spb_start) / n,
            'Average net expenditure growth (%)': np.mean(self.net_expenditure_growth[s:e + 1]),
            'EDP abrogation year (projected)': self.edp_abrogation_year(),
            'EDP binding': getattr(self, 'edp_binding', None),
            'Debt safeguard binding': getattr(self, 'debt_safeguard_binding', None),
            'Deficit resilience binding': getattr(self, 'deficit_resilience_binding', None),
        }, name=self.country)
        summary = summary[[v is not None and v != '' for v in summary.values]]

        targets = pd.DataFrame({
            'SPB at end of adjustment': {criterion_label(k): v for k, v in self.spb_target_dict.items()},
            'Average annual adjustment': {criterion_label(k): (v - spb_start) / n for k, v in self.spb_target_dict.items()},
        })
        targets.index.name = 'Criterion'

        years = range(s - 1, min(e + years_after, self.projection_period - 1) + 1)
        edp_steps = np.full(self.projection_period, np.nan)
        resilience_steps = np.full(self.projection_period, np.nan)
        edp_min_steps = getattr(self, 'edp_min_steps', getattr(self, 'edp_steps', None))
        if edp_min_steps is not None:
            edp_steps[s:e + 1] = edp_min_steps
        if getattr(self, 'deficit_resilience_steps', None) is not None:
            resilience_steps[s:e + 1] = self.deficit_resilience_steps
        path = pd.DataFrame({
            'Annual SPB adjustment (pp.)': self.spb_bca_adjustment,
            'EDP minimum step (pp.)': edp_steps,
            'Deficit resilience minimum step (pp.)': resilience_steps,
            'SPB before change in ageing costs (% of GDP)': self.spb_bca,
            'Structural primary balance (% of GDP)': self.spb,
            'Overall balance (% of GDP)': self.ob,
            'Structural balance (% of GDP)': self.sb,
            'Debt (% of GDP)': self.d,
            'Net expenditure growth (%)': self.net_expenditure_growth,
        }, index=pd.Index(range(self.start_year, self.end_year + 1), name='Year')).iloc[list(years)]
        path.loc[path.index[0], ['Annual SPB adjustment (pp.)', 'Net expenditure growth (%)']] = np.nan
        path = path.dropna(axis=1, how='all')

        self.binding_tables = {'Summary': summary.to_frame(), 'SPB targets': targets, 'Adjustment path': path}
        return self.binding_tables

    def edp_abrogation_year(self):
        """
        Year in which the EDP is projected to be abrogated on the current path, None if no EDP applies or the EDP is
        not abrogated during the projection. Commission EDP rule: first year after the status turns to zero, minus one
        (see _edp_status); default rule: DsaModel._debt_safeguard_start.
        """
        rules = getattr(self, 'rules', BINDING_RULES)
        if rules.get('edp') == 'commission':
            status = self._edp_status()
            if status[0] == 0:
                return None
            ended = np.where(status == 0)[0]
            return int(self.start_year + ended[0] - 1) if len(ended) else None
        if rules.get('edp') is None or not self._edp_applies():
            return None
        last = self.projection_period - 1
        abrogation = self._edp_abrogation_index(last=last)
        return int(self.start_year + abrogation) if abrogation <= last else None

    def find_deficit_prob(self):
        """
        Probability of an excessive deficit in each year of the adjustment period on the binding SPB path (call
        find_spb_binding first). Only interest rate and growth shocks are drawn; the primary balance responds to growth
        shocks through the budget balance semi-elasticity. Shock data and stochastic settings are restored afterwards.
        """
        # Store stochastic settings and shock data, restored at the end
        settings = {k: getattr(self, k) for k in ['stochastic_start', 'stochastic_end', 'stochastic_period', 'draw_period']}
        df_shocks = self.df_shocks

        # Project the binding path (annual steps as found by find_spb_binding)
        self.project(spb_steps=self.binding_parameter_dict['spb_steps'], scenario=None)
        if not hasattr(self, 'N'):
            self.N = 100000

        # Set stochastic period to adjustment period
        self.stochastic_start = self.adjustment_start
        self.stochastic_end = self.adjustment_end + 1
        self.stochastic_period = self.adjustment_period + 1

        if self.shock_frequency == 'quarterly':
            self.draw_period = self.stochastic_period * 4
        else:
            self.draw_period = self.stochastic_period

        # Set exchange rate and primary balance shocks to zero (on a copy of the shock data)
        self.df_shocks = df_shocks.copy()
        self.df_shocks[['EXR_EUR', 'EXR_USD', 'PRIMARY_BALANCE']] = 0

        # Draw quarterly shocks
        self._draw_shocks_normal()

        # Aggregate quarterly shocks to annual shocks
        if self.shock_frequency == 'quarterly':
            self._aggregate_shocks_quarterly()
        else:
            self._aggregate_shocks_annual()

        # Primary balance shock (index 4) is the nominal growth shock (index 3) times the budget balance semi-elasticity
        self.shocks_sim[:, 4] = self.budget_balance_elasticity * self.shocks_sim[:, 3]

        # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()

        # Simulate deficit
        self.ob_sim = np.zeros([self.N, self.stochastic_period+1])
        self.ob_sim[:, 0] = self.ob[self.stochastic_start-1]
        self._simulate_deficit()

        # Calculate probability of excessive deficit
        self.prob_deficit = self._prob_deficit()

        # Restore stochastic settings and shock data
        self.df_shocks = df_shocks
        for k, v in settings.items():
            setattr(self, k, v)

        return self.prob_deficit

    def _simulate_deficit(self):
        """
        Simulate the fiscal balance ratio using the baseline variables and the shocks.
        """
        # Call the Numba JIT function with converted self variables and d_sim as an argument
        simulate_deficit_jit(
            N=self.N, stochastic_period=self.stochastic_period, pb_sim=self.pb_sim, iir_sim=self.iir_sim, ng_sim=self.ng_sim, d_sim=self.d_sim, ob_sim=self.ob_sim
            )

    def _prob_deficit(self):
        """
        Calculate the probability of the deficit exceeding 3% in two consecutive period or 3.5% in one period during adjustment.
        """
        prob_excessive_deficit = np.full(self.adjustment_period, 0, dtype=np.float64)
        for n in range(self.N):
            for i in range(self.adjustment_period):
                if -3.5 > self.ob_sim[n, i+1] or (-3 > self.ob_sim[n, i+1] and -3 > self.ob_sim[n, i+2]):
                    prob_excessive_deficit[i] += 1
        return prob_excessive_deficit / self.N

    def var_forecast_pb(self, forecast_start_year=None):
        """
        Create conditional forecast of primary balance based on VAR estimates and deterministic forecasts.
        """
        # Set parameters
        if forecast_start_year is None:
            forecast_start_year = self.adjustment_start_year
        self.forecast_start_year = forecast_start_year
        self.forecast_horizon = self.end_year - forecast_start_year

        # project and simulate to ensure all attributes are set
        self.project()
        self.simulate()

        # Annual shock data for the VAR (shock data and frequency are restored below)
        original_shock_frequency, original_df_shocks = self.shock_frequency, self.df_shocks
        self.shock_frequency = 'annual'
        self._get_shock_data()
        var_sample = self.df_shocks[[
            'INTEREST_RATE_ST',
            'INTEREST_RATE_LT',
            'NOMINAL_GDP_GROWTH',
            'PRIMARY_BALANCE'
            ]]

        # Fit VAR model, lag length by BIC
        self.var_pb = VAR(var_sample).fit(ic='bic')

        # Restore shock data and frequency
        self.shock_frequency, self.df_shocks = original_shock_frequency, original_df_shocks

        # Extract lags and coefficients from the VAR model.
        lags = self.var_pb.k_ar
        pb_coefs = self.var_pb.params['PRIMARY_BALANCE'].values  # shape: (1 + p * len(var_names),)

        # Initialize forecasted primary balance and steps steps.
        self.forecast_pb = np.copy(self.pb)
        self.forecast_spb = np.copy(self.spb)
        self.forecast_pb_step = np.diff(self.pb, prepend=np.nan)

        # Calculate conditional forecast for each year and lag.
        for t, y in enumerate(
            range(self.forecast_start_year, self.end_year+1),
            start = self.forecast_start_year - self.start_year):
            # Construct the regressor vector X:
            X = [1]
            # For each lag, append the value of each variable at time t-lag.
            for l in range(1, lags + 1):
                for var in self.var_pb.params.columns:

                    # For pb, use the actual value before the forecast period.
                    if var == 'PRIMARY_BALANCE':
                        if self.start_year + t <= forecast_start_year:
                            value = self.pb[t-l] - self.pb[t-l-1]
                        else:
                            value = self.forecast_pb_step[t-l]

                    # For other variables, always use actual values.
                    elif var == 'INTEREST_RATE_ST':
                        value = self.i_st[t-l] - self.i_st[t-l-1]
                    elif var == 'INTEREST_RATE_LT':
                        value = self.i_lt[t-l] - self.i_lt[t-l-1]
                    elif var == 'NOMINAL_GDP_GROWTH':
                        value = self.ng[t-l] - self.ng[t-l-1]

                    # Append the value to the regressor vector.
                    X.append(value)

            # Calculate the forecasted primary balance for year t.
            X = np.array(X)
            self.forecast_pb_step[t] = np.dot(X, pb_coefs)
            self.forecast_pb[t] = self.forecast_pb[t-1] + self.forecast_pb_step[t]

# ========================================================================================= #
#                                NUMBA OPTIMIZED FUNCTIONS                                  #
# ========================================================================================= #

@jit(nopython=True)
def vecmatmul(vec, mat):
    """
    Multiply a 1d vector with a 2d matrix.
    """
    rows, cols = mat.shape
    result = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            result[i] += mat[i, j] * vec[j]
    return result

@jit(nopython=True)
def construct_var_shocks(N, draw_period, shocks_sim_draws, lags, intercept, coefs, residual_draws, stochastic_within_adjustment):
    """
    Simulate the shocks for the baseline variables.
    """
    # set all coefs for pb, which is pos -1 to zero
    coef_pb_zero = coefs.copy()
    coef_pb_zero[:,-1,:] = 0
    coef_pb_zero[:,:,-1] = 0
    intercept_pb_zero = intercept.copy()
    intercept_pb_zero[-1] = 0

    for n in range(N):
        for t in range(draw_period):
            if t < stochastic_within_adjustment:
                use_coefs = coef_pb_zero
                use_intercept = intercept_pb_zero
            else:
                use_coefs = coefs
                use_intercept = intercept
            shock = use_intercept.copy()
            for lag in range(1, lags + 1):
                if t - lag >= 0:
                    shock += vecmatmul(shocks_sim_draws[n, t - lag, :], use_coefs[lag - 1])
            shocks_sim_draws[n, t, :] = shock + residual_draws[n, t, :]
    return shocks_sim_draws

@jit(nopython=True)
def combine_shocks_baseline_jit(N, stochastic_start, stochastic_end, shocks_sim, exr_eur, exr_usd, iir, ng, pb, sf, d, d_sim, exr_eur_sim, exr_usd_sim, iir_sim, ng_sim, pb_sim, sf_sim):
    """
    Add shocks to the baseline variables and set starting values for simulation.
    """
    # Add shocks to the baseline variables for stochastic period
    for n in range(N):
        exr_eur_sim[n, 1:] = exr_eur[stochastic_start:stochastic_end+1] + shocks_sim[n, 0]
        exr_usd_sim[n, 1:] = exr_usd[stochastic_start:stochastic_end+1] + shocks_sim[n, 1]
        iir_sim[n, 1:] = iir[stochastic_start:stochastic_end+1] + shocks_sim[n, 2]
        ng_sim[n, 1:] = ng[stochastic_start:stochastic_end+1] + shocks_sim[n, 3]
        pb_sim[n, 1:] = pb[stochastic_start:stochastic_end+1] + shocks_sim[n, 4]

    # Set values for stock-flow adjustment
    sf_sim[:, 1:] = sf[stochastic_start:stochastic_end+1]

    # Set the starting values to the last value before the stochastic period
    d_sim[:, 0] = d[stochastic_start-1]
    exr_eur_sim[:, 0] = exr_eur[stochastic_start-1]
    exr_usd_sim[:, 0] = exr_usd[stochastic_start-1]
    iir_sim[:, 0] = iir[stochastic_start-1]
    ng_sim[:, 0] = ng[stochastic_start-1]
    pb_sim[:, 0] = pb[stochastic_start-1]

@jit(nopython=True)
def simulate_debt_jit(N, stochastic_period, D_share_domestic, D_share_eur, D_share_usd, d_sim, iir_sim, ng_sim, exr_eur_sim, exr_usd_sim, pb_sim, sf_sim):
    """
    Simulate the debt-to-GDP ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, stochastic_period+1):
            d_sim[n, t] = D_share_domestic * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) \
                        + D_share_eur * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) * (exr_eur_sim[n, t]) / (exr_eur_sim[n, t-1]) \
                        + D_share_usd * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) * (exr_usd_sim[n, t]) / (exr_usd_sim[n, t-1]) \
                        - pb_sim[n, t] + sf_sim[n, t]

@jit(nopython=True)
def mean_jit(arr):
    """
    Calculate the mean of an array.
    """
    total = 0.0
    count = 0
    for i in range(arr.shape[0]):
        total += arr[i]
        count += 1
    return total / count

@jit(nopython=True)
def prob_debt_declines_jit(N, d_sim, stochastic_criterion_start):
    """
    Calculate the probability of the debt-to-GDP ratio exploding.
    """
    prob_declines = 0
    d_start = mean_jit(d_sim[:, stochastic_criterion_start])
    for n in range(N):
        if d_start >= d_sim[n, -1]:
            prob_declines += 1
    return prob_declines / N

@jit(nopython=True)
def prob_debt_stable_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio stabalizing by projection end.
    """
    d_penultimate_sorted = np.sort(d_sim[:, -5])
    d_last_sorted = np.sort(d_sim[:, -1])
    n = d_sim.shape[0]
    prob_debt_stable = 0
    for i in range(10000):
        idx = int((i / 10000) * (n - 1))
        if d_penultimate_sorted[idx] >= d_last_sorted[idx]:
            prob_debt_stable += 1
    return prob_debt_stable / 10000

@jit(nopython=True)
def prob_debt_below_60_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio exceeding 60
    """
    prob_debt_below_60 = 0
    for n in range(N):
        if d_sim[n, -1] <= 60:
            prob_debt_below_60 += 1
    return prob_debt_below_60 / N

@jit(nopython=True)
def simulate_deficit_jit(N, stochastic_period, pb_sim, iir_sim, ng_sim, d_sim, ob_sim):
    """
    Simulate the fiscal balance ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, stochastic_period+1):
            ob_sim[n, t] = pb_sim[n, t] - iir_sim[n, t] / 100 / (1 + ng_sim[n, t] / 100) * d_sim[n, t-1]