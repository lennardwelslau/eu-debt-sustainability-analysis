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
# 4. Probability of an excessive deficit and conditional primary balance forecasts.
# 5. Numba functions that speed up the simulations.
#
# The EU fiscal rules (find_spb_binding, EDP, safeguards) are inherited from FiscalRules (FiscalRules.py).
#
# For comments and suggestions please contact lennard.welslau[at]gmail[dot]com
#
# Author: Lennard Welslau
# Updated: 2026-09-29
# ========================================================================================= #

# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from statsmodels.tsa.api import VAR
from numba import jit
from classes import DsaModel
from classes.DsaModelClass import DEFAULT_INPUT_FILE
from classes.FiscalRules import FiscalRules
from data_pipeline import read_shocks

# Fanchart style, shared by StochasticDsaModel.fanchart and the country charts (functions/country_charts.py)
FAN_OUTER, FAN_INNER = plt.get_cmap('Blues')(0.35), plt.get_cmap('Blues')(0.7)


def plot_fan(ax, df, deterministic_label='Deterministic'):
    """
    Plot a fanchart on ax: 10th-30th and 70th-90th percentiles (light blue), 30th-70th percentiles (dark blue), median
    and the deterministic path. df is indexed by year with columns 'baseline' and 'p10', 'p30', 'p50', 'p70', 'p90'
    (NaN outside the stochastic period; percentile columns may be missing for deterministic results).
    """
    sim = df.dropna(subset=['p50']) if 'p50' in df else df.iloc[0:0]
    if len(sim):
        ax.fill_between(sim.index, sim['p10'], sim['p30'], color=FAN_OUTER, lw=0, label='10th-30th / 70th-90th pct')
        ax.fill_between(sim.index, sim['p70'], sim['p90'], color=FAN_OUTER, lw=0)
        ax.fill_between(sim.index, sim['p30'], sim['p70'], color=FAN_INNER, lw=0, label='30th-70th pct')
        ax.plot(sim.index, sim['p50'], color='black', lw=2, label='Median')
    ax.plot(df.index, df['baseline'], color='C3', ls='--', lw=2, label=deterministic_label)




class StochasticDsaModel(FiscalRules, DsaModel):
    """
    DSA model with stochastic projections. Inherits the deterministic model (DsaModel) and the EU fiscal rules
    (FiscalRules: find_spb_binding, EDP, debt sustainability and deficit resilience safeguards).
    """

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
                winsorize_sample=True, # clip shocks to the 5th and 95th percentiles (or to q and 1 - q if a number q)
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

        # Adjust outliers by clipping shocks to the 5th and 95th percentiles (True) or to the q-th and (1 - q)-th
        # percentiles (winsorize_sample = q, e.g. 0.01)
        if self.winsorize_sample:
            q = 0.05 if self.winsorize_sample is True else float(self.winsorize_sample)
            self.df_shocks = self.df_shocks.clip(
                lower=self.df_shocks.quantile(q, axis=0),
                upper=self.df_shocks.quantile(1 - q, axis=0),
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

    def fanchart(self, var='d', plot=True, save_as=None, xlim=None, ylim=None, figsize=(10, 6)):
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

        # Save fanchart data in a dataframe
        self.df_fanchart = pd.DataFrame({'year': years, 'baseline': bl_var})
        for pct in self.pcts_dict:
            df_pct = pd.DataFrame({'year': years[self.stochastic_start-1:self.stochastic_end+1], f'p{pct}':
            self.pcts_dict[pct]})
            self.df_fanchart = self.df_fanchart.merge(df_pct, on='year', how='left')

        # Plot the fanchart (plot_fan) with the legend below the plot
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            df = self.df_fanchart.set_index('year').loc[xlim[0]:xlim[1]]  # visible years only, for the y-axis range
            plot_fan(ax, df)

            # Plot layout
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
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

        if stochastic_criterion_start_year is None:  # a setting of the model (e.g. stochastic_start_at_adjustment)
            stochastic_criterion_start_year = getattr(self, 'stochastic_criterion_start_year', None)
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
#                               DEFICIT PROBABILITY AND FORECASTS                           #
# ========================================================================================= #

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
