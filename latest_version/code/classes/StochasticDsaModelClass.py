# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Stochastic Sub Class     #
# ========================================================================================= #
#
# The StochasticDsaModel isa subclass of DsaModel to make stochastic projections around the 
# deterministic debt path. The model takes into account shocks to the exchange rates, interest 
# rates, GDP growth, and the primary balances. The subclass encompasses three primary parts:
#
# 1. **Simulation Methods:** These methods simulate the stochastic model by drawing quarterly 
#    or annual shocks from a multivariate normal distribution and aggregating them to annual shocks. 
#    They then combine the shocks with the baseline variables and set the starting values for the 
#    simulation. Finally, they simulate the debt-to-GDP ratio using the baseline variables and the shocks.
# 2. **Stochastic optimization Methods:** These methods optimize the structural primary balance 
#    to ensure debt sustainability under stochastic criteria.
# 3. **Integrated Optimizers:** These methods combine the stochastic optimization methods to find 
#    the optimal structural primary balance that ensures debt sustainability under different criteria.
# 4. **Numba Optimized Functions:** These functions are used to speed up the simulation process.
#
# For comments and suggestions please contact lennard.welslau[at]gmail[dot]com
#
# Author: Lennard Welslau
# Updated: 2024-06-01
# ========================================================================================= #

# Import libraries and modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')
from scipy.optimize import minimize_scalar
from statsmodels.tsa.api import VAR
from numba import jit
from classes import DsaModel

class StochasticDsaModel(DsaModel):

# ========================================================================================= #
#                               INIITIALIZE SUBCLASS                                        #
# ========================================================================================= #
    def __init__(self, 
                country, # ISO code of country
                start_year=2023, # start year of projection, first year is baseline value
                end_year=2070, # end year of projection
                adjustment_period=4, # number of years for linear spb_bca adjustment
                adjustment_start_year=2025, # start year of linear spb_bca adjustment
                ageing_cost_period=10, # number of years for ageing cost adjustment
                shock_sample_start=2000, # start year of shock sample
                stochastic_start_year=None, # start year of stochastic projection
                stochastic_period=5, # number of years for stochastic projection
                shock_frequency='quarterly', # start year of stochastic simulation
                estimation='normal', # estimation method for covariance matrix
                estimation_method='cholesky', # method for drawing shocks from VAR model
                fiscal_multiplier=0.75, 
                fiscal_multiplier_persistence=3,
                fiscal_multiplier_type='com',
                bond_data=False, # Use bond level data for repayment profile
                ): 
        
        # Initialize base class
        super().__init__(
            country, 
            start_year, 
            end_year, 
            adjustment_period, 
            adjustment_start_year, 
            ageing_cost_period, 
            fiscal_multiplier,
            fiscal_multiplier_persistence,
            fiscal_multiplier_type,
            bond_data
            )
        
        # Set stochastic parameters
        self.shock_frequency = shock_frequency
        self.shock_sample_start = shock_sample_start
        self.estimation = estimation
        self.estimation_method = estimation_method
        if stochastic_start_year is None: 
            stochastic_start_year = adjustment_start_year + adjustment_period
        self.stochastic_start_year = stochastic_start_year
        self.stochastic_period = stochastic_period
        self.stochastic_start = stochastic_start_year - start_year
        self.stochastic_end = self.stochastic_start + stochastic_period - 1 
        if shock_frequency == 'quarterly':
            self.draw_period = stochastic_period * 4 
        elif shock_frequency == 'annual':
            self.draw_period = stochastic_period
        
        # Get shock data
        self._get_shock_data()

    def _get_shock_data(self):
        """
        Get shock data from Excel file and adjust outliers.
        """
        # Read country shock data and get number of variables for quarterly data
        if self.shock_frequency == 'quarterly':
            self.df_shocks = pd.read_csv(self._base_dir + 'data/InputData/stochastic_data_quarterly.csv').set_index('YEAR')
            self.df_shocks = self.df_shocks.loc[self.df_shocks['COUNTRY'] == self.country]
            self.df_shocks.index = pd.PeriodIndex(self.df_shocks.index, freq='Q')

            # If quarterly shock data is not available, set parameters to annual
            if self.df_shocks.empty: 
                # print(f'No quarterly shock data available for {self.country}, using annual data instead.')
                self.shock_frequency = 'annual'
                self.draw_period = self.stochastic_period
        
        # Read country shock data for annual data        
        if self.shock_frequency == 'annual':
            self.df_shocks = pd.read_csv(self._base_dir + 'data/InputData/stochastic_data_annual.csv').set_index('YEAR')
            self.df_shocks = self.df_shocks.loc[self.df_shocks['COUNTRY'] == self.country]
            self.df_shocks.index = pd.PeriodIndex(self.df_shocks.index, freq='A')

        # Subset shock period and order variables
        self.df_shocks = self.df_shocks.loc[self.df_shocks.index.astype(str).str[:4].astype(int) >= self.shock_sample_start]
        self.df_shocks = self.df_shocks[
            ['EXR_EUR', 'EXR_USD', 'INTEREST_RATE_ST', 'INTEREST_RATE_LT', 'NOMINAL_GDP_GROWTH', 'PRIMARY_BALANCE']
            ]
        
        # Get number of variables
        self.num_variables = self.df_shocks.shape[1]
        assert self.num_variables == 6, 'Unexpected number of shock variables!'
        
        # Adjust outliers by keeping only 95th to 5th percentile
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
        elif self.estimation == 'var': 
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
        
    def _draw_shocks_var(self):
        """
        Draw quarterly or annual shocks from a VAR model.

        This method estimates a VAR model on the shock DataFrame and then draws N samples of quarterly shocks from the residuals of the VAR 
        model using a bootstrap method. It reshapes the shocks into a 4-dimensional array of shape (N, draw_period, num_variables) where N is the
        number of simulations, draw_period is the number of consecutive years or quarters drawn, and num_variables is the number of shock variables.
        """
        # Define sample for VAR model, exclude exr_eur_shock for EA countries and DNK
        ea_countries = ['AUT', 'BEL', 'BGR', 'DNK', 'HRV', 'CYP', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN', 'ESP']
        var_sample = self.df_shocks 
        if self.country in ea_countries: var_sample.drop(columns=['EXR_EUR'], inplace=True) 
        elif self.country == 'USA': var_sample.drop(columns=['EXR_USD'], inplace=True)

        # Estimate VAR model 
        varmodel = VAR(var_sample)
        self.var = varmodel.fit(ic='bic')

        # Extract parameters from the VAR results
        lags = self.var.k_ar
        intercept = self.var.params.iloc[0].values
        coefs = self.var.coefs
        residuals = self.var.resid.values

        # Use bootstrap sampling from the residuals or Cholesky decomposition of the covariance matrix
        if self.estimation_method == 'bootstrap':
            residual_draws = residuals[np.random.choice(len(residuals), size=(self.N, self.draw_period), replace=True)]
        if self.estimation_method == 'cholesky':
            cov_matrix = np.cov(residuals.T)
            chol_matrix = np.linalg.cholesky(cov_matrix)
            residual_draws = np.random.randn(self.N, self.draw_period, residuals.shape[1]) @ chol_matrix.T

        # Simulate shocks using numba
        self.shocks_sim_draws = np.zeros_like(residual_draws)
        construct_var_shocks(
            N=self.N, 
            draw_period=self.draw_period, 
            shocks_sim_draws=self.shocks_sim_draws,
            lags=lags,
            intercept=intercept, 
            coefs=coefs, 
            residual_draws=residual_draws
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

        # Try aggregating shocks for exchange rates, if not possible, set to zero
        try:
            exr_eur_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -6], axis=2)
        except:
            exr_eur_shocks = np.zeros((self.N, self.stochastic_period))
        try:
            exr_usd_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -5], axis=2)
        except:
            exr_usd_shocks = np.zeros((self.N, self.stochastic_period))

        ## Aggregate shocks for short-term interest rate, nominal GDP growth, and primary balance
        short_term_interest_rate_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -4], axis=2)
        nominal_gdp_growth_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -2], axis=2)
        primary_balance_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -1], axis=2)

        ## Aggregate shocks for long-term interest rate
        # Calculate the average maturity in quarters 
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

        # If stochastic projection starts before adjustment end, set pb shock to zero during adjustment period
        post_adjustment_period = self.projection_period - self.adjustment_end - 1
        if self.stochastic_period > post_adjustment_period:
            self.shocks_sim[:, 3, : -post_adjustment_period] = 0

    def _aggregate_shocks_annual(self):
        """
        Save annual into shock matrix, aggregate long term interest rate shocks.
        """
        # Reshape the shocks
        self.shocks_sim_grouped = self.shocks_sim_draws.reshape((self.N, self.stochastic_period, self.num_variables))

        # Try retrieving shocks for exchange rates, if not possible, set to zero
        try:
            exr_eur_shocks = self.shocks_sim_grouped[:, :, -6]
        except:
            exr_eur_shocks = np.zeros((self.N, self.stochastic_period))
        try:
            exr_usd_shocks = self.shocks_sim_grouped[:, :, -5]
        except:
            exr_usd_shocks = np.zeros((self.N, self.stochastic_period))

        ## Retrieve shocks for short-term interest rate, nominal GDP growth, and primary balance
        short_term_interest_rate_shocks = self.shocks_sim_grouped[:, :, -4]
        nominal_gdp_growth_shocks = self.shocks_sim_grouped[:, :, -2]
        primary_balance_shocks = self.shocks_sim_grouped[:, :, -1]

        ## Aggregate shocks for long-term interest rate
        # Calculate the average maturity in years 
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

        # If stochastic projection starts before adjustment end, set pb shock to zero during adjustment period
        post_adjustment_period = self.projection_period - self.adjustment_end - 1
        if self.stochastic_period > post_adjustment_period:
            self.shocks_sim[:, 3, : -post_adjustment_period] = 0

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

        # Save the resulting self.variables
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

    def fanchart(self, var='d', save_as=False, show=True):
        """
        Plot a fanchart of the simulated debt-to-GDP ratio. And save the figure if save_as is specified.
        """
        # Set stochastic variable
        sim_var = getattr(self, f'{var}_sim')
        bl_var = getattr(self, f'{var}')

        # Check if first values of baseline and simulation are equal, if not, simulate
        if not np.isclose(sim_var[0, 0], bl_var[self.stochastic_start-1]): 
            self.simulate()

        # Calculate the percentiles
        self.pcts_dict = {}
        for pct in np.arange(10, 100, 10):
            self.pcts_dict[pct] = np.percentile(sim_var, pct, axis=0)[:self.stochastic_period+1]

        # Create array of years and baseline debt-to-GDP ratio
        years = np.arange(self.start_year, self.end_year+1)

        # Set color pallette
        fanchart_palette = sns.color_palette("Blues")

        # Plot the results using fill between
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[10], self.pcts_dict[90], color=fanchart_palette[0], label='10th-90th percentile')
        ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[20], self.pcts_dict[80], color=fanchart_palette[1], label='20th-80th percentile')
        ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[30], self.pcts_dict[70], color=fanchart_palette[2], label='30th-70th percentile')
        ax.fill_between(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[40], self.pcts_dict[60], color=fanchart_palette[3], label='40th-60th percentile')
        ax.plot(years[self.stochastic_start-1:self.stochastic_end+1], self.pcts_dict[50], alpha=1, ls='-', color='black', label='Median')
        ax.plot(years, bl_var, ls='--', color='red', label='Baseline')

        # Plot layout
        ax.legend(loc='best')
        ax.xaxis.grid(False)
        ylabel = 'Debt (percent of GDP)' if var == 'd' else var
        ax.set_ylabel(ylabel)
        ax.set_title(f'{self.stochastic_period}-year fanchart for {self.country} (adjustment {self.adjustment_start_year}-{self.adjustment_end_year})')

        # Saveplot data in a dataframe if self.save_df is specified
        self.df_fanchart = pd.DataFrame({'year': years, 'baseline': bl_var})
        for pct in self.pcts_dict:
            df_pct = pd.DataFrame({'year': years[self.stochastic_start-1:self.stochastic_end+1], f'p{pct}': 
            self.pcts_dict[pct]})
            self.df_fanchart = self.df_fanchart.merge(df_pct, on='year', how='left')

        # Save plot if save_as is specified
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
        
        # Do not show plot if show is False
        if show == False:
            plt.close()

# ========================================================================================= #
#                              STOCHASTIC OPTIMIZATION METHODS                              # 
# ========================================================================================= #

    def find_spb_stochastic(self, prob_target=0.3, bounds=(-10, 10), print_update=False):
        """
        Find the structural primary balance that ensures the probability of the debt-to-GDP ratio exploding is equal to prob_target.
        """
        # Set parameters
        self.print_update = print_update
        
        if not hasattr(self, 'edp_steps'):
           self.edp_steps = None

        if not hasattr(self, 'deficit_resilience_steps'):
           self.deficit_resilience_steps = None

        if not hasattr(self, 'post_spb_steps'):
            self.post_spb_steps = None

        #if self.country in ['DNK']: bounds = (-6,0) # Denmark's target function has large local minima for high values
        
        self.stochastic_optimization_dict = {}

        # Set initial adjustment steps if predefined, this is needed for optimization of only selected steps
        if hasattr(self, 'predefined_spb_steps'):
            initital_spb_steps = np.nan_to_num(self.predefined_spb_steps)
        else:
            initital_spb_steps = None

        # Initial projection
        self.project(
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_spb_steps=self.post_spb_steps,
            spb_steps=initital_spb_steps,
            scenario=None
            )
        
        # Optimize for both debt decline and debt remaining under 60 and choose the lower SPB
        self.spb_target = self._stochastic_optimization(prob_target=prob_target, bounds=bounds)

        # Project with optimal spb
        self.project(
            spb_target=self.spb_target, 
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_spb_steps=self.post_spb_steps,
            scenario=None
            )

        return self.spb_target
    
    def _stochastic_optimization(self, prob_target, bounds):
        """
        Optimizes for SPB that ensures debt remains below 60% with probability prob_target.
        """
        # Initital simulation
        self.simulate()

        # Set parameters
        self.prob_target = prob_target
        self.spb_bounds = bounds
        self.stochastic_optimization_dict = {}

        # Optimize _target_pb to find b_target that ensures prob_debt_above_60 or prob_debt_explodes == prob_target
        self.spb_target = minimize_scalar(self._target_spb_decline, method='bounded', bounds=self.spb_bounds).x
        
        # Store results in a dataframe
        self.df_stochastic_optimization = pd.DataFrame(self.stochastic_optimization_dict).T
        
        return self.spb_target

    def _target_spb_decline(self, spb_target):
        """
        Returns zero if primary balance ensures prop_debt_explodes == prob_target or prob_debt_above_60 == prob_target.
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
        self.prob_debt_explodes()
        self.prob_debt_above_60()

        # Print spb_target for each iteration and clean print after
        if self.print_update:
            print(
                f'spb: {spb_target:.2f}, prob_debt_explodes: {self.prob_explodes:.2f}, prob_debt_above_60: {self.prob_above_60:.2f}', 
                end='\r'
            )
        self.stochastic_optimization_dict[spb_target] = {}
        self.stochastic_optimization_dict[spb_target]['prob_debt_explodes'] = self.prob_explodes
        self.stochastic_optimization_dict[spb_target]['prob_debt_above_60'] = self.prob_above_60

        # Optimize for more probable target    
        min_prob = np.min([self.prob_explodes, self.prob_above_60])
        
        # Penalty term to avoid local minima at upper bound
        penalty = np.max([0,spb_target/10]) if np.isclose(min_prob, 0) else 0
        
        return np.abs(min_prob - self.prob_target) + penalty

    def prob_debt_explodes(self):
        """
        Calculate the probability of the debt-to-GDP ratio exploding.
        """
        # Call the Numba JIT function with converted self variables
        self.prob_explodes = prob_debt_explodes_jit(
            N=self.N, 
            d_sim=self.d_sim, 
            )
        return self.prob_explodes
    
    def prob_debt_above_60(self):
        """
        Calculate the probability of the debt-to-GDP ratio exceeding 60 in 2038/2041.
        """
        # Call the Numba JIT function with converted self variables
        self.prob_above_60 = prob_debt_above_60_jit(
            N=self.N, 
            d_sim=self.d_sim, 
            )
        return self.prob_above_60
    
    def plot_target_func(self):
        """
        Plot the target function for the stochastic optimization.
        """
        results = {}
        for x in np.linspace(self.spb_bounds[0], self.spb_bounds[1], 100):
            y = self._target_spb_decline(spb_target=x)
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
                         print_results=True,
                         save_df=False):
        """
        Find the structural primary balance that meets all criteria after deficit has been brought below 3% and debt safeguard is satisfied.
        """       
        # Initiate spb_target and dataframe dictionary
        self.save_df = save_df
        self.spb_target_dict = {}
        self.pb_target_dict = {}
        self.binding_parameter_dict = {}
        if self.save_df: 
            self.df_dict = {}

        # Run DSA and deficit criteria and project toughest under baseline assumptions
        self.project(spb_target=None, edp_steps=None) # clear projection
        self._run_dsa()
        self._get_binding()

        # Apply EDP
        if edp: 
            self._apply_edp()
        elif self.ob[self.adjustment_start - 1] < -3:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 1
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 2
        
        # Apply debt safeguard
        if debt_safeguard: 
            self._apply_debt_safeguard()

        # Apply deficit resilience
        if deficit_resilience: 
            self._apply_deficit_resilience()

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
        
        # Print results
        if print_results: 
            self._print_results_tables(edp, debt_safeguard, deficit_resilience)
        
        # Save dataframe
        if self.save_df: 
            self.df_dict['binding'] = self.df(all=True)

    def _run_dsa(self, criterion='all'):
        """
        Run DSA for given criterion.
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
                except:
                    raise

            # Run stochastic scenario, skip if not possible due to lack of data
            try: 
                self.find_spb_stochastic()
                self.spb_target_dict['stochastic'] = self.spb_bca[self.adjustment_end]
                self.pb_target_dict['stochastic'] = self.pb[self.adjustment_end]
                if self.save_df: 
                    self.df_dict['stochastic'] = self.df(all=True)
            except:
                pass

        # If specific criterion given for EDP optimization, run only one optimization
        else:
            if criterion in deterministic_criteria_list:
                self.find_spb_deterministic(criterion=criterion)

            elif criterion == 'stochastic':
                self.find_spb_stochastic()

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
            print(self.edp_steps)
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
        Check if Council version of debt safeguard is binding in binding scenario and apply if it is.
        """
        # Define steps for debt safeguard
        self.edp_steps[:self.edp_period] = self.spb_steps[:self.edp_period]

        if hasattr(self, 'predefined_spb_steps'):
            debt_safeguard_start = max(self.adjustment_start + len(self.predefined_spb_steps) - 1, self.edp_end + 1)
        else:
            debt_safeguard_start = self.edp_end + 1

        # Debt safeguard binding for countries with high debt and debt decline below 4 or 2 % for 4 years after edp
        debt_safeguard_decline = 1 if self.d[self.adjustment_start - 1] > 90 else 0.5
        debt_safeguard_criterion = (self.d[debt_safeguard_start] - self.d[self.adjustment_end] 
                                    < debt_safeguard_decline * (self.adjustment_end - debt_safeguard_start))
        
        if (self.d[self.adjustment_start-1] >= 60 
            and debt_safeguard_criterion):
            
            # Call deterministic optimizer to find SPB target for debt safeguard
            self.spb_debt_safeguard_target = self.find_spb_deterministic(criterion='debt_safeguard')
            
            # If debt safeguard SPB target is higher than DSA target, save debt safeguard target
            if self.spb_debt_safeguard_target > self.binding_spb_target + 1e-8: # 1e-8 tolerance for floating point errors
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
                
    def _print_results_tables(self, edp=True, debt_safeguard=True, deficit_resilience=True):
        """
        Print two ascii tables side by side and one table underneath, ensuring the lower table is as wide as the top two combined.
        """
        # Prepare data for the tables
        model_params = {
            'country': 
            self.country,
            'adjustment period': 
            self.adjustment_period,
            'adjustment start': 
            self.adjustment_start_year,
            'shock frequency': 
            self.shock_frequency,
            'stochastic period': f"{self.stochastic_start_year}-{self.stochastic_start_year + self.stochastic_period}",
            'estimation': f"{self.estimation} {'' if self.estimation == 'normal' else '(' + self.estimation_method + ')'}",
            'bond level data': 
            self.bond_data,
            'safeguards': f"{'EDP,' if edp else ''} {'debt,' if debt_safeguard else ''} {'deficit_resilience' if deficit_resilience else ''}"
        }
        spb_targets = {key: f"{value:.3f}" for key, value in self.spb_target_dict.items()}
        binding_params = {
            key: (
                np.array2string(value, precision=3, separator=', ').replace('[', '').replace(']', '') if isinstance(value, np.ndarray)
                else f'{value:.3f}' if isinstance(value, float)
                else str(value)
            ) for key, value in self.binding_parameter_dict.items() if key != 'post_spb_steps'
        }

        # Helper function to split steps into chunks and name them based on ranges
        def split_steps(key, steps, chunk_size=7):
            for i in range(0, len(steps), chunk_size):
                part_key = f"{key} ({i + 1}-{min(i + chunk_size, len(steps))})"
                binding_params[part_key] = np.array2string(steps[i:i + chunk_size], precision=3, separator=', ').replace('[', '').replace(']', '')

        # Check for adjustment period and split steps if necessary
        if self.adjustment_period > 7:
            del binding_params['spb_steps']
            split_steps('spb_steps', self.spb_steps)
            if 'edp_steps' in self.binding_parameter_dict:
                del binding_params['edp_steps']
                split_steps('edp_steps', self.edp_steps)
            if 'deficit_resilience_steps' in self.binding_parameter_dict:
                del binding_params['deficit_resilience_steps']
                split_steps('deficit_resilience_steps', self.deficit_resilience_steps)
            if 'net_expenditure_growth' in self.binding_parameter_dict:
                del binding_params['net_expenditure_growth']
                split_steps('net_expenditure_growth', self.net_expenditure_growth[self.adjustment_start:self.adjustment_end+1])

        # Convert all values to string with proper formatting
        formatted_model_params = {key: str(value) for key, value in model_params.items()}
        formatted_spb_targets = {key: str(value) for key, value in spb_targets.items()}
        formatted_binding_params = {key: str(value) for key, value in binding_params.items()}

        # Function to print formatted table
        def print_table(title, data, total_width=None):
            max_key_len = max(len(key) for key in data.keys())
            max_val_len = total_width - max_key_len - 2 or max(len(value) for value in data.values()) 
            table_width = max_key_len + max_val_len + 2
            total_width = total_width or table_width

            print(f"{title.center(total_width)}")
            print("=" * total_width)
            for key, value in data.items():
                line = f"{key.ljust(max_key_len)}  {value.rjust(max_val_len)}"
                print(line.ljust(total_width))
            print("=" * total_width)
            print()

        # Function to print two tables side by side
        def print_two_tables_side_by_side(title1, data1, title2, data2):
            max_key_len1 = max(len(key) for key in data1.keys())
            max_val_len1 = max(len(value) for value in data1.values())
            max_key_len2 = max(len(key) for key in data2.keys())
            max_val_len2 = max(len(value) for value in data2.values())
            total_width1 = max_key_len1 + max_val_len1 + 2
            total_width2 = max_key_len2 + max_val_len2 + 2
            total_width = total_width1 + total_width2 + 5

            # Print titles
            print(f"{title1.center(total_width1)}{' ' * 5}{title2.center(total_width2)}")
            print(f"{'=' * total_width1}{' ' * 5}{'=' * total_width2}")

            # Print data rows side by side
            keys1 = list(data1.keys())
            keys2 = list(data2.keys())
            max_rows = max(len(keys1), len(keys2))
            for i in range(max_rows):
                line1 = line2 = ""
                if i < len(keys1):
                    key1 = keys1[i]
                    value1 = data1[key1]
                    line1 = f"{key1.ljust(max_key_len1)}  {value1.rjust(max_val_len1)}"
                else:
                    line1 = " " * total_width1
                if i < len(keys2):
                    key2 = keys2[i]
                    value2 = data2[key2]
                    line2 = f"{key2.ljust(max_key_len2)}  {value2.rjust(max_val_len2)}"
                else:
                    line2 = " " * total_width2
                print(f"{line1}{' ' * 5}{line2}")
            print(f"{'=' * total_width1}{' ' * 5}{'=' * total_width2}")
            print()
            return total_width

        # Print the first two tables side by side and get the total width
        total_width = print_two_tables_side_by_side('Model Params', formatted_model_params, 'SPB Targets', formatted_spb_targets)

        # Print the third table underneath with the combined width of the first two tables
        print_table('Binding Params', formatted_binding_params, total_width)

    def find_deficit_prob(self):
        """
        Find the probability of the deficit exceeding 3% in each adjustment period for binding SPB path.
        """
        # Initial projection
        self.project(
            spb_target=self.binding_spb_target, 
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            scenario=self.scenario
            )

        # Set stochastic period to adjustment period
        self.stochastic_start = self.adjustment_start
        self.stochastic_end = self.adjustment_end + 1
        self.stochastic_period = self.adjustment_period + 1

        if self.shock_frequency == 'quarterly':
            self.draw_period = self.stochastic_period * 4 
        else:
            self.draw_period = self.stochastic_period

        # Set exchange rate and primary balance shock to zero
        self.df_shocks[['EXR_EUR', 'EXR_USD', 'PRIMARY_BALANCE']] = 0

        # Draw quarterly shocks
        self._draw_shocks_normal()

        # Aggregate quarterly shocks to annual shocks
        if self.shock_frequency == 'quarterly': 
            self._aggregate_shocks_quarterly()
        else:
            self._aggregate_shocks_annual()

        # Replace PB shock (self.shocks_sim[:,3]) with growth shock (self.shocks_sim[:,2]) times budget balance multiplier
        self.shocks_sim[:, 3] = self.budget_balance_elasticity * self.shocks_sim[:, 2]
        
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
def construct_var_shocks(N, draw_period, shocks_sim_draws, lags, intercept, coefs, residual_draws):
    """
    Simulate the shocks for the baseline variables.
    """
    for n in range(N):
        for t in range(1, draw_period):
            shock = intercept.copy()
            for lag in range(1, lags + 1):
                if t - lag >= 0:
                    shock += vecmatmul(shocks_sim_draws[n, t - lag, :], coefs[lag - 1])
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
def prob_debt_explodes_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio exploding.
    """
    prob_explodes = 0
    for n in range(N):
        if d_sim[n, 0] < d_sim[n, -1]:
            prob_explodes += 1
    return prob_explodes / N

@jit(nopython=True)
def prob_debt_above_60_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio exceeding 60
    """
    prob_debt_above_60 = 0
    for n in range(N):
        if 60 < d_sim[n, -1]:
            prob_debt_above_60 += 1
    return prob_debt_above_60 / N

@jit(nopython=True)
def simulate_deficit_jit(N, stochastic_period, pb_sim, iir_sim, ng_sim, d_sim, ob_sim):
    """
    Simulate the fiscal balance ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, stochastic_period+1):
            ob_sim[n, t] = pb_sim[n, t] - iir_sim[n, t] / 100 / (1 + ng_sim[n, t] / 100) * d_sim[n, t-1]