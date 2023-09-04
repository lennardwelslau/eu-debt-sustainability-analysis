#=========================================================================================#
#          Economic and Debt Sustainability Model (EcDsaModel) - Stochastic SubClass      #
#=========================================================================================#
# This Python file defines a stochastic model subclass named "EcStochasticModel" which is built upon the base class "EcDsaModel".
# The subclass is designed to simulate a stochastic debt sustainability analysis (DSA) model, incorporating randomness and uncertainty in the model's inputs.
# The model takes into account various economic shocks and their impacts on key macroeconomic variables, such as exchange rates, interest rates, GDP growth,
# and primary balances. The simulation results provide insights into the potential variability of the debt-to-GDP ratio over time.
#
# The simulation process involves drawing quarterly shocks from a multivariate normal distribution, aggregating these shocks into annual shocks,
# combining the shocks with baseline variables, and simulating the debt-to-GDP ratio using a specific formula. The subclass also provides methods to plot
# fan charts showing the distribution of simulated debt-to-GDP ratios, calculate the probability of the debt ratio exploding, and perform stochastic optimization
# to find the structural primary balance that ensures a specified probability of the debt ratio exploding.
#
# The code uses various libraries such as NumPy, pandas, matplotlib, seaborn, scipy, and numba for efficient numerical computations and visualization.
#
# To use the "EcStochasticModel" subclass, create an instance with appropriate parameters like the ISO country code, start year, adjustment period,
# and adjustment start year. You can then call methods like "simulate" to perform simulations, "fanchart" to plot fan charts, "prob_debt_explodes" to
# calculate the probability of the debt exploding, "find_spb_stochastic" to optimize the structural primary balance based on a specified probability target,
# "find_spb_deficit" to find the binding SPB scenario after deficit has been brought below 3%, or "find_deficit_prob" to find the probability of an excessive
# deficit during the adjustment period.
#
# For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org
#
# Author: Lennard Welslau
# Date: 31/08/2023
#
#=========================================================================================#


# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')
from scipy.optimize import minimize_scalar
from numba import jit
from EcDsaModelClass import EcDsaModel

class EcStochasticModel(EcDsaModel):

    #----------------------------#
    #--- INIITIALIZE SUBCLASS ---# 
    #----------------------------#
    def __init__(self, 
                country, # ISO code of country
                start_year=2022, # start year of projection, first year is baseline value
                end_year=2053, # end year of projection
                adjustment_period=4, # number of years for linear spb_bcoa adjustment
                adjustment_start=2025 # start year of linear spb_bcoa adjustment
                ):
        super().__init__(country, start_year, end_year, adjustment_period, adjustment_start)
        
        # Set option for nfpc
        if adjustment_period == 'nfpc':
            self.nfpc = True
            adjustment_period = 4
            adjustment_start = 2025
        else:
            self.nfpc = False
        
        self.outlier_threshold = 3
        start_year_stochastic = adjustment_start - 1 + adjustment_period
        self.T_stochastic = self.end_year - start_year_stochastic
        self.num_quarters = self.T_stochastic * 4

        self._get_shock_data()

    def _get_shock_data(self):

        # Select country shock data and get number of variables
        self.df_shocks = pd.read_excel('../data/InputData/shock_data.xlsx', sheet_name=self.country, index_col=0).T
        self.num_variables = self.df_shocks.shape[1]
        
        # Adjust outliers defined by mean + outlier_threshold * standard deviation to the threshold
        self.df_shocks = self.df_shocks.clip(
            lower=self.df_shocks.mean() - self.outlier_threshold * self.df_shocks.std(), 
            upper=self.df_shocks.mean() + self.outlier_threshold * self.df_shocks.std(), 
            axis=1)
        
    #--------------------------#
    #--- SIMULATION METHODS ---# 
    #--------------------------#     
    def simulate(self, N=200000):
        """
        Simulate the stochastic model.
        """
        # Set number of simulations
        self.N = N

        # Draw quarterly shocks
        self._draw_shocks_quarterly()

        # Aggregate quarterly shocks to annual shocks
        self._aggregate_shocks()
        
         # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()

    def _draw_shocks_quarterly(self):
        """
        Draw quarterly shocks from a multivariate normal distribution.

        This method calculates the covariance matrix of the shock DataFrame and then draws N samples of quarterly shocks
        from a multivariate normal distribution with mean 0 and the calculated covariance matrix.

        It reshapes the shocks into a 4-dimensional array of shape (N, T, 4, num_variables), where N is the number of simulations,
        T is the number of years, 4 represents the four variables, and num_variables is the number of shock variables.
        """

        # Calculate the covariance matrix of the shock DataFrame
        self.cov_matrix = self.df_shocks.cov()

        # Draw samples of quarterly shocks from a multivariate normal distribution
        self.shocks_sim_quarterly = np.random.multivariate_normal(
            mean=np.zeros(self.cov_matrix.shape[0]),
            cov=self.cov_matrix,
            size=(self.N, self.num_quarters)
        )

    def _aggregate_shocks(self):
        """
        Aggregate quarterly shocks to annual shocks for specific variables.

        This method aggregates the shocks for exchange rate, short-term interest rate, nominal GDP growth, and primary balance
        from quarterly to annual shocks as the sum over four quarters. For long-term interest rate, it aggregates shocks over all past quarters up to the current year and m_res_lt.
        """

        # Reshape the shocks to sum over the four quarters
        self.shocks_sim_quarterly_grouped = self.shocks_sim_quarterly.reshape((self.N, self.T_stochastic, 4, self.num_variables))

        try:
            # Try aggregating shocks for exchange rate, if not possible, set to zero
            exchange_rate_shocks = np.sum(self.shocks_sim_quarterly_grouped[:, :, :, -5], axis=2)
        except:
            exchange_rate_shocks = np.zeros(self.N, self.T_stochastic)

        ## Aggregate shocks for short-term interest rate, nominal GDP growth, and primary balance
        short_term_interest_rate_shocks = np.sum(self.shocks_sim_quarterly_grouped[:, :, :, -4], axis=2)
        nominal_gdp_growth_shocks = np.sum(self.shocks_sim_quarterly_grouped[:, :, :, -2], axis=2)
        primary_balance_shocks = np.sum(self.shocks_sim_quarterly_grouped[:, :, :, -1], axis=2)

        ## Aggregate shocks for long-term interest rate
        # Calculate the average maturity in quarters 
        maturity_quarters = int(np.round(self.m_res_lt * 4))

        # Initialize an array to store the aggregated shocks for the long-term interest rate
        self.long_term_interest_rate_shocks = np.zeros((self.N, self.T_stochastic))

        # Iterate over each year
        for t in range(1, self.T_stochastic+1):
            q = t * 4
            # Calculate the weight for each quarter based on the current year and m_res_lt
            weight = t / self.m_res_lt

            # Determine the number of quarters to sum based on the current year and m_res_lt
            q_to_sum = np.min([q, maturity_quarters])

            # Sum the shocks (N, T, num_quarters, num_variables) across the selected quarters
            aggregated_shocks = weight * np.sum(
                self.shocks_sim_quarterly[:, q-q_to_sum-1 : q+1, -3], axis=(1))

            # Assign the aggregated shocks to the corresponding year
            self.long_term_interest_rate_shocks[:, t-1] = aggregated_shocks

        # Calculate the weighted average of short and long-term interest using share_st
        interest_rate_shocks = self.share_st * short_term_interest_rate_shocks + (1 - self.share_st) * self.long_term_interest_rate_shocks

        # Stack all shocks in a matrix
        self.shocks_sim = np.stack([exchange_rate_shocks, interest_rate_shocks, nominal_gdp_growth_shocks, primary_balance_shocks], axis=2)
        self.shocks_sim = np.transpose(self.shocks_sim, (0, 2, 1))

    def _combine_shocks_baseline(self):
        """
        Combine shocks with the respective baseline variables and set starting values for simulation.
        """

        # Create arrays to store the simulated variables
        d_sim = np.zeros([self.N, self.T_stochastic+1])  # Debt to GDP ratio
        exr_sim = np.zeros([self.N, self.T_stochastic+1])  # Exchange rate
        iir_sim = np.zeros([self.N, self.T_stochastic+1])  # Implicit interest rate
        ng_sim = np.zeros([self.N, self.T_stochastic+1])  # Nominal GDP growth
        pb_sim = np.zeros([self.N, self.T_stochastic+1])  # Primary balance 
        sf_sim = np.zeros([self.N, self.T_stochastic+1])  # Stock flow adjustment

        # Call the Numba JIT function with converted self variables
        combine_shocks_baseline_jit(
            N=self.N, T_stochastic=self.T_stochastic, shocks_sim=self.shocks_sim, exr=self.exr, 
            iir=self.iir, ng=self.ng, pb=self.pb, sf=self.sf, d=self.d, d_sim=d_sim, exr_sim=exr_sim, 
            iir_sim=iir_sim, ng_sim=ng_sim, pb_sim=pb_sim, sf_sim=sf_sim
            )

        # Save the resulting self.variables
        self.d_sim = d_sim
        self.exr_sim = exr_sim
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
            N=self.N, T_stochastic=self.T_stochastic, share_eur=self.share_eur, d_sim=self.d_sim, iir_sim=self.iir_sim, 
            ng_sim=self.ng_sim, exr_sim=self.exr_sim, pb_sim=self.pb_sim, sf_sim=self.sf_sim)


    #-------------------------#
    #--- AUXILIARY METHODS ---# 
    #-------------------------#    
    def fanchart(self, save_as=False, save_df=False, show=True, variable='d', periods=5):
        """
        Plot a fanchart of the simulated debt-to-GDP ratio. And save the figure if save_as is specified.
        """

        # Set stochastic variable
        sim_var = eval(f'self.{variable}_sim')
        bl_var = eval(f'self.{variable}')

        # Calculate the percentiles
        self.pcts_dict = {}
        for pct in np.arange(10, 100, 10):
            self.pcts_dict[pct] = np.percentile(sim_var, pct, axis=0)[:periods]

        # Create array of years and baseline debt-to-GDP ratio
        years = np.arange(self.start_year, self.end_year+1)

        # Set color pallette
        sns.set_palette(sns.color_palette("Blues"))

        # Plot the results using fill between
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods], self.pcts_dict[10], self.pcts_dict[90], label='10th-90th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods], self.pcts_dict[20], self.pcts_dict[80], label='20th-80th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods], self.pcts_dict[30], self.pcts_dict[70], label='30th-70th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods], self.pcts_dict[40], self.pcts_dict[60], label='40th-60th percentile')
        ax.plot(years[-self.T_stochastic-1:-self.T_stochastic-1+periods], self.pcts_dict[50], alpha=1, ls='-', color='black', label='median')
        ax.plot(years, bl_var, ls='--', color='red', label='Baseline')

        # Plot layout
        ax.legend(loc='upper left')
        ax.xaxis.grid(False)
        ax.set_ylabel(variable)
        ax.set_title(f'{self.country}')

        # Saveplot data in a dataframe if save_df is specified
        if save_df:
            self.df_fanchart = pd.DataFrame({'year': years, 'baseline': bl_var})
            for pct in self.pcts_dict:
                df_pct = pd.DataFrame({'year': years[-self.T_stochastic-1:-self.T_stochastic-1+periods], f'p{pct}': self.pcts_dict[pct]})
                self.df_fanchart = self.df_fanchart.merge(df_pct, on='year', how='left')

        # Save plot if save_as is specified
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
        
        # Do not show plot if show is False
        if show == False:
            plt.close()

    #---------------------------------------#
    #--- STOCHASTIC OPTIMIZATION METHODS ---# 
    #---------------------------------------#     
    def find_spb_stochastic(self, prob_target=0.3, bounds=(-3, 5), print_update=False):
        """
        Find the structural primary balance that ensures the probability of the debt-to-GDP ratio exploding is equal to prob_target.
        """
            
        self.print_update = print_update
        if not hasattr(self, 'spb_deficit_period'):
           self.spb_deficit_period = 0

        # Initial projection
        self.project(scenario=None, spb_deficit_period=self.spb_deficit_period)

        # If debt<60 and deficit<3, find spb_deterministic that ensures deficit remains <3
        if (
            self.fb[self.adjustment_start - 1] >= -3
            ) and (
            self.d[self.adjustment_start - 1] <= 60
            ):
            raise Exception("Only 'deficit_reduction' criterion for countries with debt ratio < 60% and deficit < 3%")

        
        # If debt <= 60, optimize for debt remaining under 60
        elif self.d[self.adjustment_start - 1] <= 60: 
        
            # Initital simulation
            self.simulate()

            # Set parameters
            self.find_spb_dict = {}
            self.prob_target = prob_target
            self.pb_bounds = bounds

            # Optimize _target_pb to find b_target that ensures prob_debt_above_60 == prob_target
            self.spb_target = minimize_scalar(self._target_spb_above_60, bounds=self.pb_bounds).x
            
            # Store results in a dataframe
            self.df_find_spb = pd.DataFrame(self.find_spb_dict, index=['prob_debt_explodes']).T.reset_index(names='spb_target')
            
            # Project with optimal spb
            self.project(spb_target=self.spb_target, scenario=None, spb_deficit_period=self.spb_deficit_period)

        # If debt > 60, optimize for debt decline
        elif self.d[self.adjustment_start - 1] > 60: 
        
            # Initital simulation
            self.simulate()

            # Set parameters
            self.find_spb_dict = {}
            self.prob_target = prob_target
            self.pb_bounds = bounds

            # Optimize _target_pb to find b_target that ensures prob_debt_explodes == prob_target
            self.spb_target = minimize_scalar(self._target_spb_explodes, bounds=self.pb_bounds).x
            
            # Store results in a dataframe
            self.df_find_spb = pd.DataFrame(self.find_spb_dict, index=['prob_debt_explodes']).T.reset_index(names='spb_target')
            
            # Project with optimal spb
            self.project(spb_target=self.spb_target, scenario=None, spb_deficit_period=self.spb_deficit_period)

        return self.spb_target
    
    def _target_spb_explodes(self, spb_target):
        """
        Returns zero if primary balance ensures prop_debt_explodes == prob_target.
        """

        # Simulate the debt-to-GDP ratio with the given primary balance target
        self.project(spb_target=spb_target, scenario=None, spb_deficit_period=self.spb_deficit_period)

        # Combine shocks with new baseline
        self._combine_shocks_baseline()

        # Simulate debt ratio and calculate probability of debt exploding or exceeding 60
        self._simulate_debt()
        self.prob_debt_explodes()

        # Print spb_target for each iteration and clean print after
        if self.print_update:
            print(f'spb: {spb_target:.2f}, pb: {self.pb[self.adjustment_end]:.2f}, prob_debt_explodes: {self.prob_explodes:.2f}', end='\r')
        self.find_spb_dict[spb_target] = self.prob_explodes

        return np.abs(self.prob_explodes - self.prob_target)

    def _target_spb_above_60(self, spb_target):
        """
        Returns zero if primary balance ensures prop_debt_above_60 == prob_target.
        """

        # Simulate the debt-to-GDP ratio with the given primary balance target
        self.project(spb_target=spb_target, scenario=None, spb_deficit_period=self.spb_deficit_period)

        # Combine shocks with new baseline
        self._combine_shocks_baseline()

        # Simulate debt ratio and calculate probability of debt exploding or exceeding 60
        self._simulate_debt()
        self.prob_debt_above_60()

        # Print spb_target for each iteration and clean print after
        if self.print_update:
            print(f'spb: {spb_target:.2f}, pb: {self.pb[self.adjustment_end]:.2f}, prob_debt_explodes: {self.prob_above_60:.2f}', end='\r')
        self.find_spb_dict[spb_target] = self.prob_above_60

        return np.abs(self.prob_above_60 - self.prob_target)    

    def prob_debt_explodes(self):
        """
        Calculate the probability of the debt-to-GDP ratio exploding.
        """

        # Call the Numba JIT function with converted self variables
        self.prob_explodes = prob_debt_explodes_jit(
            N=self.N, d_sim=self.d_sim
            )
        return self.prob_explodes
    
    def prob_debt_above_60(self):
        """
        Calculate the probability of the debt-to-GDP ratio exceeding 60 in 2038/2041.
        """

        # Call the Numba JIT function with converted self variables
        self.prob_above_60 = prob_debt_above_60_jit(
            N=self.N, d_sim=self.d_sim
            )
        return self.prob_above_60
    
    #-------------------------#
    #--- DEFICIT OPTIMIZER ---#
    #-------------------------#

    def find_spb_deficit(self):
        """
        Find the structural primary balance that ensures a decline in the debt ratio after reaching deficit below 3%.
        """
        
        # Find spb_target
        self.project(spb_deficit_period = 0)

        # Raise error if deficit not excessive
        if np.all(self.fb[self.adjustment_start:self.adjustment_end+1] >= -3):
            raise Exception("Deficit not excessive")
        
        # If deficit excessive, increase 0.5 adjustment periods until deficit below 3%
        elif np.any(self.fb[self.adjustment_start:self.adjustment_end+1] < -3):
            self.spb_deficit_period = 0
            while np.any(self.fb[self.adjustment_start+self.spb_deficit_period: self.adjustment_end+1] < -3):
                self.spb_deficit_period += 1
                self.project(spb_deficit_period=self.spb_deficit_period, scenario=None)
            print(f'Deficit periods: {self.spb_deficit_period}')

            # If spb_deficit_period = adjusmtent_period, return spbd at adjustment_end
            if self.spb_deficit_period == self.adjustment_period:
                return self.spb_bcoa[self.adjustment_end]
                
            # If spb_deficit_period < adjustment_period, optimize spb
            else:
                return self._find_spb_post_deficit()

    def _find_spb_post_deficit(self):
        """
        Find the structural primary balance that mmets all scenario criteria after deficit has been brought below 3%.
        """
        # Initiate spb_target_dict
        spb_target_dict = {}

        # Find spb_target for adverse scenarios
        for scenario in ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']:
            spb_target = self.find_spb_deterministic(scenario=scenario, criterion='debt_decline')
            spb_target_dict[f'{scenario}'] = spb_target

        # Find spb_target for deficit_reduction
        spb_target = self.find_spb_deterministic(scenario='main_adjustment', criterion='deficit_reduction')
        spb_target_dict['deficit_reduction'] = spb_target
        
        # Find spb_target for debt safeguard, optimize adjustment of deficit procedure to meet debt_safeguard
        self.spb_deficit_step = 0.5
        while True:
            try:
                spb_target = self.find_spb_deterministic(scenario='main_adjustment', criterion='debt_safeguard_deficit')
                try:
                    self.spb_deficit_step -= 0.01
                    if self.spb_deficit_step < 0.5:
                        raise
                    print(f'Decreasing spb_deficit_step to {self.spb_deficit_step}')                        
                    spb_target = self.find_spb_deterministic(scenario='main_adjustment', criterion='debt_safeguard_deficit')
                except:
                    self.spb_deficit_step += 0.01
                    print(f'Final spb_deficit_step {self.spb_deficit_step}')                        
                    spb_target = self.find_spb_deterministic(scenario='main_adjustment', criterion='debt_safeguard_deficit')
                    break
            except:
                self.spb_deficit_step += 0.1
                print(f'Increasing spb_deficit_step to {self.spb_deficit_step}')
        spb_target_dict['debt_safeguard'] = spb_target
        self.spb_deficit_step = 0.5


        # Find spb_target for stochastic scenario
        try:
            spb_target = self.find_spb_stochastic()
            spb_target_dict['stochastic'] = spb_target
        except:
            pass

        # Safe and project with binding spb_target
        print(f'{self.country} spb target: {spb_target_dict}')
        spb_target = np.max(list(spb_target_dict.values()))
        self.project(spb_target=spb_target, scenario=None, spb_deficit_period=self.spb_deficit_period)

        return spb_target

    #---------------------------#
    #--- DEFICIT PROBABILITY ---#
    #---------------------------#

    def find_deficit_prob(self, spb_target, spb_deficit_period=0, spb_deficit_step=0.5, N=200000):
        """
        Find the probability of the deficit exceeding 3% in the adjustment period.
        """

        # Initial projection
        self.project(scenario=None, spb_target=spb_target, spb_deficit_period=spb_deficit_period, spb_deficit_step=spb_deficit_step)
        
        # Set stochastic period to include adjustment period
        self.T_stochastic = self.T - self.adjustment_start
        self.num_quarters = self.T_stochastic * 4

        # Set exchange rate and primary balance shock to zero
        #self.df_shocks[['exchange_rate', 'primary_balance']] = 0

        # Set number of draws
        self.N = N

        # Draw quarterly shocks
        self._draw_shocks_quarterly()

        # Aggregate quarterly shocks to annual shocks
        self._aggregate_shocks()

        # Replace PB shock (self.shocks_sim[:,3]) with growth shock (self.shocks_sim[:,2]) times budget balance multiplier
        self.shocks_sim[:, 3] = self.pb_elasticity * self.shocks_sim[:, 2]
        
        # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()
        
        # Simulate deficit
        self.fb_sim = np.zeros([self.N, self.T_stochastic+1])  # Debt to GDP ratio
        self.fb_sim[:, 0] = self.fb[-(self.T_stochastic+1)]
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
            N=self.N, T_stochastic=self.T_stochastic, pb_sim=self.pb_sim, iir_sim=self.iir_sim, ng_sim=self.ng_sim, d_sim=self.d_sim, fb_sim=self.fb_sim
            )
    
    def _prob_deficit(self):
        """
        Calculate the probability of the deficit exceeding 3% in two consecutive period or 3.5% in one period during adjustment.
        """
        prob_excessive_deficit = np.full(self.adjustment_period, 0, dtype=np.float64)
        for n in range(self.N):
            for i in range(self.adjustment_period):
                if -3.5 > self.fb_sim[n, i+1] or (-3 > self.fb_sim[n, i+1] and -3 > self.fb_sim[n, i+2]):
                    prob_excessive_deficit[i] += 1
        return prob_excessive_deficit / self.N  
  
#---------------------------------#
#--- NUMBA OPTIMIZED FUNCTIONS ---#
#---------------------------------#

@jit(nopython=True)
def combine_shocks_baseline_jit(N, T_stochastic, shocks_sim, exr, iir, ng, pb, sf, d, d_sim, exr_sim, iir_sim, ng_sim, pb_sim, sf_sim):
    """
    Add shocks to the baseline variables and set starting values for simulation.
    """
    # Add shocks to the baseline variables for period after start year
    for n in range(N):
        exr_sim[n, 1:] = exr[-T_stochastic:] + shocks_sim[n, 0] 
        iir_sim[n, 1:] = iir[-T_stochastic:] + shocks_sim[n, 1]
        ng_sim[n, 1:] = ng[-T_stochastic:] + shocks_sim[n, 2]
        pb_sim[n, 1:] = pb[-T_stochastic:] + shocks_sim[n, 3]
    
    # Set values for stock-flow adjustment 
    sf_sim[:, 1:] = sf[-T_stochastic:]

    # Set the starting values t0
    d_sim[:, 0] = d[-T_stochastic-1]
    exr_sim[:, 0] = exr[-T_stochastic-1]
    iir_sim[:, 0] = iir[-T_stochastic-1]
    ng_sim[:, 0] = ng[-T_stochastic-1]
    pb_sim[:, 0] = pb[-T_stochastic-1]


@jit(nopython=True)
def simulate_debt_jit(N, T_stochastic, share_eur, d_sim, iir_sim, ng_sim, exr_sim, pb_sim, sf_sim):
    """
    Simulate the debt-to-GDP ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, T_stochastic+1):
            d_sim[n, t] = share_eur * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) \
                        + (1 - share_eur) * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) \
                        * (exr_sim[n, t]) / (exr_sim[n, t-1]) - pb_sim[n, t] + sf_sim[n, t]
            
@jit(nopython=True)
def prob_debt_explodes_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio exploding.
    """
    prob_explodes = 0
    for n in range(N):
        if d_sim[n, 0] < d_sim[n, 5]:
            prob_explodes += 1
    return prob_explodes / N

@jit(nopython=True)
def prob_debt_above_60_jit(N, d_sim):
    """
    Calculate the probability of the debt-to-GDP ratio exceeding 60
    """
    prob_debt_above_60 = 0
    for n in range(N):
        if 60 < d_sim[n, 5]:
            prob_debt_above_60 += 1
    return prob_debt_above_60 / N

@jit(nopython=True)
def simulate_deficit_jit(N, T_stochastic, pb_sim, iir_sim, ng_sim, d_sim, fb_sim):
    """
    Simulate the fiscal balance ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, T_stochastic+1):
            fb_sim[n, t] = pb_sim[n, t] - iir_sim[n, t] / 100 / (1 + ng_sim[n, t] / 100) * d_sim[n, t-1]