#=========================================================================================#
#          European Commission Debt Sustainability Analysis - Stochastic Sub Class        #
#=========================================================================================#
# This Python file defines the stochastic model subclass named "EcStochasticModel" which is 
# built upon the base class "EcDsaModel". The subclass is designed to simulate a stochastic 
# debt sustainability analysis (DSA) model, incorporating randomness and uncertainty in the 
# model's inputs. The model takes into account shocks to the exchange rates, interest rates, 
# GDP growth, and the primary balances. 
#
# The model includes methods to perform simulations, plot fan charts, calculate the probability 
# of debt exploding, and to optimize the structural primary balance.
# 
# For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org
#
# Author: Lennard Welslau
# Updated: 2024-02-26
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
from DsaModelClass import DsaModel

class StochasticDsaModel(DsaModel):

    #------------------------------------------------------------------------------------#
    #------------------------------- INIITIALIZE SUBCLASS -------------------------------# 
    #------------------------------------------------------------------------------------#
    def __init__(self, 
                country, # ISO code of country
                start_year=2022, # start year of projection, first year is baseline value
                end_year=2053, # end year of projection
                adjustment_period=4, # number of years for linear spb_bca adjustment
                adjustment_start_year=2025, # start year of linear spb_bca adjustment
                ageing_cost_period=10, # number of years for ageing cost adjustment
                shock_sample_start=1970, # start year of shock sample
                stochastic_start_year=None,
                shock_frequency='quarterly', # start year of stochastic simulation
                inv_shock=False, # Investment counterfactual with temporary 0.5pp. drop in PB from 2025 to 2027/2030
                inv_size=0.5, # Size of investment counterfactual
                inv_period=None, # Period of investment counterfactual
                inv_exception=False, # Exception for EDP and deficit resilience safeguard for investment counterfactual
                growth_policy=False,
                growth_policy_effect=0,
                growth_policy_cost=0,
                growth_policy_period=1
                ): 
        
        # Initialize base class
        super().__init__(
            country, 
            start_year, 
            end_year, 
            adjustment_period, 
            adjustment_start_year, 
            ageing_cost_period, 
            inv_shock, 
            inv_size,
            inv_period,
            inv_exception,
            growth_policy,
            growth_policy_effect,
            growth_policy_cost,
            growth_policy_period
            )
        
        self.shock_frequency = shock_frequency
        self.shocks_sample_start = shock_sample_start
        self.outlier_threshold = 3
        if stochastic_start_year is None: stochastic_start_year = adjustment_start_year + adjustment_period - 1
        self.stochastic_start_year = stochastic_start_year
        self.stochastic_start = stochastic_start_year - start_year
        self.T_stochastic = self.end_year - stochastic_start_year
        if shock_frequency == 'quarterly':
            self.draw_period = self.T_stochastic * 4 
        else:
            self.draw_period = self.T_stochastic

        self._get_shock_data()

    def _get_shock_data(self):
        """
        Get shock data from Excel file and adjust outliers.
        """

        # Read country shock data and get number of variables
        if self.shock_frequency == 'quarterly': 
            self.df_shocks = pd.read_excel('../data/InputData/stochastic_model_data.xlsx', sheet_name=self.country, index_col=0).T.loc[str(self.shocks_sample_start)+'Q1':, :]
        elif self.shock_frequency == 'annual': 
            self.df_shocks = pd.read_excel('../data/InputData/stochastic_model_data_annual.xlsx', sheet_name=self.country, index_col=0).T.loc[self.shocks_sample_start:, :]
        self.num_variables = self.df_shocks.shape[1]
        
        # Adjust outliers defined by mean + outlier_threshold * standard deviation to the threshold
        self.df_shocks = self.df_shocks.clip(
            lower=self.df_shocks.mean() - self.outlier_threshold * self.df_shocks.std(), 
            upper=self.df_shocks.mean() + self.outlier_threshold * self.df_shocks.std(), 
            axis=1)
        
    #----------------------------------------------------------------------------------#
    #------------------------------- SIMULATION METHODS -------------------------------# 
    #----------------------------------------------------------------------------------#     
    def simulate(self, N=100000): # Set to 1 mio. for final version
        """
        Simulate the stochastic model.
        """
        # Set number of simulations
        self.N = N

        # Draw quarterly shocks
        self._draw_shocks()

        # Aggregate quarterly shocks to annual shocks
        if self.shock_frequency == 'quarterly': self._aggregate_shocks_quarterly()
        if self.shock_frequency == 'annual': self._aggregate_shocks_annual()
                
        # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()

    def _draw_shocks(self):
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

    def _aggregate_shocks_quarterly(self):
        """
        Aggregate quarterly shocks to annual shocks for specific variables.

        This method aggregates the shocks for exchange rate, short-term interest rate, nominal GDP growth, and primary balance
        from quarterly to annual shocks as the sum over four quarters. For long-term interest rate, it aggregates shocks over all past quarters up to the current year and m_res_lt.
        """

        # Reshape the shocks to sum over the four quarters
        self.shocks_sim_grouped = self.shocks_sim_draws.reshape((self.N, self.T_stochastic, 4, self.num_variables))

        try:
            # Try aggregating shocks for exchange rate, if not possible, set to zero
            exchange_rate_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -5], axis=2)
        except:
            exchange_rate_shocks = np.zeros((self.N, self.T_stochastic))

        ## Aggregate shocks for short-term interest rate, nominal GDP growth, and primary balance
        short_term_interest_rate_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -4], axis=2)
        nominal_gdp_growth_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -2], axis=2)
        primary_balance_shocks = np.sum(self.shocks_sim_grouped[:, :, :, -1], axis=2)

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
                self.shocks_sim_draws[:, q-q_to_sum-1 : q+1, -3], axis=(1))

            # Assign the aggregated shocks to the corresponding year
            self.long_term_interest_rate_shocks[:, t-1] = aggregated_shocks

        # Calculate the weighted average of short and long-term interest using share_st
        interest_rate_shocks = self.share_st * short_term_interest_rate_shocks + (1 - self.share_st) * self.long_term_interest_rate_shocks

        # Stack all shocks in a matrix
        self.shocks_sim = np.stack([exchange_rate_shocks, interest_rate_shocks, nominal_gdp_growth_shocks, primary_balance_shocks], axis=2)
        self.shocks_sim = np.transpose(self.shocks_sim, (0, 2, 1))

        # If stochastic projection starts before adjustment end, set pb shock to zero during adjustment period
        T_post_adjustment = self.T - self.adjustment_end - 1
        if self.T_stochastic > T_post_adjustment:
            self.shocks_sim[:, 3, :-T_post_adjustment] = 0

    def _aggregate_shocks_annual(self):
        """
        Save annual into shock matrix, aggregate long term interest rate shocks.
        """

        # Reshape the shocks
        self.shocks_sim_grouped = self.shocks_sim_draws.reshape((self.N, self.T_stochastic, self.num_variables))

        try:
            # Try retrieving shocks for exchange rate, if not possible, set to zero
            exchange_rate_shocks = self.shocks_sim_grouped[:, :, -5]
        except:
            exchange_rate_shocks = np.zeros((self.N, self.T_stochastic))

        ## Retrieve shocks for short-term interest rate, nominal GDP growth, and primary balance
        short_term_interest_rate_shocks = self.shocks_sim_grouped[:, :, -4]
        nominal_gdp_growth_shocks = self.shocks_sim_grouped[:, :, -2]
        primary_balance_shocks = self.shocks_sim_grouped[:, :, -1]

        ## Aggregate shocks for long-term interest rate
        # Calculate the average maturity in years 
        maturity_years = int(np.round(self.m_res_lt))

        # Initialize an array to store the aggregated shocks for the long-term interest rate
        self.long_term_interest_rate_shocks = np.zeros((self.N, self.T_stochastic))

        # Iterate over each year
        for t in range(1, self.T_stochastic+1):
            # Calculate the weight for each year based on the current year and m_res_lt
            weight = t / self.m_res_lt

            # Determine the number of years to sum based on the current year and m_res_lt
            t_to_sum = np.min([t, maturity_years])

            # Sum the shocks (N, T, num_variables) across the selected years
            aggregated_shocks = weight * np.sum(
                self.shocks_sim_draws[:, t-t_to_sum-1 : t+1, -3], axis=(1))

            # Assign the aggregated shocks to the corresponding year
            self.long_term_interest_rate_shocks[:, t-1] = aggregated_shocks

        # Calculate the weighted average of short and long-term interest using share_st
        interest_rate_shocks = self.share_st * short_term_interest_rate_shocks + (1 - self.share_st) * self.long_term_interest_rate_shocks

        # Stack all shocks in a matrix
        self.shocks_sim = np.stack([exchange_rate_shocks, interest_rate_shocks, nominal_gdp_growth_shocks, primary_balance_shocks], axis=2)
        self.shocks_sim = np.transpose(self.shocks_sim, (0, 2, 1))

        # If stochastic projection starts before adjustment end, set pb shock to zero during adjustment period
        T_post_adjustment = self.T - self.adjustment_end - 1
        if self.T_stochastic > T_post_adjustment:
            self.shocks_sim[:, 3, :-T_post_adjustment] = 0

    def _combine_shocks_baseline(self):
        """
        Combine shocks with the respective baseline variables and set starting values for simulation.
        """

        # Create arrays to store the simulated variables
        d_sim = np.zeros([self.N, self.T_stochastic+1])  # Debt to GDP ratio
        exr_eur_sim = np.zeros([self.N, self.T_stochastic+1])  # Exchange rate
        iir_sim = np.zeros([self.N, self.T_stochastic+1])  # Implicit interest rate
        ng_sim = np.zeros([self.N, self.T_stochastic+1])  # Nominal GDP growth
        pb_sim = np.zeros([self.N, self.T_stochastic+1])  # Primary balance
        sf_sim = np.zeros([self.N, self.T_stochastic+1])  # Stock flow adjustment

        # Call the Numba JIT function with converted self variables
        combine_shocks_baseline_jit(
            N=self.N, 
            T_stochastic=self.T_stochastic, 
            shocks_sim=self.shocks_sim, 
            exr_eur=self.exr_eur, 
            iir=self.iir, 
            ng=self.ng, 
            pb=self.pb, 
            sf=self.sf, 
            d=self.d, 
            d_sim=d_sim, 
            exr_eur_sim=exr_eur_sim, 
            iir_sim=iir_sim, 
            ng_sim=ng_sim, 
            pb_sim=pb_sim, 
            sf_sim=sf_sim
            )

        # Save the resulting self.variables
        self.d_sim = d_sim
        self.exr_eur_sim = exr_eur_sim
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
            N=self.N, T_stochastic=self.T_stochastic, share_eur_stochastic=self.share_eur_stochastic, d_sim=self.d_sim, iir_sim=self.iir_sim, 
            ng_sim=self.ng_sim, exr_eur_sim=self.exr_eur_sim, pb_sim=self.pb_sim, sf_sim=self.sf_sim)

    #---------------------------------------------------------------------------------#
    #------------------------------- AUXILIARY METHODS -------------------------------# 
    #---------------------------------------------------------------------------------#    
    def fanchart(self, save_as=False, show=True, variable='d', periods=5):
        """
        Plot a fanchart of the simulated debt-to-GDP ratio. And save the figure if save_as is specified.
        """

        # Set stochastic variable
        sim_var = eval(f'self.{variable}_sim')
        bl_var = eval(f'self.{variable}')

        # Calculate the percentiles
        self.pcts_dict = {}
        for pct in np.arange(10, 100, 10):
            self.pcts_dict[pct] = np.percentile(sim_var, pct, axis=0)[:periods+1]

        # Create array of years and baseline debt-to-GDP ratio
        years = np.arange(self.start_year, self.end_year+1)

        # Set color pallette
        sns.set_palette(sns.color_palette("Blues"))

        # Plot the results using fill between
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], self.pcts_dict[10], self.pcts_dict[90], label='10th-90th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], self.pcts_dict[20], self.pcts_dict[80], label='20th-80th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], self.pcts_dict[30], self.pcts_dict[70], label='30th-70th percentile')
        ax.fill_between(years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], self.pcts_dict[40], self.pcts_dict[60], label='40th-60th percentile')
        ax.plot(years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], self.pcts_dict[50], alpha=1, ls='-', color='black', label='median')
        ax.plot(years, bl_var, ls='--', color='red', label='Baseline')

        # Plot layout
        ax.legend(loc='upper left')
        ax.xaxis.grid(False)
        ax.set_ylabel(variable)
        ax.set_title(f'{self.country}_{self.adjustment_period}')

        # Saveplot data in a dataframe if self.save_df is specified
        self.df_fanchart = pd.DataFrame({'year': years, 'baseline': bl_var})
        for pct in self.pcts_dict:
            df_pct = pd.DataFrame({'year': years[-self.T_stochastic-1:-self.T_stochastic-1+periods+1], f'p{pct}': self.pcts_dict[pct]})
            self.df_fanchart = self.df_fanchart.merge(df_pct, on='year', how='left')

        # Save plot if save_as is specified
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
        
        # Do not show plot if show is False
        if show == False:
            plt.close()

    #-----------------------------------------------------------------------------------------------#
    #------------------------------- STOCHASTIC OPTIMIZATION METHODS -------------------------------# 
    #-----------------------------------------------------------------------------------------------#     
    def find_spb_stochastic(self, prob_target=0.3, bounds=(-10, 10), stochastic_criterion_period=5, print_update=False):
        """
        Find the structural primary balance that ensures the probability of the debt-to-GDP ratio exploding is equal to prob_target.
        """

        self.stochastic_criterion_period = stochastic_criterion_period   
        self.stochastic_criterion_start = self.adjustment_period - self.stochastic_start + 2
        self.stochastic_criterion_end = self.stochastic_criterion_start + self.stochastic_criterion_period
        self.print_update = print_update
        
        if not hasattr(self, 'edp_steps'):
           self.edp_steps = None

        if not hasattr(self, 'deficit_resilience_steps'):
           self.deficit_resilience_steps = None

        if not hasattr(self, 'post_adjustment_steps'):
            self.post_adjustment_steps = None
        
        self.stochastic_optimization_dict = {}

        # Initial projection
        self.project(
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_adjustment_steps=self.post_adjustment_steps,
            scenario=None
            )
        
        # Optimize for both dbet decline and debt remaining under 60 and choose the lower SPB
        self.spb_target = self._stochastic_optimization(prob_target=prob_target, bounds=bounds)

        # Project with optimal spb
        self.project(
            spb_target=self.spb_target, 
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps,
            post_adjustment_steps=self.post_adjustment_steps,
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

        # Optimize _target_pb to find b_target that ensures prob_debt_above_60 == prob_target
        self.spb_target = minimize_scalar(self._target_spb_decline, bounds=self.spb_bounds).x
        
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
            post_adjustment_steps=self.post_adjustment_steps,
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

        min_prob = np.min([self.prob_explodes, self.prob_above_60])

        return np.abs(min_prob - self.prob_target)

    def prob_debt_explodes(self):
        """
        Calculate the probability of the debt-to-GDP ratio exploding.
        """

        # Call the Numba JIT function with converted self variables
        self.prob_explodes = prob_debt_explodes_jit(
            N=self.N, 
            d_sim=self.d_sim, 
            stochastic_criterion_start=self.stochastic_criterion_start,
            stochastic_criterion_end=self.stochastic_criterion_end
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
            stochastic_criterion_end=self.stochastic_criterion_end
            )
        return self.prob_above_60
    
    #-------------------------------------------------------------------------------------#
    #------------------------------- INTEGRATED OPTIMIZERS -------------------------------#
    #-------------------------------------------------------------------------------------#

    def find_spb_binding(self, save_df=False, edp=True, debt_safeguard=True, deficit_resilience=True):
        """
        Find the structural primary balance that meets all criteria after deficit has been brought below 3% and debt safeguard is satisfied.
        """

        # Initiate spb_target and dataframe dictionary
        print(f'\n________________________ Optimizing {self.country} {self.adjustment_period}-year ________________________')
        self.save_df = save_df
        self.spb_target_dict = {}
        self.binding_parameter_dict = {}
        if self.save_df: self.df_dict = {}

        # Run DSA and deficit criteria and project toughest under baseline assumptions
        self.project(spb_target=None, edp_steps=None) # clear projection
        self._run_dsa()
        self._get_binding()

        # Apply EDP and safeguards
        if edp: self._apply_edp()
        if debt_safeguard: self._apply_debt_safeguard()
        if deficit_resilience: self._apply_deficit_resilience()

        # Save binding SPB target
        self.spb_target_dict['binding'] = self.spb_bca[self.adjustment_end]

        # Save binding parameters to reproduce adjustment path
        self.binding_parameter_dict['adjustment_steps'] = self.adjustment_steps
        self.binding_parameter_dict['spb_target'] = self.binding_spb_target
        if edp: self.binding_parameter_dict['edp_binding'] = self.edp_binding
        if edp: self.binding_parameter_dict['edp_steps'] = self.edp_steps
        if deficit_resilience: self.binding_parameter_dict['deficit_resilience_steps'] = self.deficit_resilience_steps
        if deficit_resilience: self.binding_parameter_dict['post_adjustment_steps'] = self.post_adjustment_steps
        if self.inv_shock: self.binding_parameter_dict['inv_space'] = self.inv_space

        # Save dataframe
        if self.save_df: self.df_dict['binding'] = self.df(all=True)
        return f'Binding SPB target: {self.spb_bca[self.adjustment_end]} ({self.binding_criterion})'

    def _run_dsa(self, criterion='all'):
        """
        Run DSA for given criterion.
        """

        deterministic_criteria_list = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g', 'deficit_reduction']

        # If all criteria, run all deterministic and stochastic
        if criterion == 'all':
            
            # Run all deterministic scenarios, skip if criterion not applicable
            for deterministic_criterion in deterministic_criteria_list:
                try:
                    spb_target = self.find_spb_deterministic(criterion=deterministic_criterion)
                    self.spb_target_dict[deterministic_criterion] = spb_target
                    if self.save_df:
                        self.df_dict[deterministic_criterion] = self.df(all=True)
                except:
                    raise

            # Run stochastic scenario, skip if not possible due to lack of data
            try: 
                spb_target = self.find_spb_stochastic()
                self.spb_target_dict['stochastic'] = spb_target
                if self.save_df: self.df_dict['stochastic'] = self.df(all=True)
            except:
                pass

        # If specific criterion given, run only one optimization
        else:
            if criterion in deterministic_criteria_list:
                self.binding_spb_target = self.find_spb_deterministic(criterion=criterion)

            elif criterion == 'stochastic':
                self.binding_spb_target = self.find_spb_stochastic()

            # Replace binding scenario
            self.binding_spb_target = self.spb_bca[self.adjustment_end]
            print(f'SPB* after EDP for {criterion}: {self.binding_spb_target}')
            self.spb_target_dict[criterion] = self.binding_spb_target
            
    def _get_binding(self):
        """
        Get binding SPB target and scenario from dictionary with SPB targets.
        """
        # Get binding SPB target and scenario from dictionary with SPB targets
        self.binding_spb_target = np.max(list(self.spb_target_dict.values()))
        self.binding_criterion = list(self.spb_target_dict.keys())[np.argmax(list(self.spb_target_dict.values()))]
        
        # Project under baseline assumptions
        self.project(spb_target=self.binding_spb_target, scenario=None)
        
        # Print results
        print(f'SPB*: {self.spb_bca[self.adjustment_end]} ({self.binding_criterion})')

    def _apply_edp(self):
        """ 
        Check if EDP is binding in binding scenario and apply if it is.
        """
        # Check if EDP binding, run DSA for periods after EDP and project new path under baseline assumptions
        self.find_edp(spb_target=self.binding_spb_target)
        self._run_dsa(criterion=self.binding_criterion)
        self.project(
            spb_target=self.binding_spb_target, 
            edp_steps=self.edp_steps,
            scenario=None
            )
        if (not np.isnan(self.edp_steps[0]) 
            and self.edp_steps[0] > self.adjustment_steps[0]): 
            self.edp_binding = True 
        else: 
            self.edp_binding = False
        if self.edp_binding: 
            print(f'SPB* after applying EDP: {self.spb_bca[self.adjustment_end]}, EDP period: {self.edp_period}')
        else: 
            print(f'EDP not binding')
        if self.save_df: self.df_dict['edp'] = self.df(all=True)

    def _apply_debt_safeguard(self): 
        """ 
        Check if Council version of debt safeguard is binding in binding scenario and apply if it is.
        """
        # Define steps for debt safeguard
        self.edp_steps[:self.edp_period] = self.adjustment_steps[:self.edp_period]

        # Debt safeguard binding for countries with high debt and debt decline below 4 or 2 % for 4 years after edp
        debt_safeguard_decline = 1 if self.d[self.adjustment_start - 1] > 90 else 0.5

        # If investment counterfactual, calculate non-investment debt decline
        if self.inv_exception:
            self._calculate_d_non_inv()
        
            # If EDP period is 0, need debt ratio from 2024, pre-investment
            if self.edp_period == 0:
                debt_safeguard_criterion = (self.d[self.edp_end] - self.d_non_inv[-1] 
                                            < debt_safeguard_decline * (self.adjustment_end - self.edp_end))
            
            # If EDP period is not 0, need debt ratio from EDP period, post-investment
            else:
                debt_safeguard_criterion = (self.d_non_inv[self.edp_period] - self.d_non_inv[-1] 
                                            < debt_safeguard_decline * (self.adjustment_end - self.edp_end))
        
        # If no investment counterfactual, calculate debt decline
        else:
            #debt_safeguard_criterion_inv_exception = True
            debt_safeguard_criterion = (self.d[self.edp_end] - self.d[self.adjustment_end] 
                                        < debt_safeguard_decline * (self.adjustment_end - self.edp_end))
        
        if (self.d[self.adjustment_start-1] >= 60 
            and debt_safeguard_criterion):
            
            # Call deterministic optimizer to find SPB target for debt safeguard
            self.spb_debt_safeguard_target = self.find_spb_deterministic(criterion='debt_safeguard')
            
            # If debt safeguard SPB target is higher than DSA target, save debt safeguard target
            if self.spb_debt_safeguard_target > self.binding_spb_target + 1e-8: # 1e-8 tolerance for floating point errors
                self.binding_spb_target = self.spb_debt_safeguard_target
                self.spb_target = self.binding_spb_target
                self.binding_criterion = 'debt_safeguard'
                self.spb_target_dict['debt_safeguard'] = self.binding_spb_target
                if self.save_df: self.df_dict['debt_safeguard'] = self.df(all=True)
                print(f'SPB* after binding debt safeguard: {self.binding_spb_target}')
        else:
            print(f'Debt safeguard not binding')

    def _apply_deficit_resilience(self):
        """ 
        Apply deficit resilience safeguard after binding scenario.
        """
        # For countries with high deficit, find long term SPB target that brings and keeps deficit below 1.5%
        if (self.ob[self.adjustment_start-1] < -3
            or self.d[self.adjustment_start-1] > 60): 
            self.find_spb_deficit_resilience()
                
        # Save results and print update
        if np.any([~np.isnan(self.deficit_resilience_steps)]):
            print(f'SPB* after deficit resilience: {self.spb_bca[self.adjustment_end]}')
            self.spb_target_dict['deficit_resilience'] = self.spb_bca[self.adjustment_end]
        else:
            print('Deficit resilience safeguard not binding during adjustment period')            
        
        if np.any([~np.isnan(self.post_adjustment_steps)]):
            print(f'SPB post-adjustment: {self.spb[self.adjustment_end+10]}')
            self.spb_target_dict['post_adjustment'] = self.spb[self.adjustment_end+10]
        else:
            print('Deficit resilience safeguard not binding after adjustment period')

        if self.save_df: self.df_dict['deficit_resilience'] = self.df(all=True)

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

        # Set stochastic period to include adjustment period
        self.T_stochastic = self.T - self.adjustment_start
        if self.shock_frequency == 'quarterly':
            self.draw_period = self.T_stochastic * 4 
        else:
            self.draw_period = self.T_stochastic

        # Set exchange rate and primary balance shock to zero
        self.df_shocks[['exchange_rate', 'primary_balance']] = 0

        # Draw quarterly shocks
        self._draw_shocks()

        # Aggregate quarterly shocks to annual shocks
        if self.shock_frequency == 'quarterly': 
            self._aggregate_shocks_quarterly()
        else:
            self._aggregate_shocks_annual()

        # Replace PB shock (self.shocks_sim[:,3]) with growth shock (self.shocks_sim[:,2]) times budget balance multiplier
        self.shocks_sim[:, 3] = self.pb_elasticity * self.shocks_sim[:, 2]
        
        # Add shocks to baseline variables and set start values
        self._combine_shocks_baseline()

        # Simulate debt
        self._simulate_debt()
        
        # Simulate deficit
        self.ob_sim = np.zeros([self.N, self.T_stochastic+1])  # Debt to GDP ratio
        self.ob_sim[:, 0] = self.ob[-(self.T_stochastic+1)]
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
            N=self.N, T_stochastic=self.T_stochastic, pb_sim=self.pb_sim, iir_sim=self.iir_sim, ng_sim=self.ng_sim, d_sim=self.d_sim, ob_sim=self.ob_sim
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
  
#-----------------------------------------------------------------------------------------#
#------------------------------- NUMBA OPTIMIZED FUNCTIONS -------------------------------#
#-----------------------------------------------------------------------------------------#

@jit(nopython=True)
def combine_shocks_baseline_jit(N, T_stochastic, shocks_sim, exr_eur, iir, ng, pb, sf, d, d_sim, exr_eur_sim, iir_sim, ng_sim, pb_sim, sf_sim):
    """
    Add shocks to the baseline variables and set starting values for simulation.
    """
    # Add shocks to the baseline variables for period after start year
    # Combines shocks with baseline variables for periods after adjustment end, so from - T_stochastic onward
    for n in range(N):
        exr_eur_sim[n, 1:] = exr_eur[-T_stochastic:] + shocks_sim[n, 0] 
        iir_sim[n, 1:] = iir[-T_stochastic:] + shocks_sim[n, 1]
        ng_sim[n, 1:] = ng[-T_stochastic:] + shocks_sim[n, 2]
        pb_sim[n, 1:] = pb[-T_stochastic:] + shocks_sim[n, 3]
    
    # Set values for stock-flow adjustment 
    sf_sim[:, 1:] = sf[-T_stochastic:]

    # Set the starting values t0
    d_sim[:, 0] = d[-T_stochastic-1]
    exr_eur_sim[:, 0] = exr_eur[-T_stochastic-1]
    iir_sim[:, 0] = iir[-T_stochastic-1]
    ng_sim[:, 0] = ng[-T_stochastic-1]
    pb_sim[:, 0] = pb[-T_stochastic-1]

@jit(nopython=True)
def simulate_debt_jit(N, T_stochastic, share_eur_stochastic, d_sim, iir_sim, ng_sim, exr_eur_sim, pb_sim, sf_sim):
    """
    Simulate the debt-to-GDP ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, T_stochastic+1):
            d_sim[n, t] = share_eur_stochastic * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) \
                        + (1 - share_eur_stochastic) * d_sim[n, t-1] * (1 + iir_sim[n, t]/100) / (1 + ng_sim[n, t]/100) \
                        * (exr_eur_sim[n, t]) / (exr_eur_sim[n, t-1]) - pb_sim[n, t] + sf_sim[n, t]
            
@jit(nopython=True)
def prob_debt_explodes_jit(N, d_sim, stochastic_criterion_start, stochastic_criterion_end):
    """
    Calculate the probability of the debt-to-GDP ratio exploding.
    """
    prob_explodes = 0
    for n in range(N):
        if d_sim[n, stochastic_criterion_start] < d_sim[n, stochastic_criterion_end]:
            prob_explodes += 1
    return prob_explodes / N

@jit(nopython=True)
def prob_debt_above_60_jit(N, d_sim, stochastic_criterion_end):
    """
    Calculate the probability of the debt-to-GDP ratio exceeding 60
    """
    prob_debt_above_60 = 0
    for n in range(N):
        if 60 < d_sim[n, stochastic_criterion_end]:
            prob_debt_above_60 += 1
    return prob_debt_above_60 / N

@jit(nopython=True)
def simulate_deficit_jit(N, T_stochastic, pb_sim, iir_sim, ng_sim, d_sim, ob_sim):
    """
    Simulate the fiscal balance ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, T_stochastic+1):
            ob_sim[n, t] = pb_sim[n, t] - iir_sim[n, t] / 100 / (1 + ng_sim[n, t] / 100) * d_sim[n, t-1]