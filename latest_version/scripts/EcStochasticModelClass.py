#=========================================================================================#
#          European Commission Debt Sustainability Analysis - Stochastic SubClass         #
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
# Updated: 2023-12-21
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

    #------------------------------------------------------------------------------------#
    #------------------------------- INIITIALIZE SUBCLASS -------------------------------# 
    #------------------------------------------------------------------------------------#
    def __init__(self, 
                country, # ISO code of country
                start_year=2022, # start year of projection, first year is baseline value
                end_year=2053, # end year of projection
                adjustment_period=4, # number of years for linear spb_bca adjustment
                adjustment_start=2025, # start year of linear spb_bca adjustment
                inv_shock=False): # Investment counterfactual with temporary 0.5pp. drop in PB from 2025 to 2027/2030
        super().__init__(country, start_year, end_year, adjustment_period, adjustment_start, inv_shock)
        
        self.outlier_threshold = 3
        start_year_stochastic = adjustment_start - 1 + adjustment_period
        self.T_stochastic = self.end_year - start_year_stochastic
        self.num_quarters = self.T_stochastic * 4

        self._get_shock_data()

    def _get_shock_data(self):
        """
        Get shock data from Excel file and adjust outliers.
        """

        # Select country shock data and get number of variables
        self.df_shocks = pd.read_excel('../data/InputData/stochastic_model_data.xlsx', sheet_name=self.country, index_col=0).T
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
        exr_eur_sim = np.zeros([self.N, self.T_stochastic+1])  # Exchange rate
        iir_sim = np.zeros([self.N, self.T_stochastic+1])  # Implicit interest rate
        ng_sim = np.zeros([self.N, self.T_stochastic+1])  # Nominal GDP growth
        pb_sim = np.zeros([self.N, self.T_stochastic+1])  # Primary balance 
        sf_sim = np.zeros([self.N, self.T_stochastic+1])  # Stock flow adjustment

        # Call the Numba JIT function with converted self variables
        combine_shocks_baseline_jit(
            N=self.N, T_stochastic=self.T_stochastic, shocks_sim=self.shocks_sim, exr_eur=self.exr_eur, 
            iir=self.iir, ng=self.ng, pb=self.pb, sf=self.sf, d=self.d, d_sim=d_sim, exr_eur_sim=exr_eur_sim, 
            iir_sim=iir_sim, ng_sim=ng_sim, pb_sim=pb_sim, sf_sim=sf_sim
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
        ax.set_title(f'{self.country}_{self.adjustment_period}')

        # Saveplot data in a dataframe if self.save_df is specified
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

    #-----------------------------------------------------------------------------------------------#
    #------------------------------- STOCHASTIC OPTIMIZATION METHODS -------------------------------# 
    #-----------------------------------------------------------------------------------------------#     
    def find_spb_stochastic(self, prob_target=0.3, bounds=(-3, 5), print_update=False):
        """
        Find the structural primary balance that ensures the probability of the debt-to-GDP ratio exploding is equal to prob_target.
        """
            
        self.print_update = print_update
        if not hasattr(self, 'initial_adjustment_period'):
           self.initial_adjustment_period = 0
           self.initial_adjustment_step = 0.5
        if not hasattr(self, 'intermediate_adjustment_period'):
           self.initial_adjustment_period = 0
           self.initial_adjustment_step = 0

        # Initial projection
        self.project(scenario=None, 
                     initial_adjustment_period=self.initial_adjustment_period, 
                     initial_adjustment_step=self.initial_adjustment_step, 
                     intermediate_adjustment_period=self.intermediate_adjustment_period, 
                     intermediate_adjustment_step=self.intermediate_adjustment_step
                     )

        # If debt<60 and deficit<3, find spb_deterministic that ensures deficit remains <3
        if (
            self.ob[self.adjustment_start - 1] >= -3
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
            self.project(spb_target=self.spb_target, 
                         scenario=None, 
                         initial_adjustment_period=self.initial_adjustment_period, 
                         initial_adjustment_step=self.initial_adjustment_step, 
                         intermediate_adjustment_period=self.intermediate_adjustment_period, 
                         intermediate_adjustment_step=self.intermediate_adjustment_step
                         )

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
            self.project(spb_target=self.spb_target, 
                         scenario=None, 
                         initial_adjustment_period=self.initial_adjustment_period, 
                         initial_adjustment_step=self.initial_adjustment_step, 
                         intermediate_adjustment_period=self.intermediate_adjustment_period, 
                         intermediate_adjustment_step=self.intermediate_adjustment_step
                         )

        return self.spb_target
    
    def _target_spb_explodes(self, spb_target):
        """
        Returns zero if primary balance ensures prop_debt_explodes == prob_target.
        """

        # Simulate the debt-to-GDP ratio with the given primary balance target
        self.project(spb_target=spb_target, 
                     scenario=None, 
                     initial_adjustment_period=self.initial_adjustment_period, 
                     initial_adjustment_step=self.initial_adjustment_step, 
                     intermediate_adjustment_period=self.intermediate_adjustment_period, 
                     intermediate_adjustment_step=self.intermediate_adjustment_step
                     )

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
        self.project(spb_target=spb_target, 
                     scenario=None, 
                     initial_adjustment_period=self.initial_adjustment_period, 
                     initial_adjustment_step=self.initial_adjustment_step, 
                     intermediate_adjustment_period=self.intermediate_adjustment_period, 
                     intermediate_adjustment_step=self.intermediate_adjustment_step
                     )

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
    
    #-------------------------------------------------------------------------------------#
    #------------------------------- INTEGRATED OPTIMIZERS -------------------------------#
    #-------------------------------------------------------------------------------------#

    def find_spb_binding(self, save_df=False, edp=True, debt_safeguard=True, deficit_resilience=True):
        """
        Find the structural primary balance that meets all criteria after deficit has been brought below 3% and debt safeguard is satisfied.
        """

        # Initiate spb_target and dataframe dictionary
        print(f'\n________________________ {self.country} {self.adjustment_period}-year adjustment ________________________')
        self.save_df = save_df
        self.spb_target_dict = {}
        self.binding_parameter_dict = {}
        if self.save_df: self.df_dict = {}

        # Run DSA and deficit criteria and project toughest under baseline assumptions
        self._run_dsa()
        self._get_binding()

        # Apply EDP and safeguards
        if edp: self._apply_edp()
        if debt_safeguard: self._apply_debt_safeguard()
        if deficit_resilience: self._apply_deficit_resilience()

        # Save binding SPB target
        self.spb_target_dict['binding'] = self.spb_bca[self.adjustment_end]

        # Save binding parameters to reproduce adjustment path
        self.binding_parameter_dict['binding_spb_target'] = self.binding_spb_target
        self.binding_parameter_dict['initial_adjustment_period'] = self.initial_adjustment_period
        self.binding_parameter_dict['initial_adjustment_step'] = self.initial_adjustment_step
        self.binding_parameter_dict['intermediate_adjustment_period'] = self.intermediate_adjustment_period
        self.binding_parameter_dict['intermediate_adjustment_step'] = self.intermediate_adjustment_step
        self.binding_parameter_dict['deficit_resilience_periods'] = self.deficit_resilience_periods
        self.binding_parameter_dict['deficit_resilience_step'] = self.deficit_resilience_step
        self.binding_parameter_dict['post_adjustment_periods'] = self.post_adjustment_periods

        # Save dataframe
        if self.save_df: self.df_dict['binding'] = self.df(all=True)
        return f'Binding SPB target: {self.spb_bca[self.adjustment_end]} ({self.binding_scenario})'

    def _run_dsa(self, scenario='all'):
        """
        Run DSA for given scenario and criterion.
        """
        # Initiate dictionary for deterministic scenarios and criteria
        deterministic_scenario_dict = {'main_adjustment': ['deficit_reduction', 'debt_decline'],
                                       'lower_spb': ['debt_decline'],
                                       'financial_stress': ['debt_decline'],
                                       'adverse_r_g': ['debt_decline']
                                       }

        ## If all scenarios, run all deterministic and stochastic
        if scenario == 'all':
            for deterministic_scenario, criteria_list in deterministic_scenario_dict.items():
                for criterion in criteria_list:

                    # Try to call determiistic optimizer, will return error if criterion not applicable
                    try:
                        spb_target = self.find_spb_deterministic(scenario=deterministic_scenario, criterion=criterion)
                        self.spb_target_dict[f'{deterministic_scenario}_{criterion}'] = spb_target
                        if self.save_df:
                            self.df_dict[deterministic_scenario] = self.df(all=True)
                    except:
                        continue

            # Try to call stochastic optimizer, will return error for countries with lacking data
            try: 
                spb_target = self.find_spb_stochastic()
                self.spb_target_dict['stochastic_debt_decline'] = spb_target
                if self.save_df: self.df_dict['stochastic_debt_decline'] = self.df(all=True)
            except:
                pass

        ## If specific scenario_criterion combination given, retrieve combination and run optimizer
        else:
            deterministic_scenario, criterion = next(
            ((deterministic_scenario, criterion) for deterministic_scenario, criteria_list in deterministic_scenario_dict.items() 
            for criterion in criteria_list if scenario == f'{deterministic_scenario}_{criterion}'
            ), (None, None))
        
            # If we retrieve combination of a deterministic scenario and criterion, call deterministic optimizer
            if deterministic_scenario is not None:
                self.binding_spb_target = self.find_spb_deterministic(scenario=deterministic_scenario, criterion=criterion)

            # If stochastic scenario, call stochastic optimizer
            elif scenario == 'stochastic_debt_decline':
                self.binding_spb_target = self.find_spb_stochastic()

            # Replace binding scenario target with frontloaded version
            self.spb_target_dict[scenario] = self.binding_spb_target
            
    def _get_binding(self):
        """
        Get binding SPB target and scenario from dictionary with SPB targets.
        """
        # Get binding SPB target and scenario from dictionary with SPB targets
        self.binding_spb_target = np.max(list(self.spb_target_dict.values()))
        self.binding_scenario = list(self.spb_target_dict.keys())[np.argmax(list(self.spb_target_dict.values()))]
        
        # Project under baseline assumptions
        self.project(spb_target=self.binding_spb_target, scenario=None)
        
        # Print results
        print(f'SPB*: {self.spb_bca[self.adjustment_end]} ({self.binding_scenario})')

    def _apply_edp(self):
        """ 
        Check if EDP is binding in binding scenario and apply if it is.
        """
        # If deficit excessive in 2024 and 2025 calcualte EDP
        if np.all(self.ob[self.adjustment_start-1:self.adjustment_start+1] < -3):
            
            # If adjustment step < 0.5, calculate EDP period and adjustment
            if self.adjustment_steps[0] < 0.5: 
                self.calculate_edp()

                # Run DSA for periods after EDP and project new path under baseline assumptions
                self._run_dsa(scenario=self.binding_scenario)
                self.project(spb_target=self.binding_spb_target, 
                            initial_adjustment_period=self.initial_adjustment_period,
                            initial_adjustment_step=self.initial_adjustment_step,
                            scenario=None)

                # Print update
                if self.save_df: self.df_dict['edp'] = self.df(all=True)
                print(f'SPB* after binding EDP: {self.spb_bca[self.adjustment_end]}, EDP period: {self.initial_adjustment_period}')
            
            # If adjustment step >= 0.5, save duration of excessive deficit for application of debt safeguard
            else: 
                self.initial_adjustment_period = np.where(self.ob[self.adjustment_start-1:self.adjustment_end+1] > -3)[0][0]
                self.initial_adjustment_step = self.adjustment_steps[0]
                self.edp_end = self.adjustment_start - 1 + self.initial_adjustment_period
                print(f'EDP already satisfied, deficit below 3% in period {self.initial_adjustment_period}')
        else:
            print(f'EDP not binding')

    def _apply_debt_safeguard(self):
        """ 
        Check if debt safeguard is binding in binding scenario and apply if it is.
        """
        # Debt safeguard binding for countries with high debt and debtl decline below 4 or 2 % for 4 years after edp
        if self.d[self.adjustment_start-1] >= 60 and self.d[self.edp_end] - self.d[self.adjustment_end] < self.debt_safeguard_decline * self.debt_safeguard_period:

            # Call deterministic optimizer to find SPB target for debt safeguard
            self.spb_debt_safeguard_target = self.find_spb_deterministic(criterion='debt_safeguard')

            # If debt safeguard SPB target is higher than DSA SPB target, save debt safeguard SPB target to prevent deconsolidation
            if self.spb_debt_safeguard_target > self.binding_spb_target + 1e-8: # 1e-8 tolerance for floating point errors
                self.binding_spb_target = self.spb_debt_safeguard_target
                self.binding_scenario = 'debt_safeguard'
                self.spb_target_dict['debt_safeguard'] = self.binding_spb_target
                if self.save_df: self.df_dict['debt_safeguard'] = self.df(all=True)
                print(f'SPB* after binding debt safeguard: {self.binding_spb_target}')
        
        else:
            print(f'Debt safeguard not binding')

    def _apply_deficit_resilience(self):
        """ 
        Apply deficit resilience safeguard after binding scenario.
        """
        # Call deficit resilience optimizer to find long term SPB target that brings and keeps deficit below 1.5%
        self.find_spb_deficit_resilience(self.binding_spb_target,
                                         initial_adjustment_period=self.initial_adjustment_period,
                                         initial_adjustment_step=self.initial_adjustment_step,
                                         )
                
        # Save results and print update
        if np.any(self.deficit_resilience_periods == True):
            print(f'SPB* after deficit resilience: {self.spb_bca[self.adjustment_end]}, Deficit resilience periods: {np.arange(self.adjustment_start+self.start_year, self.adjustment_end+self.start_year+1)[self.deficit_resilience_periods]}')
        else:
            print('Deficit resilience safeguard not binding during adjustment period')
        
        if self.spb_bca[self.adjustment_end] > self.binding_spb_target + 1e-8: # 1e-8 tolerance for floating point errors
            self.spb_target_dict['deficit_resilience'] = self.spb_bca[self.adjustment_end]
        
        if np.any(self.post_adjustment_periods == True):
            print(f'SPB post-adjustment: {self.spb_bca[self.adjustment_end+10]}, Post adjustment periods: {np.arange(self.adjustment_end+1+self.start_year, self.T+self.start_year)[self.post_adjustment_periods]}')
        else:
            print('Deficit resilience safeguard not binding after adjustment period')

        self.spb_target_dict['post_adjustment'] = self.spb_bca[self.adjustment_end+10]
        if self.save_df: self.df_dict['deficit_resilience'] = self.df(all=True)

    def find_deficit_prob(self):
        """
        Find the probability of the deficit exceeding 3% in each adjustment period for binding SPB path.
        """
        # Initial projection
        self.project(spb_target=self.binding_spb_target, 
                    initial_adjustment_period=self.initial_adjustment_period, 
                    initial_adjustment_step=self.initial_adjustment_step, 
                    deficit_resilience_periods=self.deficit_resilience_periods,
                    deficit_resilience_step=self.deficit_resilience_step,
                    scenario=self.scenario)
        
        # Set stochastic period to include adjustment period
        self.T_stochastic = self.T - self.adjustment_start
        self.num_quarters = self.T_stochastic * 4

        # Set exchange rate and primary balance shock to zero
        self.df_shocks[['exchange_rate', 'primary_balance']] = 0

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
def simulate_deficit_jit(N, T_stochastic, pb_sim, iir_sim, ng_sim, d_sim, ob_sim):
    """
    Simulate the fiscal balance ratio using the baseline variables and the shocks.
    """
    for n in range(N):
        for t in range(1, T_stochastic+1):
            ob_sim[n, t] = pb_sim[n, t] - iir_sim[n, t] / 100 / (1 + ng_sim[n, t] / 100) * d_sim[n, t-1]