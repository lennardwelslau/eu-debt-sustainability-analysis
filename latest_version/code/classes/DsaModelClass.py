# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Base Class               #
# ========================================================================================= #
#
# The DsaModel class serves as the base framework for simulating baseline and detemrinistic 
# scenario debt paths following the methodology of the European Commission's Debt Sustainability 
# Monitor. The class encompasses three primary parts:
#
# 1. **Data Methods:** These methods clean and combine input data. Input data can be compiled using 
#    the provided jupyter notebook "01_data_preparation.ipynb".
# 2. **Projection Methods:** These methods handle the projection of economic variables such as GDP
#    growth, primary balance, interest rates, and debt dynamics, based on different scenarios and
#    adjustment steps.
# 3. **Optimization and Auxiliary Methods:** These methods include functions to optimize the primary
#    balance to meet specific criteria, and check deterministic conditions.
#
# In addition, the StochasticDsaModel subclass, a specialized subclass building upon this base class,
# provides additional features for stochastic projection around the deterministic debt path.
#
# For comments and suggestions please contact lennard.welslau[at]gmail[dot]com
#
# Author: Lennard Welslau
# Updated: 2024-12-01
#
# ========================================================================================= #

# Import libraries and modules
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.grid':True,'grid.color':'black','grid.alpha':'0.25','grid.linestyle':'--'})
plt.rcParams.update({'font.size': 14})
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class DsaModel:

    # ========================================================================================= #
    #                                   INIITIALIZE MODEL                                       #
    # ========================================================================================= #
        
    def __init__(
            self,
            country,  # ISO code of country
            start_year=2023,  # start year of projection, first year is baseline value
            end_year=2070,  # end year of projection
            adjustment_period=4,  # number of years for linear spb_bca adjustment
            adjustment_start_year=2025,  # start year of linear spb_bca adjustment
            ageing_cost_period=10,  # number of years for ageing cost adjustment after adjustment period
            fiscal_multiplier=0.75, # fiscal multiplier for fiscal adjustment
            fiscal_multiplier_persistence=3, # persistence of fiscal multiplier in years
            fiscal_multiplier_type='com', # type of fiscal multiplier, commission or pers version 
            bond_data=False, # Use bond level data for repayment profile
        ):

        # Initialize model parameters
        self.country = country  # country ISO code
        self.start_year = start_year  # start year of projection (T), normally the last year of non-forecast observations
        self.end_year = end_year  # end year of projection (T+30)
        self.projection_period = self.end_year - self.start_year + 1  # number of years in projection
        self.adjustment_period = adjustment_period  # adjustment period for structural primary balance, for COM 4 or 7 years
        self.adjustment_start_year = adjustment_start_year  # start year of adjustment period
        self.adjustment_start = self.adjustment_start_year - self.start_year  # start (T+x) of adjustment period
        self.adjustment_end_year = self.adjustment_start_year + self.adjustment_period - 1  # end year of adjustment period
        self.adjustment_end = self.adjustment_end_year - self.start_year  # end (T+x) of adjustment period
        self.ageing_cost_period = ageing_cost_period  # number of years during which ageing costs must be accounted for by SPB adjustment
        self.fiscal_multiplier = fiscal_multiplier  # fiscal multiplier for fiscal adjustment
        self.fiscal_multiplier_persistence = fiscal_multiplier_persistence  # persistence of fiscal multiplier
        self.fiscal_multiplier_type = fiscal_multiplier_type  # type of fiscal multiplier
        self.bond_data = bond_data  # True if bond level data is available
        self.policy_change = False # Turns true if projected with spb target/steps
        self.scenario = None # scenario parameter

        # Initiate model variables as numpy arrays
        nan_vars = [
            'rg_bl',                      # baseline growth rate
            'rg_pot_bl',                  # baseline potential growth rate
            'ng_bl',                      # baseline nominal growth rate
            'ngdp_bl',                    # baseline nominal GDP
            'rgdp_bl',                    # baseline real GDP
            'rgdp_pot_bl',                # baseline potential GDP
            'output_gap_bl',              # baseline output gap
            'rg',                         # real growth rate adjusted for fiscal_multiplier
            'ng',                         # nominal growth rate
            'ngdp',                       # nominal GDP adjusted for fiscal_multiplier
            'rgdp',                       # real GDP adjusted for fiscal_multiplier
            'rgdp_pot',                   # potential GDP
            'output_gap',                 # output gap
            'rg_pot',                     # potential growth rate
            'pi',                         # inflation rate
            'PB',                         # primary balance
            'pb',                         # primary balance over GDP
            'SPB',                        # structural primary balance
            'spb_bl',                     # baseline primary balance over GDP
            'spb_bca',                    # structural primary balance over GDP before cost of ageing
            'spb',                        # structural primary balance over GDP
            'spb_bca_adjustment',         # change in structural primary balance
            'GFN',                        # gross financing needs
            'OB',                         # fiscal balance
            'ob',                         # fiscal balance over GDP
            'SB',                         # structural balance
            'sb',                         # structural balance over GDP
            'net_expenditure_growth',     # expenditure growth rate
            'd',                          # debt to GDP ratio
            'D_share_lt_maturing',        # share of long-term debt maturing in the current year
            'repayment_st',               # repayment of short-term debt
            'repayment_lt',               # repayment of long-term debt
            'repayment',                  # repayment of total debt
            'interest_st',                # interest payment on short-term debt
            'interest_lt',                # interest payment on long-term debt
            'interest',                   # interest payment on total debt
            'interest_ratio',             # interest payment over GDP
            'i_st',                       # market interest rate on short-term debt
            'i_lt',                       # market interest rate on long-term debt
            'i_st_bl',                    # baseline market interest rate on short-term debt
            'i_lt_bl',                    # baseline market interest rate on long-term debt
            'exr_eur',                    # euro exchange rate
            'exr_usd',                    # usd exchange rate
            'iir_bl',                     # baseline implicit interest rate
            'alpha',                      # share of short-term debt in total debt
            'beta',                       # share of new long-term debt in total long-term debt
            'iir',                        # implicit interest rate
            'iir_lt',                     # implicit long-term interest rate
            'exr'                         # exchange rate
        ]

        for var in nan_vars:
            setattr(self, var, np.full(self.projection_period, np.nan, dtype=np.float64))

        zero_vars = [
            'fiscal_multiplier_effect',  # fiscal multiplier impulse
            'ageing_cost',               # ageing cost
            'ageing_component',          # ageing component of primary balance
            'revenue',                   # revenue
            'revenue_component',         # revenue component of primary balance
            # 'pension_revenue',           # pension revenue
            # 'pension_revenue_component', # pension revenue component of primary balance
            # 'property_income',           # property income
            # 'property_income_component', # property income component of primary balance
            'cyclical_component',        # cyclical component of primary balance
            'SF',                        # stock-flow adjustment
            'sf',                        # stock-flow adjustment over GDP
            'D',                         # total debt
            'D_lt',                      # total long-term debt
            'D_new_lt',                  # new long-term debt
            'D_lt_esm',                  # inst debt
            'D_st',                      # total short-term debt
            'repayment_lt_esm',          # repayment of inst debt
            'repayment_lt_bond'          # repayment of past bond issuance
        ]

        for var in zero_vars:
            setattr(self, var, np.full(self.projection_period, 0, dtype=np.float64))

        # Clean data
        self._clean_data()

    # ========================================================================================= #
    #                               DATA METHODS (INTERNAL)                                     #
    # ========================================================================================= #

    def _clean_data(self):
        """
        Import data from CSV deterministic input data file.
        """
        self._load_input_data()
        self._clean_rgdp_pot()
        self._clean_rgdp()
        self._calc_output_gap()
        self._clean_inflation()
        self._clean_ngdp()
        self._clean_debt()
        self._clean_esm_repayment()
        self._clean_debt_redemption()
        if self.bond_data: 
            self._clean_bond_repayment()
        self._clean_pb()
        self._clean_implicit_interest_rate()
        self._clean_market_rates()
        self._clean_stock_flow()
        self._clean_exchange_rate()
        self._clean_ageing_cost()
        # self._clean_pension_revenue()
        # self._clean_property_income()
        self._clean_revenue()

    def _load_input_data(self):
        """
        Load deterministic data from CSV file.
        """
        # Set base directory relative to code folder
        self._base_dir = '../' * (os.getcwd().split(os.sep)[::-1].index('code')+1)
        self.df_deterministic_data = pd.read_csv(self._base_dir + 'data/InputData/deterministic_data_november.csv') 
        self.df_deterministic_data = self.df_deterministic_data.loc[self.df_deterministic_data['COUNTRY'] == self.country].set_index('YEAR').iloc[:,1:]

    def _clean_rgdp_pot(self):
        """
        Clean baseline real potential growth.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            
            # potential growth is based on OGWG up to T+5, long-run estimates from 2033, interpoalted in between
            self.rg_pot_bl[t] = self.df_deterministic_data.loc[y, 'POTENTIAL_GDP_GROWTH']
            
            # potential GDP up to T+5 are from OGWG, after that projected based on growth rate
            if t <= 5:
                self.rgdp_pot_bl[t] = self.df_deterministic_data.loc[y, 'POTENTIAL_GDP']
            else:
                self.rgdp_pot_bl[t] = self.rgdp_pot_bl[t - 1] * (1 + self.rg_pot_bl[t] / 100) 

        # Set initial values to baseline
        self.rg_pot = np.copy(self.rg_pot_bl)   
        self.rgdp_pot = np.copy(self.rgdp_pot_bl)

    def _clean_rgdp(self):
        """
        Clean baseline real growth. Baseline refers to forecast values without fiscal multiplier effect.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            
            # potential GDP up to T+5 are from OGWG, after that projected based on growth rate
            if t <= 5:
                self.rgdp_bl[t] = self.df_deterministic_data.loc[y, 'REAL_GDP']
                self.rg_bl[t] = self.df_deterministic_data.loc[y, 'REAL_GDP_GROWTH']

            else:
                self.rgdp_bl[t] = self.rgdp_pot[t]
                self.rg_bl[t] = self.rg_pot[t]

        # Set initial values to baseline
        self.rg = np.copy(self.rg_bl)
        self.rgdp = np.copy(self.rgdp_bl)

    def _calc_output_gap(self):
        """
        Calculate the Output gap.
        """ 
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.output_gap_bl[t] = (self.rgdp_bl[t] / self.rgdp_pot[t] - 1) * 100

        # Set initial values to baseline
        self.output_gap = np.copy(self.output_gap_bl)

    def _clean_inflation(self):
        """
        Clean inflation rate data.
        """
        # Up to T+3 from Ameco GDP deflator
        for t, y in enumerate(range(self.start_year, self.start_year + 3)):
            self.pi[t] = self.df_deterministic_data.loc[y, 'GDP_DEFLATOR_PCH']

        # Set T+10 value based on inflation swaps, T+30 is 2 percent
        self.pi[10] = self.df_deterministic_data.loc[0, 'FWD_INFL_5Y5Y']
        self.pi[30] = 2

        # Poland and Romania T+10 target is increased by half of the difference with Euro Area in T+2
        # T+30 target is set to 2.5 for Poland and Romania, 3 for Hungary
        if self.country in ['POL', 'ROU']:
            self.pi[10] += (self.pi[2] - self.df_deterministic_data.loc[self.start_year+2, 'EA_GDP_DEFLATOR_PCH']) / 2
            self.pi[30] = 2.5
        elif self.country in ['HUN']:
            self.pi[30] += 1

        # Interpolate missing values
        x = np.arange(len(self.pi))
        mask_pi = np.isnan(self.pi)
        self.pi[mask_pi] = np.interp(x[mask_pi], x[~mask_pi], self.pi[~mask_pi])
        
    def _clean_ngdp(self):
        """
        Clean baseline nominal growth.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            
            # Up to T+3 from Ameco Nominal GDP, after that projected based on growth rate
            if t <= 2:
                self.ngdp_bl[t] = self.df_deterministic_data.loc[y, 'NOMINAL_GDP']
                self.ng_bl[t] = self.df_deterministic_data.loc[y, 'NOMINAL_GDP_GROWTH']
            else:
                self.ng_bl[t] = (1 + self.rg_bl[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100
                self.ngdp_bl[t] = self.ngdp_bl[t - 1] * (1 + self.ng_bl[t] / 100)

        # Set initial values to baseline
        self.ng = np.copy(self.ng_bl)
        self.ngdp = np.copy(self.ngdp_bl)

    def _clean_debt(self):
        """
        Clean debt data and parameters.
        """
        # Get baseline debt from Ameco
        for t, y in enumerate(range(self.start_year, self.start_year + 3)):
            self.d[t] = self.df_deterministic_data.loc[y, 'DEBT_RATIO']
            self.D[t] = self.df_deterministic_data.loc[y, 'DEBT_TOTAL']
        
        # Set maturity shares and average maturity
        self.D_share_st = self.df_deterministic_data.loc[0, 'DEBT_ST_SHARE']
        self.D_share_lt = 1 - self.D_share_st
        self.D_share_lt_maturing_T = self.df_deterministic_data.loc[0, 'DEBT_LT_MATURING_SHARE']
        self.D_share_lt_mat_avg = self.df_deterministic_data.loc[0, 'DEBT_LT_MATURING_AVG_SHARE']
        self.avg_res_mat = np.min([round((1 / self.D_share_lt_mat_avg)), 30])

        # Set share of domestic, euro and usd debt, ensure no double counting for non-euro countries
        self.D_share_domestic = np.round(self.df_deterministic_data.loc[0, 'DEBT_DOMESTIC_SHARE'], 4)
        if self.country in ['BGR', 'CZE', 'DNK', 'HUN', 'POL', 'ROU', 'SWE', 'GBR', 'USA']:
            self.D_share_eur = np.round(self.df_deterministic_data.loc[0, 'DEBT_EUR_SHARE'], 4)
        else:
            self.D_share_eur = 0
        if self.country != 'USA':
            self.D_share_usd = np.round(1 - self.D_share_domestic - self.D_share_eur, 4)
        else:
            self.D_share_usd = 0

        # Set initial values for long and short-term debt
        self.D_st[0] = self.D_share_st * self.D[0]
        self.D_lt[0] = self.D_share_lt * self.D[0] 

    def _clean_esm_repayment(self):
        """
        Clean institutional debt data.
        """
        # Set to zero if missing
        self.df_deterministic_data['ESM_REPAYMENT'] = self.df_deterministic_data['ESM_REPAYMENT'].fillna(0)

        # Calculate initial value of institutional debt 
        self.D_lt_esm[0] = self.df_deterministic_data['ESM_REPAYMENT'].sum()
        
        # Import ESM institutional debt repayments
        for t, y in enumerate(range(self.start_year + 1, self.end_year + 1)):
            self.repayment_lt_esm[t] = self.df_deterministic_data.loc[y, 'ESM_REPAYMENT']
            self.D_lt_esm[t] = self.D_lt_esm[t - 1] - self.repayment_lt_esm[t] if t > 0 else self.D_lt_esm[0]

    def _clean_debt_redemption(self):
        """
        Clean debt redemption data for institutional debt.
        """
        # Set T and T + 10 value of maturing lt debt share
        self.D_share_lt_maturing[0] = self.D_share_lt_maturing_T
        self.D_share_lt_maturing[10:] = self.D_share_lt_mat_avg

        # Interpolate missing values
        x = np.arange(len(self.D_share_lt_maturing))
        mask = np.isnan(self.D_share_lt_maturing)
        self.D_share_lt_maturing[mask] = np.interp(x[mask], x[~mask], self.D_share_lt_maturing[~mask])
    
    def _clean_bond_repayment(self):
        """
        Clean long-term bond repayment data.
        """
        # Import bond repayment data
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.repayment_lt_bond[t] = self.df_deterministic_data.loc[y, 'BOND_REPAYMENT']
    
    def _clean_pb(self):
        """
        Clean structural primary balance.
        """
        # Get baseline spb from Ameco
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            if t <= 2:
                self.spb_bl[t] = self.df_deterministic_data.loc[y, 'STRUCTURAL_PRIMARY_BALANCE']
                self.SPB[t] = self.spb_bl[t] / 100 * self.ngdp_bl[t]
                self.pb[t] = self.df_deterministic_data.loc[y, 'PRIMARY_BALANCE']
                self.PB[t] = self.pb[t] / 100 * self.ngdp_bl[t]
                self.ob[t] = self.df_deterministic_data.loc[y, 'FISCAL_BALANCE']
                self.OB[t] = self.ob[t] / 100 * self.ngdp_bl[t]
                self.sb[t] = self.spb_bl[t] + (self.ob[t] * self.ngdp_bl[t] - self.pb[t] * self.ngdp_bl[t]) / self.ngdp_bl[t]
                self.SB[t] = self.sb[t] / 100 * self.ngdp_bl[t]
            else:
                self.spb_bl[t] = self.spb_bl[t - 1]
                self.pb[t] = self.pb[t - 1]
        
        # Set initial values to baseline
        self.spb_bca = np.copy(self.spb_bl)
        self.spb = np.copy(self.spb_bl)

        # Get budget balance semi-elasticity
        self.budget_balance_elasticity = self.df_deterministic_data.loc[0, 'BUDGET_BALANCE_ELASTICITY']

        # Get primary expenditure share
        self.expenditure_share = self.df_deterministic_data.loc[2024, 'PRIMARY_EXPENDITURE_SHARE']

    def _clean_implicit_interest_rate(self):
        """
        Clean implicit interest rate.
        """
        # Get implicit interest rate from Ameco
        for t, y in enumerate(range(self.start_year, self.start_year + 3)):
            self.iir_bl[t] = self.df_deterministic_data.loc[y, 'IMPLICIT_INTEREST_RATE']

        # Set initial values to baseline
        self.iir = np.copy(self.iir_bl)

        # Initial lt baseline
        self.iir_lt[0] = self.iir[0] * (1 - self.D_share_st)
    
    def _clean_market_rates(self):
        """
        Clean forward Bloomberg forward and benchmark rates. Interpolate missing values.
        """
        # Get benchmark rates for first years
        for t, y in enumerate(range(self.start_year, self.start_year + 2)):
            self.i_st_bl[t] = self.df_deterministic_data.loc[y, 'INTEREST_RATE_ST']
            self.i_lt_bl[t] = self.df_deterministic_data.loc[y, 'INTEREST_RATE_LT']

        # Load 10 year forward rates
        self.fwd_rate_st = self.df_deterministic_data.loc[0, 'FWD_RATE_3M10Y']
        self.fwd_rate_lt = self.df_deterministic_data.loc[0, 'FWD_RATE_10Y10Y']

        # Clean vectors in case of repeated projection with different scenarios
        self.i_st_bl[2:] = np.nan
        self.i_lt_bl[2:] = np.nan

        # Set T + 10 value as market fwd rate
        self.i_st_bl[10] = self.fwd_rate_st
        self.i_lt_bl[10] = self.fwd_rate_lt

        # Set t + 30 values
        if self.country in ['POL', 'ROU']: 
            self.i_lt_bl[30:] = 4.5
        elif self.country in ['HUN']: 
            self.i_lt_bl[30:] = 5
        else: 
            self.i_lt_bl[30:] = 4

        yield_curve_coef = 0.5
        self.i_st_bl[30:] = self.i_lt[30] * yield_curve_coef

        # Interpolate missing values
        x_st = np.arange(len(self.i_st_bl))
        mask_st = np.isnan(self.i_st_bl)
        self.i_st_bl[mask_st] = np.interp(x_st[mask_st], x_st[~mask_st], self.i_st_bl[~mask_st])

        x_lt = np.arange(len(self.i_lt_bl))
        mask_lt = np.isnan(self.i_lt_bl)
        self.i_lt_bl[mask_lt] = np.interp(x_lt[mask_lt], x_lt[~mask_lt], self.i_lt_bl[~mask_lt])

        # Set initial values to baseline
        self.i_st = np.copy(self.i_st_bl)
        self.i_lt = np.copy(self.i_lt_bl)

    def _clean_stock_flow(self):
        """
        Clean stock flow adjustment.
        """
        # Get stock-flow adjustment from Ameco
        for t, y in enumerate(range(self.start_year, self.start_year + 3)):
            self.SF[t] = self.df_deterministic_data.loc[y, 'STOCK_FLOW']

        # For Luxembourg, Finland, get Pension balance ratio for projection
        if self.country in ['LUX', 'FIN']:
            self.pension_balance = np.full(self.projection_period, 0, dtype=np.float64)
            for t, y in enumerate(range(self.start_year, self.end_year + 1)):
                self.pension_balance[t] = self.df_deterministic_data.loc[y, 'PENSION_BALANCE']

    def _clean_exchange_rate(self):
        """
        Clean exchange rate data for non-euro countries.
        """
        # Get exchange rate data from Ameco
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):

            if t <= 2:
                self.exr_eur[t] = self.df_deterministic_data.loc[y, 'EXR_EUR']
                self.exr_usd[t] = self.df_deterministic_data.loc[y, 'EXR_USD']
            else:
                self.exr_usd[t] = self.exr_usd[t - 1]
                self.exr_eur[t] = self.exr_eur[t - 1]

    def _clean_ageing_cost(self):
        """
        Clean ageing cost data.
        """
        # Import ageing costs from Ageing Report data
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.ageing_cost[t] = self.df_deterministic_data.loc[y, 'AGEING_COST']
    
    def _clean_pension_revenue(self):
        """
        Clean pension revenue data.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.pension_revenue[t] = self.df_deterministic_data.loc[y, 'PENSION_REVENUE'] if not np.isnan(self.df_deterministic_data.loc[y, 'PENSION_REVENUE']) else 0

    def _clean_property_income(self):
        """
        Clean property income data.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.property_income[t] = self.df_deterministic_data.loc[y, 'PROPERTY_INCOME'] if not np.isnan(self.df_deterministic_data.loc[y, 'PROPERTY_INCOME']) else 0
    
    def _clean_revenue(self):
        """
        Clean property income and pension revenue data. (Raw data are relative changes to 2024)
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.revenue[t] = self.df_deterministic_data.loc[y, 'TAX_AND_PROPERTY_INCOME'] if not np.isnan(self.df_deterministic_data.loc[y, 'TAX_AND_PROPERTY_INCOME']) else 0
    
    # ========================================================================================= #
    #                                   PROJECTION METHODS                                      #
    # ========================================================================================= #

    def project(self,
                spb_target=None,
                spb_steps=None,  # list of annual adjustment steps during adjustment
                edp_steps=None,  # list of annual adjustment steps during EDP
                deficit_resilience_steps=None,  # list of years during adjustment where minimum step size is enforced
                post_spb_steps=None,  # list of years after adjustment where minimum step size is enforced
                scenario='main_adjustment',  # scenario parameter, needed for DSA criteria
                ):
        """
        Project debt dynamics
        """

        # Reset starting values
        self._reset_starting_values()

        # Set adjustment targets and steps
        self._set_adjustment(spb_target, spb_steps, edp_steps, deficit_resilience_steps, post_spb_steps)

        # Set scenario parameter
        self.scenario = scenario

        # Project debt dynamics
        self._project_net_expenditure_path()
        self._project_gdp()
        self._project_stock_flow()
        self._project_spb()
        self._project_pb_from_spb()
        self._project_debt_ratio()

    def _reset_starting_values(self):
        """
        Reset starting values for projection to avoid cumulative change from scenario application.
        """
        # Reset starting values for market rates
        self.i_st = np.copy(self.i_st_bl)
        self.i_lt = np.copy(self.i_lt_bl)

        # Reset starting values for growth
        self.rgdp = np.copy(self.rgdp_bl)
        self.rg = np.copy(self.rg_bl)
        self.rg_pot = np.copy(self.rg_pot_bl)
        self.rgdp_pot = np.copy(self.rgdp_pot_bl)

        # Reset starting values for debt issuance and implicit interest rate
        self.D_new_lt = np.full(self.projection_period, 0, dtype=np.float64)
        self.iir = np.copy(self.iir_bl)
        self.iir_lt[0] = self.iir[0] * (1 - self.D_share_st)

    def _set_adjustment(self, spb_target, spb_steps, edp_steps, deficit_resilience_steps, post_spb_steps):
        """
        Set adjustment parameters steps or targets depending on input
        """
        # Set spb_target
        if (spb_target is None
                and spb_steps is None):
            self.policy_change = False
            self.spb_target = self.spb_bca[self.adjustment_start - 1]
        elif (spb_target is None
              and spb_steps is not None):
            self.policy_change = True
            self.spb_target = self.spb_bca[self.adjustment_start - 1] + spb_steps.sum()
        else:
            self.policy_change = True
            self.spb_target = spb_target

        # Set adjustment steps
        if (spb_steps is None
                and spb_target is not None):
            # If adjustment steps are predifined, adjust only non-nan values
            if hasattr(self, 'predefined_spb_steps'):
                self.spb_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)
                num_predefined_steps = len(self.predefined_spb_steps)
                self.spb_steps[:num_predefined_steps] = np.copy(self.predefined_spb_steps)
                num_steps = self.adjustment_period - num_predefined_steps
                step_size = (spb_target - self.spb_bca[self.adjustment_start + num_predefined_steps - 1]) / num_steps
                self.spb_steps[num_predefined_steps:] = np.full(num_steps, step_size)
            else:
                self.spb_steps = np.full((self.adjustment_period,), (self.spb_target - self.spb_bca[self.adjustment_start - 1]) / self.adjustment_period, dtype=np.float64)
        elif (spb_steps is None
              and spb_target is None):
            self.spb_steps = np.full((self.adjustment_period,), 0, dtype=np.float64)
        else:
            self.spb_steps = spb_steps

        # Set edp steps
        if edp_steps is None:
            self.edp_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)
        else:
            self.edp_steps = edp_steps

        # Set deficit resilience steps
        if deficit_resilience_steps is None:
            self.deficit_resilience_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)
        else:
            self.deficit_resilience_steps = deficit_resilience_steps

        # Set post adjustment steps
        if post_spb_steps is None:
            self.post_spb_steps = np.full((self.projection_period - self.adjustment_end - 1,), 0, dtype=np.float64)
        else:
            self.post_spb_steps = post_spb_steps

    def _project_net_expenditure_path(self):
        """
        Project structural primary balance, excluding ageing cost
        """
        # Adjust path for EDP and deficit resilience steps
        self._adjust_for_edp()
        self._adjust_for_deficit_resilience()
        self._apply_spb_steps()

        # If lower_spb scenario, adjust path
        if self.scenario == 'lower_spb':
            self._apply_lower_spb()

    def _adjust_for_edp(self):
        """
        Adjust linear path for minimum EDP adjustment steps
        """
        # Save copy of baseline adjustment steps
        self.spb_steps_baseline = np.copy(self.spb_steps)

        # Apply EDP steps to adjustment steps
        self.spb_steps[~np.isnan(self.edp_steps)] = np.where(
            self.edp_steps[~np.isnan(self.edp_steps)] > self.spb_steps[~np.isnan(self.edp_steps)],
            self.edp_steps[~np.isnan(self.edp_steps)],
            self.spb_steps[~np.isnan(self.edp_steps)]
        )

        # Identify periods that are after EDP and correct them for frontloading
        if not np.isnan(self.edp_steps).all():
            last_edp_index = np.where(~np.isnan(self.edp_steps))[0][-1]
        else:
            last_edp_index = 0
        post_edp_index = np.arange(last_edp_index + 1, len(self.spb_steps))
        self.diff_adjustment_baseline = np.sum(self.spb_steps_baseline - self.spb_steps)
        offset_edp = self.diff_adjustment_baseline / len(post_edp_index) if len(post_edp_index) > 0 else 0
        self.spb_steps[post_edp_index] += offset_edp

    def _adjust_for_deficit_resilience(self):
        """
        Adjust linear path for minimum deficit resilience adjustment steps
        """
        # Save copy of edp adjusted steps
        self.spb_steps_baseline = np.copy(self.spb_steps)

        # Apply deficit resilience safeguard steps to adjustment steps
        self.spb_steps[~np.isnan(self.deficit_resilience_steps)] = np.where(
            self.deficit_resilience_steps[~np.isnan(self.deficit_resilience_steps)] > self.spb_steps[~np.isnan(self.deficit_resilience_steps)],
            self.deficit_resilience_steps[~np.isnan(self.deficit_resilience_steps)],
            self.spb_steps[~np.isnan(self.deficit_resilience_steps)]
        )

        # Identify periods that are after EDP and deficit resilience and correct for frontloading
        if not (np.isnan(self.edp_steps).all()
                and np.isnan(self.deficit_resilience_steps).all()):
            last_edp_deficit_resilience_index = np.where(~np.isnan(self.edp_steps) | ~np.isnan(self.deficit_resilience_steps))[0][-1]
        else:
            last_edp_deficit_resilience_index = 0
        post_edp_deficit_resilience_index = np.arange(last_edp_deficit_resilience_index + 1, len(self.spb_steps))
        self.diff_adjustment_baseline = np.sum(self.spb_steps_baseline - self.spb_steps)
        self.offset_deficit_resilience = self.diff_adjustment_baseline / len(post_edp_deficit_resilience_index) if len(post_edp_deficit_resilience_index) > 0 else 0
        self.spb_steps[post_edp_deficit_resilience_index] += self.offset_deficit_resilience

    def _apply_spb_steps(self):
        """
        Project spb_bca
        """
        # Apply adjustment steps based on the current period
        for t in range(self.adjustment_start, self.projection_period):
            if t in range(self.adjustment_start, self.adjustment_end + 1):
                self.spb_bca[t] = self.spb_bca[t - 1] + self.spb_steps[t - self.adjustment_start]
            else:
                self.spb_bca[t] = self.spb_bca[t - 1] + self.post_spb_steps[t - self.adjustment_end - 1]

        # Save adjustment step size
        self.spb_bca_adjustment[1:] = np.diff(self.spb_bca)

    def _apply_lower_spb(self):
        """
        Apply lower_spb scenario
        """
        if not hasattr(self, 'lower_spb_shock'):
            self.lower_spb_shock = 0.5
        # If 4-year adjustment period, spb_bca decreases by 0.5 for 2 years after adjustment period, if 7-year for 3 years
        lower_spb_adjustment_period = int(np.floor(self.adjustment_period / 2))
        for t in range(self.adjustment_end + 1, self.projection_period):
            if t <= self.adjustment_end + lower_spb_adjustment_period:
                self.spb_bca[t] -= self.lower_spb_shock / lower_spb_adjustment_period * (t - self.adjustment_end)
            else:
                self.spb_bca[t] = self.spb_bca[t - 1]

    def _project_gdp(self):
        """
        Project nominal GDP.
        """ 
        # Project real growth and apply fiscal multiplier
        if self.fiscal_multiplier_type == 'com': 
            self._calc_rgdp_com()
        elif self.fiscal_multiplier_type == 'pers': 
            self._calc_rgdp_pers()
        else : 
            raise ValueError('Fiscal multiplier type not recognized')
            
        # Apply adverse r-g scenario if specified
        if self.scenario == 'adverse_r_g': 
            self._apply_adverse_r_g()

        # Project nominal growth
        self._calc_ngdp()

    def _calc_rgdp_pers(self):
        """
        Calculates real GDP and real growth, assumes persistence in fiscal_multiplier effect leading to output gap closing in 3 years
        """
        for t in range(1, self.projection_period):

            # Fiscal multiplier effect from change in SPB relative to baseline
            self.fiscal_multiplier_effect[t] = self.fiscal_multiplier * ((self.spb_bca[t] - self.spb_bca[t - 1]) - (self.spb_bl[t] - self.spb_bl[t - 1]))

            # Add spillover effect to fiscal_multiplier effect if defined
            if hasattr(self, 'fiscal_multiplier_spillover'): 
                self.fiscal_multiplier_effect[t] += self.fiscal_multiplier_spillover[t]

            # Calculate persistence term of multiplier effect
            persistence_term = sum([self.fiscal_multiplier_effect[t - i] * (self.fiscal_multiplier_persistence - i) / self.fiscal_multiplier_persistence for i in range(1, self.fiscal_multiplier_persistence)])

            # Fiscal multiplier effect on output gap
            self.output_gap[t] = self.output_gap_bl[t] - self.fiscal_multiplier_effect[t] - persistence_term

            # Real growth and real GDP
            self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]
            self.rg[t] = (self.rgdp[t] - self.rgdp[t - 1]) / self.rgdp[t - 1] * 100

    def _calc_rgdp_com(self):
        """
        Calculates real GDP and real growth, assumes output gap closes in 3 years with 2/3 and 1/3 rule
        """
        for t in range(1, self.projection_period):

            # Fiscal multiplier effect from change in SPB relative to baseline
            self.fiscal_multiplier_effect[t] = self.fiscal_multiplier * ((self.spb_bca[t] - self.spb_bca[t - 1]) - (self.spb_bl[t] - self.spb_bl[t - 1]))

            # Add spillover effect to fiscal_multiplier effect if defined
            if hasattr(self, 'fiscal_multiplier_spillover'): 
                self.fiscal_multiplier_effect[t] += self.fiscal_multiplier_spillover[t]

            # Output gap
            if t == self.adjustment_start: 
                self.output_gap[t] = self.output_gap_bl[t] - self.fiscal_multiplier_effect[t]

            elif t in range(self.adjustment_start + 1, self.adjustment_end + 1 ) and self.policy_change:
                self.output_gap[t] = (self.fiscal_multiplier_persistence - 1) / self.fiscal_multiplier_persistence * self.output_gap[t-1] - self.fiscal_multiplier_effect[t]
                #self.output_gap[t] = 2 / 3 * self.output_gap[t-1] - self.fiscal_multiplier_effect[t]
            
            elif t in range(self.adjustment_end + 1, self.adjustment_end + self.fiscal_multiplier_persistence + 1) and self.policy_change:
                self.output_gap[t] = self.output_gap[t-1] - 1 / self.fiscal_multiplier_persistence * self.output_gap[self.adjustment_end]
                    
            # Real growth and real GDP
            self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]
            self.rg[t] = (self.rgdp[t] - self.rgdp[t - 1]) / self.rgdp[t - 1] * 100

    def _apply_adverse_r_g(self):
        """
        Applies adverse interest rate and growth conditions for adverse r-g scenario
        """
        if not hasattr(self, 'adverse_r_g_shock'): 
            self.adverse_r_g_shock = 0.5

        for t in range(self.adjustment_end+1, self.projection_period):
            
            # Increase short and long term interest rates by 0.5
            self.i_st[t] += self.adverse_r_g_shock
            self.i_lt[t] += self.adverse_r_g_shock

            # Decrease real and potential growth by 0.5
            self.rg[t] -= self.adverse_r_g_shock
            self.rgdp[t] = self.rgdp[t - 1] * (1 + (self.rg[t]) / 100)
            
    def _calc_ngdp(self):
        """
        Calculates nominal GDP and nominal growth
        """
        for t in range(self.adjustment_start, self.projection_period):
                
            # From adjustment start, nominal growth based on real growth and inflation
                self.ng[t] = (1 + self.rg[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100
                self.ngdp[t] = self.ngdp[t - 1] * (1 + self.ng[t] / 100)

    def _project_stock_flow(self):
        """
        Calculate stock-flow adjustment as share of NGDP
        For specification of exceptions see DSM2023
        """
        for t in range(self.projection_period):

            # For Luxembourg and Finland stock flow is extended beyond T+2 using pension balance ratio
            if self.country in ['LUX']: #, include FInland here if using Commission assumptions
                
                if t < 3:
                    self.sf[t] = self.SF[t] / self.ngdp[t] * 100

                # From T+3 to T+10 apply percentage change of pension balance ratio
                if t >= 3 and t <= 10:
                    self.sf[t] = self.pension_balance[t]

                # Commission assumptions for Finland linearly interpolate to 0 from T+11 to T+20
                elif self.country == 'FIN' and t > 10 and t <= 20:
                    self.sf[t] = self.sf[10] - (t - 10) * self.sf[10] / 10

                # For Luxembourg, linearly interpolate to 0 from T+11 to T+24
                elif self.country == 'LUX' and t > 10 and t <= 24:
                    self.sf[t] = self.sf[10] - (t - 10) * self.sf[10] / 14
            
            # updated sf for Finland based on their 2024 MTFP numbers, comment out if using Commission assumptions
            elif self.country == 'FIN':
                sf_fin = {
                    2024: 2.5, 2025: 1.3, 2026: 1.7, 2027: 1.4, 2028: 1.4, 2029: 1.0, 2030: 1.1, 2031: 1.5, 2032: 1.5, 
                    2033: 1.4, 2034: 1.2, 2035: 1.1, 2036: 0.9, 2037: 0.8, 2038: 0.6, 2039: 0.5, 2040: 0.3, 2041: 0.2
                }
                if t in range(2024-self.start_year,len(sf_fin)+1):
                    self.sf[t] = sf_fin[t+self.start_year]

            # For Greece stock-flow is based on deferal of ESM/EFSF interest payments
            elif self.country == 'GRC':

                # Stock flow is 5.4% in 2022
                if t == 0:
                    self.sf[t] = - 5.4

                # Increases to 11.1% in 2032 in cumulative terms
                elif t > 0 and t <= 9:
                    self.sf[t] = (- (11.1 / 100 * self.ngdp[9] 
                                     + np.sum(self.SF[:t])) 
                                     / (10-t) 
                                     / self.ngdp[t] * 100)

                # Decline to zero by projection end, 2070
                elif t > 9: 
                    self.sf[t] = (- np.sum(self.SF[:10]) 
                                  / (self.projection_period - 9) 
                                  / self.ngdp[t] * 100)
            
            # For other countries stock flow is simply based on Ameco data
            else:
                self.sf[t] = self.SF[t] / self.ngdp[t] * 100

            # Project SF for all countries
            self.SF[t] = self.sf[t] / 100 * self.ngdp[t]
            

    def _project_spb(self):
        """
        Project structural primary balance
        """
        for t in range(1, self.projection_period):

            # After adjustment period ageing costs affect the SPB for duration of "ageing_cost_period"
            if t > self.adjustment_end and t <= self.adjustment_end + self.ageing_cost_period:
                self.ageing_component[t] = self.ageing_cost[t] - self.ageing_cost[self.adjustment_end]
                # self.pension_revenue_component[t] = self.pension_revenue[t] - self.pension_revenue[self.adjustment_end]
                # self.property_income_component[t] = self.property_income[t] - self.property_income[self.adjustment_end]
                self.revenue_component[t] = self.revenue[t] - self.revenue[self.adjustment_end]
            elif t > self.adjustment_end + self.ageing_cost_period:
                self.ageing_component[t] = self.ageing_component[t-1]
                # self.pension_revenue_component[t] = self.pension_revenue_component[t-1]
                # self.property_income_component[t] = self.property_income_component[t-1]   
                self.revenue_component[t] = self.revenue_component[t-1]

            self.spb[t] = self.spb_bca[t] - self.ageing_component[t] + self.revenue_component[t] #+ self.pension_revenue_component[t] + self.property_income_component[t]

            # Total SPB for calcualtion of structural deficit
            self.SPB[t] = self.spb[t] / 100 * self.ngdp[t]

            # Calculate expenditure growth
            self.net_expenditure_growth[t] = self.ng[t] - (self.spb[t] - self.spb[t - 1])/self.expenditure_share * 100

    def _project_pb_from_spb(self):
        """
        Project primary balance adjusted as sum of SPB, cyclical component.
        """
        for t in range(self.projection_period):

            # Calculate components
            self.cyclical_component[t] = self.budget_balance_elasticity * self.output_gap[t]

            # Calculate primary balance ratio as sum of components and total primary balance
            self.pb[t] = self.spb[t] + self.cyclical_component[t]
            self.PB[t] = self.pb[t] / 100 * self.ngdp[t]

    def _project_debt_ratio(self):
        """
        Main loop for debt dynamics
        """
        for t in range(1, self.projection_period):

            # Apply financial stress scenario if specified
            if self.scenario == 'financial_stress' and t == self.adjustment_end + 1:
                self._apply_financial_stress(t)

            # Calculate implicit interest rate, interestst, repayments, gross financing needs, debt stock, overall balance, and debt ratio
            self._calc_iir(t)
            self._calc_interest(t)
            self._calc_repayment(t)
            self._calc_gfn(t)
            self._calc_debt_stock(t)
            if t >= self.adjustment_start: # We keep input data for 2024
                self._calc_balance(t)
                self._calc_debt_ratio(t)

    def _apply_financial_stress(self, t):
        """
        Adjust interest rates for financial stress scenario
        """
        if not hasattr(self, 'financial_stress_shock'):
            self.financial_stress_shock = 1
        # Adjust market rates for high debt countries financial stress scenario
        if self.d[self.adjustment_end] > 90:
            self.i_st[t] += (self.financial_stress_shock + (self.d[self.adjustment_end] - 90) * 0.06)
            self.i_lt[t] += (self.financial_stress_shock + (self.d[self.adjustment_end] - 90) * 0.06)

        # Adjust market rates for low debt countries financial stress scenario
        else:
            self.i_st[t] += self.financial_stress_shock
            self.i_lt[t] += self.financial_stress_shock

    def _calc_iir(self, t):
        """
        Calculate implicit interest rate
        """
        # Calculate the shares of short term and long term debt in total debt
        self.alpha[t - 1] = self.D_st[t - 1] / self.D[t - 1]
        self.beta[t - 1] = self.D_new_lt[t - 1] / self.D_lt[t - 1]

        # Use ameco implied interest until T+3 and derive iir_lt
        if t <= 2:
            self.iir_lt[t] = (self.iir[t] - self.alpha[t - 1] * self.i_st[t]) / (1 - self.alpha[t - 1])
            self.iir[t] = self.iir_bl[t]

        # Use DSM 2023 Annex A3 formulation after
        else:
            self.iir_lt[t] = self.beta[t - 1] * self.i_lt[t] + (1 - self.beta[t - 1]) * self.iir_lt[t - 1]
            self.iir[t] = self.alpha[t - 1] * self.i_st[t] + (1 - self.alpha[t - 1]) * self.iir_lt[t]

        # Replace all 10 < iir < 0 with previous period value to avoid implausible values
        for iir in [self.iir, self.iir_lt]:
            if iir[t] < 0 or iir[t] > 10 or np.isnan(iir[t]):
                iir[t] = iir[t - 1]

    def _calc_interest(self, t):
        """
        Calculate interest payments on newly issued debt
        """
        self.interest_st[t] = self.D_st[t - 1] * self.i_st[t - 1] / 100  # interest payments on newly issued short-term debt
        self.interest_lt[t] = self.iir_lt[t] / 100 * self.D_lt[t - 1]  # lt interest is t-1 lt debt times implicit lt interest rate
        self.interest[t] = self.interest_st[t] + self.interest_lt[t]  # interest payments on newly issued debt and outstanding legacy debt
        self.interest_ratio[t] = self.interest[t] / self.ngdp[t] * 100

    def _calc_repayment(self, t):
        """
        Calculate repayment of newly issued debt
        """
        self.repayment_st[t] = self.D_st[t - 1]  # repayment payments on short-term debt share in last years gross financing needs

        # If bond data is True, add repayemnt of new issuance to legacy debt repayment_lt from _clean_bond_repayment method 
        if self.bond_data:
            self.repayment_lt[t] = np.sum(self.D_new_lt[np.max([0, t - 20]) : t] / 20) # Average maturity of new issuance is 10 years, spread evenly over 20 years

        # If bond data is false, repayment share is simply a function of last periods debt stock
        else:
            self.repayment_lt[t] = self.D_share_lt_maturing[t] * self.D_lt[t - 1]

        # Calculate total repayment
        self.repayment[t] = self.repayment_st[t] + self.repayment_lt[t] + self.repayment_lt_bond[t] + self.repayment_lt_esm[t]

    def _calc_gfn(self, t):
        """
        Calculate gross financing needs
        """
        self.GFN[t] = np.max([self.interest[t] + self.repayment[t] - self.PB[t] + self.SF[t], 0])

    def _calc_debt_stock(self, t):
        """
        Calculate new debt stock and distribution of new short and long-term issuance
        """
        # Total debt stock is equal to last period stock minus repayment plus financing needs
        self.D[t] = np.max([self.D[t - 1] - self.repayment[t] + self.GFN[t], 0])

        # Distribution of short-term and long-term debt in financing needs
        D_theoretical_issuance_st = self.D_share_st * self.D[t]  # st debt to keep share equal to D_share_st
        D_theoretical_issuance_lt = np.max([(1 - self.D_share_st) * self.D[t] - (self.D_lt[t - 1] - self.repayment_lt[t] - self.repayment_lt_bond[t]), 0]) # lt debt to keep share equal to 1 - D_share_st, non-negative
        D_issuance_share_st = D_theoretical_issuance_st / (D_theoretical_issuance_st + D_theoretical_issuance_lt)  # share of st in gfn
        
        # Calculate short-term and long-term debt issuance
        self.D_st[t] = D_issuance_share_st * self.GFN[t]
        self.D_new_lt[t] = (1 - D_issuance_share_st) * self.GFN[t]
        self.D_lt[t] = np.max([self.D_lt[t - 1] - self.repayment_lt[t] - self.repayment_lt_bond[t] + self.D_new_lt[t] , 0])

    def _calc_balance(self, t):
        """
        Calculate overall balance and structural fiscal balance
        """
        self.OB[t] = self.PB[t] - self.interest[t]  # overall balance
        self.SB[t] = self.SPB[t] - self.interest[t]  # structural balance
        self.ob[t] = self.OB[t] / self.ngdp[t] * 100 # overall balance as share of NGDP
        self.sb[t] = self.SB[t] / self.ngdp[t] * 100 # structural balance as share of NGDP

    def _calc_debt_ratio(self, t):
        """
        Calculate debt ratio (zero floor)
        """
        self.d[t] = np.max([
            self.D_share_domestic * self.d[t - 1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100)
            + self.D_share_eur * self.d[t - 1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_eur[t] / self.exr_eur[t - 1])
            + self.D_share_usd * self.d[t - 1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_usd[t] / self.exr_usd[t - 1])
            - self.pb[t] + self.sf[t], 0
        ])
        
    # ========================================================================================= #
    #                               OPTIMIZATION METHODS                                        #
    # ========================================================================================= #

    def find_edp(self, spb_target=None):
        """
        Find the number of periods needed to correct an excessive deficit if possible within adjustment period.
        """
        # Project baseline and check if deficit is excessive
        if spb_target is None:
            self.spb_target = None
        else:
            self.spb_target = spb_target
        self.project(spb_target=spb_target)

        # Define EDP threshold, set to 3% of GDP
        self.edp_target = -3

        # If deficit excessive, increase spb by 0.5 annually until deficit below 3%
        if self.ob[self.adjustment_start] < self.edp_target:

            # Set start indices for spb and pb adjustment parts of EDP
            self.edp_spb_index = 0
            self.edp_sb_index = 3

            # Calculate EDP adjustment steps for spb, sb, and final periods
            self._calc_edp_spb()
            self._calc_edp_sb()
            self._calc_edp_end(spb_target=spb_target)

        # If excessive deficit in year before adjustment start, set edp_end to year before adjustment start
        elif self.ob[self.adjustment_start - 1] < self.edp_target:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 1
            
        # If deficit not excessive, set EDP period to 0
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 2

    def _save_edp_period(self):
        """
        Saves EDP period and end period
        """
        self.edp_period = np.where(~np.isnan(self.edp_steps))[0][-1] + 1
        self.edp_end = self.adjustment_start + self.edp_period - 1        

    def _calc_edp_spb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral primary balance adjustment
        """
        # Loop for SPB part of EDP: min. 0.5 spb adjustment while deficit > 3 and in spb adjustmet period
        while (self.ob[self.adjustment_start + self.edp_spb_index] <= self.edp_target
                and self.edp_spb_index < self.edp_sb_index):
            
            # Set EDP step to 0.5
            self.edp_steps[self.edp_spb_index] = 0.5

            # Project using last periods SPB as target, move to next period
            self.project(
                spb_target=self.spb_target,
                edp_steps=self.edp_steps
            )
            self.edp_spb_index += 1
            self._save_edp_period()

    def _calc_edp_sb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral balance adjustment
        """
        # Loop for SB balance part of EDP: min. 0.5 ob adjustment while deficit > 3 and before last period
        while (self.ob[self.adjustment_start + self.edp_sb_index] <= self.edp_target
                and self.edp_sb_index + 1 <= self.adjustment_period):
            
            # If sb adjustment is less than 0.5, increase by 0.001
            while (self.sb[self.adjustment_start + self.edp_sb_index] 
                   - self.sb[self.adjustment_start + self.edp_sb_index - 1] < 0.5):
            
                # Initiate sb step at current adjustment_step value, increase by 0.001
                self.edp_steps[self.edp_sb_index] = self.spb_steps[self.edp_sb_index]
                self.edp_steps[self.edp_sb_index] += 0.001

                # Project using last periods SPB as target, move to next period
                self.project(
                    spb_target=self.spb_target,
                    edp_steps=self.edp_steps
                )

            # If sb adjustment reaches min. 0.5, move to next period
            if self.sb[self.adjustment_start + self.edp_sb_index] - self.sb[self.adjustment_start + self.edp_sb_index - 1] >= 0.5:

                # set edp step to spb step in this period to ensure EDP recorded even in cases where step exceeds 0.5
                self.edp_steps[self.edp_sb_index] = self.spb_steps[self.edp_sb_index]
                self.edp_sb_index += 1
                self._save_edp_period()

    def _calc_edp_end(self, spb_target):
        """
        Calculate EDP adjustment steps or SPB target ensuring deficit below 3% at adjustment end
        """
        # If EDP lasts until penultimate adjustmet period, increase EDP steps to ensure deficit < 3
        if self.edp_period == self.adjustment_period:
            while self.ob[self.adjustment_end] < self.edp_target:

                # Aim for linear adjustment path by increasing smallest EDP steps first
                min_edp_steps = np.min(self.edp_steps[~np.isnan(self.edp_steps)])
                min_edp_indices = np.where(self.edp_steps == min_edp_steps)[0]
                self.edp_steps[min_edp_indices] += 0.0001
                self.project(
                    spb_target=self.spb_target,
                    edp_steps=self.edp_steps
                )
                self._save_edp_period()

        # If last EDP period has deficit < 3, we do not impose additional adjustment
        if self.ob[self.adjustment_start - 1 + self.edp_period] >= self.edp_target:
            self.edp_steps[self.edp_sb_index:] = np.nan
            self._save_edp_period()

        # If no spb_target was specified, calculate to ensure deficit < 3 until adjustment end
        if spb_target is None:
            print('No SPB target specified, calculating to ensure deficit < 3')
            while np.any(self.ob[self.edp_end + 1:self.adjustment_end + 1] <= self.edp_target):
                self.spb_target += 0.001
                self.project(spb_target=self.spb_target, edp_steps=self.edp_steps)

    def find_spb_deterministic(self, criterion, bounds=(-10, 10), steps=[0.01, 0.0001]):
        """
        Find the primary balance that ensures complience with deterministic criteria
        """
        # Check if input parameter correctly specified
        assert criterion in [
            None,
            'main_adjustment',
            'lower_spb',
            'financial_stress',
            'adverse_r_g',
            'deficit_reduction',
            'debt_safeguard',
        ], 'Unknown deterministic criterion'

        # Set scenario parameter
        if criterion in [None, 'main_adjustment', 'debt_safeguard']:
            self.scenario = 'main_adjustment'
        else:
            self.scenario = criterion

        # Precalculate EDP for debt safeguard if not specified and call optimizer
        if not hasattr(self, 'edp_steps'):
            if criterion == 'debt_safeguard':
                print('Precalculating EDP steps for debt safeguard')
                self.find_edp()
            else:
                self.edp_steps = None

        # Run deterministic optimization
        return self._deterministic_optimization(criterion=criterion, bounds=bounds, steps=steps)

    def _deterministic_optimization(self, criterion, bounds, steps):
        """
        Main loop of optimizer 
        """
        # If debt safeguard and EDP lasts until penultimate adjustment year, debt safeguard satisfied by default
        if (criterion == 'debt_safeguard'
                and self.edp_period >= self.adjustment_period - 1):
            self.spb_target = self.spb_bca[self.edp_end]
            self.project(
                spb_target=self.spb_target,
                edp_steps=self.edp_steps
            )
            return self.spb_target

        # Initialize spb_target to the lower bound
        spb_target = bounds[0]

        # Optimization loop
        while spb_target <= bounds[1]:
            try:
                # Project the model with the current spb_target
                self._get_spb_steps(criterion=criterion, spb_target=spb_target,)
                self.project(
                    edp_steps=self.edp_steps,
                    spb_steps=self.spb_steps,
                    scenario=self.scenario
                )

                # If condition is met, enter nested loop and decrease spb_target in small steps
                if self._deterministic_condition(criterion=criterion):
                    while (self._deterministic_condition(criterion=criterion)
                           and spb_target >= bounds[0]):
                        current_spb_target = spb_target
                        spb_target -= steps[1]
                        self._get_spb_steps(criterion=criterion, spb_target=spb_target)
                        self.project(
                            edp_steps=self.edp_steps,
                            spb_steps=self.spb_steps,
                            scenario=self.scenario
                        )
                    break

                # If condition is not met, increase spb_target in large steps
                current_spb_target = spb_target
                spb_target += steps[0]

            except BaseException:
                raise  # Exception(f'No solution found for {criterion}')

        # If spb_target exceeds upper bound, raise exception
        if spb_target > bounds[1] - steps[1]:
            raise  # Exception(f'No solution found for {criterion}')

        # Return last valid spb_target as optimal spb and project with target
        self.spb_target = current_spb_target
        spb_target -= steps[1]
        self._get_spb_steps(criterion=criterion, spb_target=spb_target)

        return self.spb_bca[self.adjustment_end]

    def _get_spb_steps(self, criterion, spb_target):
        """
        Get adjustment steps 
        """
        # If debt safeguard, apply adjustment to period after EDP
        if criterion == 'debt_safeguard':
            num_steps = self.adjustment_period - self.edp_period
            step_size = (spb_target - self.spb_bca[self.edp_end]) / num_steps
            non_edp_steps = np.full(num_steps, step_size)
            edp_steps_nonan = self.edp_steps[~np.isnan(self.edp_steps)]
            self.spb_steps = np.concatenate([edp_steps_nonan, non_edp_steps])

        # If adjustment steps are predifined, use them
        if hasattr(self, 'predefined_spb_steps'):
            num_predefined_steps = len(self.predefined_spb_steps)
            self.spb_steps[:num_predefined_steps] = np.copy(self.predefined_spb_steps)
            num_steps = self.adjustment_period - num_predefined_steps
            step_size = (spb_target - self.spb_bca[self.adjustment_start + num_predefined_steps - 1]) / num_steps
            self.spb_steps[num_predefined_steps:] = np.full(num_steps, step_size)

        # Otherwise apply adjustment to all periods
        else:
            num_steps = self.adjustment_period
            step_size = (spb_target - self.spb_bca[self.adjustment_start - 1]) / num_steps
            self.spb_steps = np.full(num_steps, step_size)

    def _deterministic_condition(self, criterion):
        """
        Defines deterministic criteria and checks if they are met.
        """
        if (criterion == 'main_adjustment'
            or criterion == 'lower_spb'
            or criterion == 'financial_stress'
            or criterion == 'adverse_r_g'):
            return self._debt_decline_criterion()
        elif criterion == 'deficit_reduction':
            return self._deficit_reduction_criterion()
        elif criterion == 'debt_safeguard':
            return self._debt_safeguard_criterion()
        else:
            return False

    def _debt_decline_criterion(self):
        """
        Checks the debt decline criterion from adjustment end to 10 years after adjustment end.
        """
        return (np.all(np.diff(self.d[self.adjustment_end:self.adjustment_end + 11]) < 0)
                or self.d[self.adjustment_end + 10] <= 60)

    def _deficit_reduction_criterion(self):
        """
        Checks the deficit reduction criterion for <3% deficit for 10 years after adjustment end.
        """
        return np.all(self.ob[self.adjustment_end:self.adjustment_end + 11] >= -3)

    def _debt_safeguard_criterion(self):
        """
        Checks the debt safeguard criterion.
        """
        debt_safeguard_decline = 1 if self.d[self.adjustment_start - 1] > 90 else 0.5

        if hasattr(self, 'predefined_spb_steps'):
            debt_safeguard_start = max(self.adjustment_start + len(self.predefined_spb_steps) - 1, self.edp_end + 1)
            
        else:
            debt_safeguard_start = self.edp_end + 1

        return (self.d[debt_safeguard_start] - self.d[self.adjustment_end]
                >= debt_safeguard_decline * (self.adjustment_end - debt_safeguard_start))

    def find_spb_deficit_resilience(self):
        """
        Apply the deficit resilience targets that sets min. annual spb adjustment if structural deficit exceeds 1.5%.
        """
        # Initialize deficit_resilience_steps
        self.deficit_resilience_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)

        # Define structural deficit target
        self.deficit_resilience_target = np.full(self.adjustment_period, -1.5, dtype=float)

        # Define deficit resilience step size
        if self.adjustment_period <= 4:
            self.deficit_resilience_step = 0.4
        else:
            self.deficit_resilience_step = 0.25     

        # Project baseline
        self.project(
            spb_target=self.spb_target,
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps
        )

        self.deficit_resilience_start = self.adjustment_start

        # Run deficit resilience loop
        self._deficit_resilience_loop_adjustment()

        return self.spb_bca[self.adjustment_end]

    def _deficit_resilience_loop_adjustment(self):
        """
        Loop for adjustment period violations of deficit resilience
        """
        for t in range(self.deficit_resilience_start, self.adjustment_end + 1):
            if ((self.d[t] > 60 or self.ob[t] < -3)
                and self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start] 
                and self.spb_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8):  # 1e-8 tolerance for floating point errors
                self.deficit_resilience_steps[t - self.adjustment_start] = self.spb_steps[t - self.adjustment_start]
                while (self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start]
                       and self.deficit_resilience_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8):  # 1e-8 tolerance for floating point errors
                    self.deficit_resilience_steps[t - self.adjustment_start] += 0.001
                    self.project(
                        spb_target=self.spb_target,
                        edp_steps=self.edp_steps,
                        deficit_resilience_steps=self.deficit_resilience_steps
                    )
    
    # ========================================================================================= #
    #                                   AUXILIARY METHODS                                       #
    # ========================================================================================= #

    def df(self, *vars, all=False):
        """
        Return a dataframe with the specified variables as columns and years as rows.
        Takes a variable name (string) or a list of variable names as input.
        Alternatively takes a dictionary as input, where keys are variables (string) and values are variable names.
        """
        # Get all attributes of the class that are of type np.ndarray, excluding private and built-in attributes
        all_vars = [attr for attr in dir(self) 
                    if not attr.startswith("_")
                    and isinstance(getattr(self, attr), np.ndarray)
                    and len(getattr(self, attr)) <= self.projection_period]

        # if no variables specified, return default variables
        if not vars and not all:
            vars = ['d', 'ob', 'sb', 'spb_bca', 'spb_bca_adjustment']

        # if all option True specified, return all variables
        elif not vars and all:
            vars = all_vars

        # If given dictionary as input, convert to list of variables and variable names
        if isinstance(vars[0], dict):
            var_dict = vars[0]
            var_names = list(var_dict.values())
            vars = list(var_dict.keys())

        # If given list as input, convert to list of variables and variable names
        elif isinstance(vars[0], list):
            vars = vars[0]
            var_names = None
        else:
            var_names = None

        var_values = []
        for var in vars:
            value = getattr(self, var) if isinstance(var, str) else var
            if len(value) < self.projection_period:
                value = np.append(value, [np.nan] * (self.projection_period - len(value)))
            var_values.append(value)

        df = pd.DataFrame(
            {vars[i]: var for i, var in enumerate(var_values)},
            index=range(self.start_year, self.end_year + 1)
        )

        if var_names:
            df.columns = var_names
        df.reset_index(names='y', inplace=True)
        df.reset_index(names='t', inplace=True)
        df.set_index(['t', 'y'], inplace=True)

        return df