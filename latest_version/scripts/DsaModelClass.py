#=========================================================================================#
#           European Commission Debt Sustainability Analysis - Base Class                 #
#=========================================================================================#
#
# The EcDsaModel class serves as the base framework for simulating economic and debt sustainability
# scenarios following the structure of the European Commission's 2023 Debt Sustainability Monitor. 
# This class provides the core functionalities to project key economic variables and analyse debt 
# dynamics under different scenario assumptions. 
#
# The class encompasses three primary parts:
# 1. **Data Methods:** These methods clean and combine input data.
# 2. **Projection Methods:** These methods handle the projection of economic variables such as GDP
#    growth, primary balance, interest rates, and debt dynamics, based on different scenarios.
# 3. **Optimization and Auxiliary Methods:** These methods include functions to optimize the primary
#    balance to meet specific criteria, check deterministic conditions, and the creation of DataFrames.
# 
# In addition, the Stochastic Model Subclass, a specialized subclass building upon this base class,
# provides additional features for stochastic projection of economic variables under uncertainty. The
# stochastic model subclass is defined in the file EcStochasticModelClass.py.
#
# For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org
#
# Author: Lennard Welslau
# Updated: 2023-12-22
#
#=========================================================================================#

# Import libraries and modules
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')

class DsaModel:
    #-----------------------------------------------------------------------------#
    #----------------------------- INIITIALIZE MODEL -----------------------------# 
    #-----------------------------------------------------------------------------#
    def __init__(
            self, 
            country, # ISO code of country
            start_year=2022, # start year of projection, first year is baseline value
            end_year=2053, # end year of projection
            adjustment_period=4, # number of years for linear spb_bca adjustment
            adjustment_start_year=2025, # start year of linear spb_bca adjustment
            ageing_cost_period=10, # number of years for ageing cost adjustment after adjustment period
            inv_shock=False,
            inv_size=0.5,
            inv_period=None,
            inv_exception=False,
            growth_policy=False,
            growth_policy_effect=0,
            growth_policy_cost=0,
            growth_policy_period=1,
            ):

        ## Initialize model parameters
        self.country = country
        self.start_year = start_year
        self.end_year = end_year
        self.T = self.end_year - start_year + 1
        self.adjustment_period = adjustment_period
        self.adjustment_start = adjustment_start_year - start_year
        self.adjustment_end_year = adjustment_start_year + adjustment_period - 1
        self.adjustment_end = adjustment_start_year + adjustment_period - start_year - 1
        self.ageing_cost_period = ageing_cost_period
        self.inv_shock = inv_shock
        self.inv_size = inv_size
        self.inv_space = inv_size
        if inv_period is None:
            self.inv_period = self.adjustment_period - 1
        else:
            self.inv_period = np.min([inv_period, self.adjustment_period])
        self.inv_exception = inv_exception
        self.growth_policy = growth_policy
        self.growth_policy_effect = growth_policy_effect
        self.growth_policy_cost = growth_policy_cost
        self.growth_policy_period = growth_policy_period
        self.growth_policy_cost_inflated = np.full(self.T, 0, dtype=np.float64) # cost of growth policy
        self.growth_policy_cost_ratio = np.full(self.T, 0, dtype=np.float64) # cost of growth policy as share of GDP

        ## Initialize model variables
        # GDP, growth, inflation
        self.rg_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline growth rate
        self.ng_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline nominal growth rate
        self.ngdp_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline nominal GDP
        self.rgdp_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline real GDP
        self.output_gap_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline output gap
        self.rg = np.full(self.T, np.nan, dtype=np.float64) # real growth rate adjusted for fiscal_multiplier
        self.ng = np.full(self.T, np.nan, dtype=np.float64) # nominal growth rate
        self.ngdp = np.full(self.T, np.nan, dtype=np.float64) # nominal GDP adjusted for fiscal_multiplier
        self.rgdp = np.full(self.T, np.nan, dtype=np.float64) # real GDP adjusted for fiscal_multiplier
        self.rgdp_pot = np.full(self.T, np.nan, dtype=np.float64) # potential GDP
        self.output_gap = np.full(self.T, np.nan, dtype=np.float64) # output gap
        self.rg_pot = np.full(self.T, np.nan, dtype=np.float64) # potential growth rate
        self.pi = np.full(self.T, np.nan, dtype=np.float64) # inflation rate
        self.fm = 0.75 # fiscal multiplier
        self.fm_effect = np.full(self.T, 0, dtype=np.float64) # fiscal multiplier impulse

        # Primary balance and components
        self.ageing_cost = np.full(self.T, np.nan, dtype=np.float64) # ageing cost
        self.property_income = np.full(self.T, np.nan, dtype=np.float64) # property income
        self.property_income_component = np.full(self.T, np.nan, dtype=np.float64) # property income component of primary balance
        self.cyclical_component = np.full(self.T, 0, dtype=np.float64) # cyclical_component component of primary balance
        self.ageing_component = np.full(self.T, 0, dtype=np.float64) # ageing_component component of primary balance
        self.PB = np.full(self.T, np.nan, dtype=np.float64) # primary balance
        self.SPB = np.full(self.T, np.nan, dtype=np.float64) # structural primary balance
        self.pb_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline primary balance over GDP
        self.spb_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline primary balance over GDP
        self.spb_bca = np.full(self.T, np.nan, dtype=np.float64) # structural primary balance over GDP before cost of ageing
        self.spb = np.full(self.T, np.nan, dtype=np.float64) # structural primary balance over GDP
        self.spb_bca_adjustment = np.full(self.T, np.nan, dtype=np.float64) # change in structural primary balance 
        self.pb_cyclical_adj = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP adjusted for cyclical_component component
        self.pb_cyclical_ageing_adj = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP adjusted for cyclical and ageing_component
        self.pb = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP

        # Implicit interest rate projection
        self.share_lt_maturing = np.full(self.T, np.nan, dtype=np.float64) # share of long-term debt maturing in the current year
        self.interest_st = np.full(self.T, np.nan, dtype=np.float64) # interest payment on short-term debt
        self.interest_lt = np.full(self.T, np.nan, dtype=np.float64) # interest payment on long-term debt
        self.interest = np.full(self.T, np.nan, dtype=np.float64) # interest payment on total debt
        self.interest_ratio = np.full(self.T, np.nan, dtype=np.float64) # interest payment over GDP
        self.i_st = np.full(self.T, np.nan, dtype=np.float64) # market interest rate on short-term debt
        self.i_lt = np.full(self.T, np.nan, dtype=np.float64) # market interest rate on long-term debt
        self.amortization_st = np.full(self.T, np.nan, dtype=np.float64) # amortization of short-term debt
        self.amortization_lt = np.full(self.T, np.nan, dtype=np.float64) # amortization of long-term debt
        self.amortization_lt_inst = np.full(self.T, 0, dtype=np.float64) # amortization of inst debt
        self.amortization = np.full(self.T, np.nan, dtype=np.float64) # amortization of total debt
        self.D_lt_inst = np.full(self.T, 0, dtype=np.float64) # inst debt
        self.D_st = np.full(self.T, 0, dtype=np.float64) # total short-term debt
        self.D_ltn = np.full(self.T, 0, dtype=np.float64) # new long-term debt
        self.D_lt = np.full(self.T, 0, dtype=np.float64) # total long-term debt
        self.D = np.full(self.T, 0, dtype=np.float64) # total debt
        self.gfn = np.full(self.T, np.nan, dtype=np.float64) # gross financing needs
        self.SF = np.full(self.T, 0, dtype=np.float64) # stock-flow adjustment

        # debt ratio projection
        self.exr_eur = np.full(self.T, np.nan, dtype=np.float64) # euro exchange rate
        self.exr_usd = np.full(self.T, np.nan, dtype=np.float64) # usd exchange rate
        self.sf = np.full(self.T, 0, dtype=np.float64) # stock-flow adjustment over GDP
        self.ob = np.full(self.T, np.nan, dtype=np.float64) # fiscal balance
        self.sb = np.full(self.T, np.nan, dtype=np.float64) # structural balance
        self.d = np.full(self.T, np.nan, dtype=np.float64) # debt to GDP ratio
        
        # Implicit interest rate
        self.iir_bl = np.full(self.T, np.nan, dtype=np.float64) # baseline implicit interest rate
        self.alpha = np.full(self.T, np.nan, dtype=np.float64) # share of short-term debt in total debt
        self.beta = np.full(self.T, np.nan, dtype=np.float64) # share of new long-term debt in total long-term debt
        self.iir = np.full(self.T, np.nan, dtype=np.float64) # impoict interest rate
        self.iir_lt = np.full(self.T, np.nan, dtype=np.float64) # implicit long-term interest rate

        # Auxiliary variables for stochastic simulations
        self.exr = np.full(self.T, np.nan, dtype=np.float64) # exchange rate           
        
        ## Import data
        self._import_data()

        ## Clean data
        self._clean_data()

        # If investment shock with an exception for EDP and SPB, calculate deficit elasticity
        # if self.inv_exception: self._calculate_deficit_elasticity()
    
    #-----------------------------------------------------------------------------------#
    #----------------------------- DATA METHODS (INTERNAL) -----------------------------#
    #-----------------------------------------------------------------------------------#  
    def _import_data(self):
        """
        Import data from Excel input data file.
        """
        self._import_inst_debt_data()
        self._import_ameco_data()
        self._import_output_gap_data()
        self._import_com_data()
        self._import_interest_rate_data()
        self._import_inflation_data()
        self._import_ecb_data()
        self._import_budget_elasticity_data()

    def _import_inst_debt_data(self):
        """
        Import institutional debt data from Excel input data file.
        """
        self.df_debt_inst = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='esm_data')
        if self.country in self.df_debt_inst['ISO'].unique():
            self.df_debt_inst = self.df_debt_inst.loc[self.df_debt_inst['ISO'] == self.country]
            self.D_lt_inst[0] = self.df_debt_inst['amortization'].sum()
            self.esm = True
        else:
            self.esm = False

    def _import_ameco_data(self):
        """
        Import ameco projection baseline data from Excel input data file.
        """
        self.df_ameco = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='ameco_data')
        self.pi_ea_2025 = self.df_ameco.loc[(self.df_ameco['year'] == 2025) & (self.df_ameco['ISO'] == 'EA20'), 'gdp_def_pch'].values[0]
        self.df_ameco = self.df_ameco.loc[self.df_ameco['ISO'] == self.country]
        self.ameco_end_y = self.df_ameco['year'].max()
        self.ameco_end_t = self.ameco_end_y - self.start_year

    def _import_output_gap_data(self):
        """
        Import output gap working group data from Excel input data file.
        """
        self.df_ogwg = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='output_gap_working_group')
        self.df_ogwg = self.df_ogwg.loc[self.df_ogwg['ISO'] == self.country]
        self.ogwg_projection_end = self.df_ogwg['year'].max()

    def _import_com_data(self):
        """
        Import commission projections data from Excel input data file.
        """
        df_commission = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='commission_data')
        self.df_ageing_cost = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['ageing_cost']
        self.df_real_growth = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['real_growth']
        self.df_property_income = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['property_income']

    def _import_interest_rate_data(self):
        """
        Import BBG interest rate baseline and expectations from Excel input data file.
        """
        df_gov_rates = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='bbg_data')
        self.df_2022_benchmark_rates = df_gov_rates.loc[df_gov_rates['year'] == 2022, ['ISO', '3M', '10Y']]
        self.df_2023_benchmark_rates = df_gov_rates.loc[df_gov_rates['year'] == 2023, ['ISO', '3M', '10Y']]
        self.df_2024_benchmark_rates = df_gov_rates.loc[df_gov_rates['year'] == 2024, ['ISO', '3M', '10Y']]
        self.df_fwd_rates = df_gov_rates.loc[df_gov_rates['year'] == 2024, ['ISO', '3M10Y', '10Y10Y']]
        
        if self.country in ['BGR', 'CZE', 'DNK', 'HUN', 'POL', 'ROU', 'SWE']:
            try:
                self.fwd_rate_st = self.df_fwd_rates.loc[(df_gov_rates['ISO'] == self.country), '3M10Y'].values[0]
            except:
                self.fwd_rate_st = self.df_fwd_rates.loc[(self.df_fwd_rates['ISO'] == 'EUZ'), '3M10Y'].values[0] # If no country specific rate available, use EA rate
        else:
            self.fwd_rate_st = self.df_fwd_rates.loc[(self.df_fwd_rates['ISO'] == 'EUZ'), '3M10Y'].values[0] # The Commission uses a common Eurozone short-term rate for EA countries
        
        self.fwd_rate_lt = self.df_fwd_rates.loc[(self.df_fwd_rates['ISO'] == self.country), '10Y10Y'].values[0]
        self.benchmark_rate_st_2022 = self.df_2022_benchmark_rates.loc[(self.df_2022_benchmark_rates['ISO'] == self.country), '3M'].values[0] 
        self.benchmark_rate_st_2023 = self.df_2023_benchmark_rates.loc[(self.df_2023_benchmark_rates['ISO'] == self.country), '3M'].values[0]         
        self.benchmark_rate_st_2024 = self.df_2024_benchmark_rates.loc[(self.df_2024_benchmark_rates['ISO'] == self.country), '3M'].values[0]
        self.benchmark_rate_lt_2022 = self.df_2022_benchmark_rates.loc[(self.df_2022_benchmark_rates['ISO'] == self.country), '10Y'].values[0]
        self.benchmark_rate_lt_2023 = self.df_2023_benchmark_rates.loc[(self.df_2023_benchmark_rates['ISO'] == self.country), '10Y'].values[0]
        self.benchmark_rate_lt_2024 = self.df_2024_benchmark_rates.loc[(self.df_2024_benchmark_rates['ISO'] == self.country), '10Y'].values[0]

    def _import_inflation_data(self):
        """
        Import inflation expectations from Excel input data file.
        """
        self.df_infl_fwd = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name= 'inflation_fwd')
        self.infl_fwd = self.df_infl_fwd.loc[(self.df_infl_fwd['maturity'] == 10), 'infl_expectation'].values[0]

    def _import_ecb_data(self):
        """
        Import ECB data on debt stock from Excel input data file.
        """
        self.df_ecb = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='ecb_data')
        self.share_st = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'share_st_org'].values[0]
        self.D[0] = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'debt_total'].values[0]
        self.D_st[0] = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'debt_st'].fillna(0).values[0]
        self.D_lt[0] = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'debt_total'].values[0] - self.D_st[0]
        self.share_lt_maturing_t0 = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'share_lt_maturing'].values[0]
        self.share_lt_maturing_t10 = self.df_ecb.loc[(self.df_ecb['ISO'] == self.country), 'share_lt_maturing_6y_avg'].values[0]
        self.m_res_lt = min(round((1 / self.share_lt_maturing_t10)), 30)
        self.share_domestic = self.df_ecb.loc[self.df_ecb['ISO'] == self.country, 'share_domestic'].values[0]
        self.share_foreign = self.df_ecb.loc[self.df_ecb['ISO'] == self.country, 'share_foreign_non_euro'].values[0]
        self.share_eur_stochastic = self.df_ecb.loc[self.df_ecb['ISO'] == self.country, 'share_eur_stochastic'].values[0]
        if self.country in ['BGR', 'CZE', 'DNK', 'HUN', 'POL', 'ROU', 'SWE']:
            self.share_eur = self.share_eur = self.df_ecb.loc[self.df_ecb['ISO'] == self.country, 'share_eur'].values[0]
        else:
            self.share_eur = 0 # Euro countries euro share is accounted by domestic share

    def _import_budget_elasticity_data(self):
        """
        Import budget semi elasticities from Excel input data file.
        """
        self.df_elasticity = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='budget_elasticity')
        self.pb_elasticity = self.df_elasticity.loc[(self.df_elasticity['ISO'] == self.country), 'budget_elasticity'].values[0]
        
    def _clean_data(self):
        """
        Clean data, extrapolate missing values, and save as vectors.
        """
        self._clean_rgdp_pot()
        self._clean_rgdp()
        if self.growth_policy: self._apply_growth_policy_effect()
        self._clean_output_gap()
        self._clean_pi()
        self._clean_ngdp()
        self._clean_spb()
        self._clean_pb()
        self._clean_iir()
        self._clean_sf()
        self._clean_exchange_rate()
        self._clean_ageing_cost()
        self._clean_property_income()
        self._clean_debt_redemption()

    def _clean_rgdp_pot(self):
        """
        Clean baseline real potential growth.
        """

        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            
            # Values up to t+2 are from AMECO
            if y <= self.ameco_end_y:
                self.rgdp_pot[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'gdp_pot'].values[0]
                if t > 0: 
                    self.rg_pot[t] = (self.rgdp_pot[t] - self.rgdp_pot[t-1]) / self.rgdp_pot[t-1] * 100

            # Values from t+3 to t+5 are from OGWG
            elif y <= self.ogwg_projection_end:
                self.rgdp_pot[t] = self.df_ogwg.loc[self.df_ogwg['year'] == y, 'gdp_pot'].values[0]
                self.rg_pot[t] = (self.rgdp_pot[t] - self.rgdp_pot[t-1]) / self.rgdp_pot[t-1] * 100
            
            # From t+4 projectiosn are based on long run COM estimates
            elif y > self.ogwg_projection_end:
                self.rg_pot[t] = self.df_real_growth.loc[y]
                self.rgdp_pot[t] = self.rgdp_pot[t-1] * (1 + self.rg_pot[t] / 100)

    def _clean_rgdp(self):
        """
        Clean baseline real growth. 
        """        
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            
            # Values up to t+2 are from AMECO
            if y <= self.ameco_end_y:
                self.rgdp_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'rgdp'].values[0]
                if t > 0: 
                    self.rg_bl[t] = (self.rgdp_bl[t] - self.rgdp_bl[t-1]) / self.rgdp_bl[t-1] * 100

            # Values from t+3 to t+5 are from OGWG
            elif y <= self.ogwg_projection_end:
                self.rgdp_bl[t] = self.df_ogwg.loc[self.df_ogwg['year'] == y, 'gdp_real'].values[0]
                self.rg_bl[t] = (self.rgdp_bl[t] - self.rgdp_bl[t-1]) / self.rgdp_bl[t-1] * 100

            # From t+4 projectiosn are based on long run COM estimates
            elif y > self.ogwg_projection_end:
                self.rg_bl[t] = self.df_real_growth.loc[y]
                self.rgdp_bl[t] = self.rgdp_bl[t-1] * (1 + self.rg_bl[t] / 100)
            
        # Set initial values to baseline
        self.rg = np.copy(self.rg_bl)
        self.rgdp = np.copy(self.rgdp_bl)

    def _apply_growth_policy_effect(self):
        """
        Apply growth enhancing policy to potential and real GDP.
        """
        for t in range(self.adjustment_start, self.T):
            
            # Calcualte increase of real potential gdp during policy period
            if t in range(self.adjustment_start, self.adjustment_start + self.growth_policy_period):
                rgdp_pot_annual_effect = (self.rgdp_pot[self.adjustment_start + self.growth_policy_period - 1] 
                                          * (self.growth_policy_effect / 100) 
                                          * (t - self.adjustment_start + 1) / self.growth_policy_period)

                # Apply increase to rgdp
                self.rgdp_pot[t] += rgdp_pot_annual_effect 
                self.rgdp_bl[t] += rgdp_pot_annual_effect
                self.rgdp[t] += rgdp_pot_annual_effect

                # Recalculate implied growth rates
                self.rg_pot[t] = (self.rgdp_pot[t] - self.rgdp_pot[t-1]) / self.rgdp_pot[t-1] * 100
                self.rg_bl[t] = (self.rgdp_bl[t] - self.rgdp_bl[t-1]) / self.rgdp_bl[t-1] * 100
                self.rg[t] = (self.rgdp[t] - self.rgdp[t-1]) / self.rgdp[t-1] * 100
            
            # Use default growth rates to project GDP after
            else:
                self.rgdp_pot[t] = self.rgdp_pot[t-1] * (1 + self.rg_pot[t] / 100) 
                self.rgdp_bl[t] = self.rgdp_bl[t-1] * (1 + self.rg_bl[t] / 100) 
                self.rgdp[t] = self.rgdp_bl[t-1] * (1 + self.rg[t] / 100) 
        
    def _clean_output_gap(self):
        """
        Clean GDP gap. 
        Uses T+5 projection. Rest calculated during projection.
        """        
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            self.output_gap_bl[t] = (self.rgdp_bl[t] / self.rgdp_pot[t] - 1) * 100
        
        # Set initial values to baseline        
        self.output_gap = np.copy(self.output_gap_bl)

    def _clean_pi(self):
        """
        Clean inflation rate data. 
        Uses ameco up to 2024, interpolates to swap implied t+10 value, then to country-specific t+30 target.
        """
        for t, y in enumerate(range(self.start_year, self.ameco_end_y+1)):
            self.pi[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'gdp_def_pch'].values[0]
        
        # Set t+10 value from inflation expectations
        self.pi[2033-self.start_year] = self.infl_fwd

        # Set t+30 value
        self.df_pi = self.df_ameco.loc[self.df_ameco['year'] >= self.start_year, ['year', 'gdp_def_pch']] 
        self.df_pi = self.df_pi.set_index('year').reindex(range(self.start_year, self.start_year+self.T)) 
        if self.country in ['POL', 'ROU']:
            self.df_pi.loc[2033, 'gdp_def_pch'] = self.infl_fwd + (self.df_pi.loc[2025].values[0] - self.pi_ea_2025) / 2
            self.df_pi.loc[2053, 'gdp_def_pch'] = 2.5
            self.df_pi.loc[2033] = self.infl_fwd + (self.df_pi.loc[2025].values[0] - self.pi_ea_2025) / 2
            self.df_pi.loc[2053] = 2.5
        elif self.country in ['HUN']:
            self.df_pi.loc[2033, 'gdp_def_pch'] = self.infl_fwd
            self.df_pi.loc[2053, 'gdp_def_pch'] = 3
            self.df_pi.loc[2033] = self.infl_fwd
            self.df_pi.loc[2053] = 3
        else:
            self.df_pi.loc[2033, 'gdp_def_pch'] = self.infl_fwd
            self.df_pi.loc[2053, 'gdp_def_pch'] = 2
            self.df_pi.loc[2033] = self.infl_fwd
            self.df_pi.loc[2053] = 2
        
        # Interpolate missing values and save as vector
        self.df_pi = self.df_pi.interpolate(method='linear')
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            self.pi[t] = self.df_pi.loc[y, 'gdp_def_pch']
            self.pi[t] = self.df_pi.loc[y].values[0]

    def _clean_ngdp(self):
        """
        Clean baseline nominal growth. 
        Uses T+2 from Ameco for totals, T+5 for growth rates, then real growth and inflation rate.
        """  
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if y <= self.ameco_end_y:
                self.ngdp_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'ngdp'].values[0]
                self.ng_bl[t] = (self.ngdp_bl[t] / self.df_ameco.loc[self.df_ameco['year'] == y-1, 'ngdp'].values[0] - 1) * 100
            else:
                self.ng_bl[t] = (1 + self.rg_bl[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100
                self.ngdp_bl[t] = self.ngdp_bl[t-1] * (1 + self.ng_bl[t] / 100)
            
        # Set initial values to baseline
        self.ng = np.copy(self.ng_bl)
        self.ngdp = np.copy(self.ngdp_bl)
        self.d[0] = self.D[0] / self.ngdp[0] * 100

    def _clean_spb(self):
        """
        Clean structural primary balance.
        Uses ameco projection, then constant.
        """          
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if t < self.adjustment_start:
                self.spb_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'spb']
            else:
                self.spb_bl[t] = self.spb_bl[t-1]
            self.spb_bca[t] = self.spb_bl[t]
            self.spb[t] = self.spb_bl[t]
    
    def _clean_pb(self):
        """
        Clean primary balance.
        Uses ameco projection, ten populated during projection.
        """
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if y <= self.ameco_end_y:
                self.pb_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'pb']
            if y > self.ameco_end_y:
                break
        
        # Set initial values to baseline
        self.pb = np.copy(self.pb_bl)

    def _clean_iir(self):
        """
        Clean implicit interest rate.
        Uses ameco projection, rest only populated during projection.
        """          
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if y <= self.ameco_end_y:
                self.iir_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'iir'].values[0]
                self.iir[t] = self.iir_bl[t]
            if y > self.ameco_end_y:
                break
        # Iniital lt baseline 
        self.iir_lt[0] = self.iir[0] * (1 - self.share_st)

    def _clean_sf(self):
        """
        CLean stock flow adjustment.
        """          
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if y <= self.ameco_end_y:
                self.SF[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'sf'].values[0]
            if y > self.ameco_end_y:
                break

    def _clean_exchange_rate(self):
        """
        CLean exchange rate data for non-euro countries.
        """    
        for t, y in enumerate(range(self.start_year, self.end_year+1)):

            if y <= self.ameco_end_y:
                # USD exchange rate for all countries
                self.exr_usd[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'exr_usd'].values[0]
                # Euro exchange rate for non-euro countries
                if self.country in ['BGR', 'CZE', 'DNK', 'HUN', 'POL', 'ROU', 'SWE']:
                    if y <= self.ameco_end_y:
                        self.exr_eur[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'exr_eur'].values[0]
                    if y > self.ameco_end_y:
                        self.exr_eur[t:] = self.exr_eur[t-1]
                else:
                    self.exr_eur[t] = 1
            if y > self.ameco_end_y:
                self.exr_usd[t:] = self.exr_usd[t-1]
                self.exr_eur[t:] = self.exr_eur[t-1]

    def _clean_ageing_cost(self):
        """
        Clean ageing cost data.
        """  
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            self.ageing_cost[t] = self.df_ageing_cost.loc[y]

    def _clean_property_income(self):
        """
        Clean property income data.
        """  
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            self.property_income[t] = self.df_property_income.loc[y]

    def _clean_debt_redemption(self):
        """
        Clean debt redemption data for institutional debt.
        """
        # Add ESM ESFS loans and interest payments if available
        if self.esm:

            # Loop over years and calculate amortization and interest payments of inst debt
            for t, y in enumerate(range(self.start_year+1, self.end_year+1)):
                t += 1 # start in t = 1
                self.amortization_lt_inst[t] = self.df_debt_inst.loc[self.df_debt_inst['year'] == y, 'amortization'].values[0]
                self.D_lt_inst[t] = self.D_lt_inst[t-1] - self.amortization_lt_inst[t]
            
        # Set t and t + 10 value of maturing lt debt share
        self.share_lt_maturing[2023-self.start_year] = self.share_lt_maturing_t0 
        self.share_lt_maturing[(2033-self.start_year):] = self.share_lt_maturing_t10

        # Interpolate missing values
        x = np.arange(len(self.share_lt_maturing))
        mask = np.isnan(self.share_lt_maturing)
        self.share_lt_maturing[mask] = np.interp(x[mask], x[~mask], self.share_lt_maturing[~mask])

    #------------------------------------------------------------------------------#
    #----------------------------- PROJECTION METHODS -----------------------------# 
    #------------------------------------------------------------------------------#
    def project(self,
                spb_target=None,
                adjustment_steps=None, # list of annual adjustment steps during adjustment
                edp_steps=None, # list of annual adjustment steps during EDP             
                deficit_resilience_steps=None, # list of years during adjustment where minimum step size is enforced
                post_adjustment_steps=None, # list of years after adjustment where minimum step size is enforced
                scenario='main_adjustment', # scenario parameter, needed for DSA criteria
                inv_shock=None, # investment shock
                ): 
        """
        Project debt dynamics
        """
        # Clear starting values from memory
        self.D_ltn = np.full(self.T, 0, dtype=np.float64) 
        self._clean_iir()

        # Set spb_target 
        if (spb_target is None
            and adjustment_steps is None):
            self.spb_target = self.spb_bca[self.adjustment_start-1]
        elif (spb_target is None
            and adjustment_steps is not None):
            self.spb_target = self.spb_bca[self.adjustment_start-1] + adjustment_steps.sum()
        else:  
            self.spb_target = spb_target
        
        # Set adjustment steps
        if (adjustment_steps is None
            and spb_target is not None):
            self.adjustment_steps = np.full((self.adjustment_period,), (self.spb_target - self.spb_bca[self.adjustment_start-1]) / self.adjustment_period, dtype=np.float64)      
        elif (adjustment_steps is None
            and spb_target is None):
            self.adjustment_steps = np.full((self.adjustment_period,), 0, dtype=np.float64)
        else:
            self.adjustment_steps = adjustment_steps

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
        if post_adjustment_steps is None:
            self.post_adjustment_steps = np.full((self.T - self.adjustment_end-1,), 0, dtype=np.float64)
        else:
            self.post_adjustment_steps = post_adjustment_steps    

        # Set scenario parameter
        self.scenario = scenario

        # Set investment shock parameter if defined
        if inv_shock is not None: 
            self.inv_shock = inv_shock

        ## Project debt dynamics
        self._project_net_expenditure_path()
        self._project_market_rate()
        self._project_gdp()
        self._project_spb()
        self._project_pb()
        self._project_d()

    def _project_net_expenditure_path(self):
        """
        Project structural primary balance, excluding ageing cost
        """

        # If investment shock, adjust first and last adjustment step
        if self.inv_shock: self._apply_inv_shock()

        # Adjust path for EDP and deficit resilience steps
        self._adjust_for_edp()
        self._adjust_for_deficit_resilience()
        self._apply_adjustment_steps()

        # If lower_spb scenario, adjust path
        if self.scenario == 'lower_spb': self._apply_lower_spb()

        # If growth policy, adjust path
        if self.growth_policy: self._calculate_growth_policy_cost()

    def _adjust_for_edp(self):
        """
        Adjust linear path for minimum EDP adjustment steps 
        """        
        # Save copy of baseline adjustment steps 
        self.adjustment_steps_baseline = np.copy(self.adjustment_steps)

        # Apply EDP steps to adjustment steps
        self.adjustment_steps[~np.isnan(self.edp_steps)] = np.where(
            self.edp_steps[~np.isnan(self.edp_steps)] > self.adjustment_steps[~np.isnan(self.edp_steps)], 
            self.edp_steps[~np.isnan(self.edp_steps)], 
            self.adjustment_steps[~np.isnan(self.edp_steps)]
            )
                
        # Identify periods that are after EDP and correct them for frontloading
        if not np.isnan(self.edp_steps).all(): 
            last_edp_index = np.where(~np.isnan(self.edp_steps))[0][-1] 
        else: 
            last_edp_index = 0
        post_edp_index = np.arange(last_edp_index + 1, len(self.adjustment_steps))
        self.diff_adjustment_baseline = np.sum(self.adjustment_steps_baseline - self.adjustment_steps)
        offset_edp = self.diff_adjustment_baseline / len(post_edp_index) if len(post_edp_index) > 0 else 0
        self.adjustment_steps[post_edp_index] += offset_edp

    def _adjust_for_deficit_resilience(self):
        """
        Adjust linear path for minimum deficit resilience adjustment steps 
        """

        # Save copy of edp adjusted steps 
        self.adjustment_steps_baseline = np.copy(self.adjustment_steps)

        # Apply deficit resilience safeguard steps to adjustment steps
        self.adjustment_steps[~np.isnan(self.deficit_resilience_steps)] = np.where(
            self.deficit_resilience_steps[~np.isnan(self.deficit_resilience_steps)] > self.adjustment_steps[~np.isnan(self.deficit_resilience_steps)], 
            self.deficit_resilience_steps[~np.isnan(self.deficit_resilience_steps)], 
            self.adjustment_steps[~np.isnan(self.deficit_resilience_steps)]
            )
        
        # Identify periods that are after EDP and deficit resilience and correct for frontloading
        if not (np.isnan(self.edp_steps).all() 
                and np.isnan(self.deficit_resilience_steps).all()): 
            last_edp_deficit_resilience_index = np.where(~np.isnan(self.edp_steps) | ~np.isnan(self.deficit_resilience_steps))[0][-1] 
        else: 
            last_edp_deficit_resilience_index = 0
        post_edp_deficit_resilience_index = np.arange(last_edp_deficit_resilience_index + 1, len(self.adjustment_steps))
        self.diff_adjustment_baseline = np.sum(self.adjustment_steps_baseline - self.adjustment_steps)
        self.offset_deficit_resilience = self.diff_adjustment_baseline / len(post_edp_deficit_resilience_index) if len(post_edp_deficit_resilience_index) > 0 else 0
        self.adjustment_steps[post_edp_deficit_resilience_index] += self.offset_deficit_resilience

    def _apply_adjustment_steps(self):
        """
        Project spb_bca
        """
        # Apply adjustment steps based on the current period
        for t in range(self.adjustment_start, self.T):
            if t in range(self.adjustment_start, self.adjustment_end + 1):
                self.spb_bca[t] = self.spb_bca[t - 1] + self.adjustment_steps[t - self.adjustment_start]
            else:
                self.spb_bca[t] = self.spb_bca[t - 1] + self.post_adjustment_steps[t - self.adjustment_end - 1]
            
        # Save adjustment step size
        self.spb_bca_adjustment[1:] = np.diff(self.spb_bca)

    def _apply_lower_spb(self):
        """
        Apply lower_spb scenario
        """
        # If 4-year adjustment period, spb_bca decreases by 0.5 for 2 years after adjustment period, if 7-year for 3 years
        lower_spb_adjustment_period = int(np.floor(self.adjustment_period/2))
        for t in range(self.adjustment_end + 1, self.T):
            if t <= self.adjustment_end + lower_spb_adjustment_period:
                self.spb_bca[t] -= 0.5 / lower_spb_adjustment_period * (t - self.adjustment_end)
            else:
                self.spb_bca[t] = self.spb_bca[t-1]
    
    def _calculate_growth_policy_cost(self):
        """
        Calculate cost of growth policy
        """
        
        for t in range(self.adjustment_start, self.T):
            
            # During growth policy period, cost is phased in
            if t < self.adjustment_start + self.growth_policy_period:
                self.growth_policy_cost_inflated[t] = self.growth_policy_cost * (t - self.adjustment_start + 1) / self.growth_policy_period
            
            # After growth policy period, cost stays constant
            else:
                self.growth_policy_cost_inflated[t] = self.growth_policy_cost
            
            # Inflate costs with nominal growth rate and calculate as share of NGDP
            for ng_value in self.ng[:t]:
                self.growth_policy_cost_inflated[t] *= 1 + ng_value / 100

            self.growth_policy_cost_ratio[t] = self.growth_policy_cost_inflated[t] / self.ngdp[t] * 100

    def _apply_inv_shock(self):
        """
        Apply inv_shock scenario that reduces spb_bca by 0.5% of GDP from first to penultimate adjustment period.
        Used for counterfactual analysis to check how much investment would be allowed under various specifications.
        """
        # Investment shock size faces lower bound of EDP and deficit resilience safeguard
        if (not np.isnan(self.edp_steps[0]) 
            or not np.isnan(self.deficit_resilience_steps[0])):
            self.inv_space = np.max([
                np.min([
                    self.adjustment_steps[0] - np.nan_to_num(self.edp_steps[0]),
                    self.adjustment_steps[0] - np.nan_to_num(self.deficit_resilience_steps[0])
                ]),
                0
            ])
        self.adjustment_steps[0] -= self.inv_size
        if self.inv_period < self.adjustment_period:
            self.adjustment_steps[self.inv_period] += self.inv_size

    def _project_market_rate(self):
        """
        Project market rate data, needs to be in projection method because of scenario parameter.
        Uses BBG forward rates up to T+10, then linearly interpolates to long-term values.
        """
        # Clean vectors in case of repeated projection with different scenarios
        self.i_st = np.full(self.T, np.nan) 
        self.i_lt = np.full(self.T, np.nan) 

        # Set t + 10 value as market fwd rate 
        self.i_st[2033-self.start_year] = self.fwd_rate_st
        self.i_lt[2033-self.start_year] = self.fwd_rate_lt

        # Set t + 30 values
        if self.country in ['POL', 'ROU']:
            i_lt_30 = 4.5
        elif self.country in ['HUN']:
            i_lt_30 = 5
        else:
            i_lt_30 = 4

        yield_curve_coef = 0.5
        self.i_st[2053-self.start_year:] = i_lt_30 * yield_curve_coef 
        self.i_lt[2053-self.start_year:] = i_lt_30 
        
        # Set short term 2022 value
        self.i_st[2022-self.start_year] = self.benchmark_rate_st_2022
        self.i_lt[2022-self.start_year] = self.benchmark_rate_lt_2022
       
        # Set 2023 value
        self.i_st[2023-self.start_year] = self.benchmark_rate_st_2023 
        self.i_lt[2023-self.start_year] = self.benchmark_rate_lt_2023 

        # Set 2024 value
        self.i_st[2024-self.start_year] = self.benchmark_rate_st_2024
        self.i_lt[2024-self.start_year] = self.benchmark_rate_lt_2024

        # Interpolate missing values
        x_st = np.arange(len(self.i_st))
        mask_st = np.isnan(self.i_st)
        self.i_st[mask_st] = np.interp(x_st[mask_st], x_st[~mask_st], self.i_st[~mask_st])

        x_lt = np.arange(len(self.i_lt))
        mask_lt = np.isnan(self.i_lt)
        self.i_lt[mask_lt] = np.interp(x_lt[mask_lt], x_lt[~mask_lt], self.i_lt[~mask_lt])
    
        if self.scenario == 'adverse_r_g': self._apply_adverse_r()

    def _apply_adverse_r(self):
        """ 
        Applies adverse interest rate conditions for adverse r-g scenario
        """
        self.i_st[self.adjustment_end+1:] += 0.5
        self.i_lt[self.adjustment_end+1:] += 0.5

    def _project_gdp(self):
        """
        Project nominal GDP.
        """  
        for t in range(1, self.T):
            self._calculate_rgdp(t)
            self._calculate_ngdp(t)
    
    def _calculate_rgdp(self, t):
        """
        Calcualtes real GDP and real growth
        """
        # Fiscal multiplier effect from change in SPB relative to baseline
        self.fm_effect[t] = self.fm * ((self.spb_bca[t] - self.spb_bca[t-1]) - (self.spb_bl[t] - self.spb_bl[t-1]))   
        
        # Effect on output gap
        self.output_gap[t] = self.output_gap_bl[t] - self.fm_effect[t] - 2/3 * self.fm_effect[t-1] - 1/3 * self.fm_effect[t-2]
        
        # Real growth and real GDP
        self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]
        self.rg[t] = (self.rgdp[t] - self.rgdp[t-1]) / self.rgdp[t-1] * 100

    def _calculate_ngdp(self, t):
        """
        Calcualtes nominal GDP and nominal growth
        """
        # Before adjustment period, nominal growth is baseline
        if t < self.adjustment_start:
            self.ng[t] = self.ng_bl[t]
            self.ngdp[t] = self.ngdp_bl[t]

        # After adjustment period, nominal growth based on real growth and inflation
        elif t >= self.adjustment_start:
            self.ng[t] = (1 + self.rg[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100 

            # Adjust nominal growth for adverse r-g scenario
            if self.scenario == 'adverse_r_g' and t > self.adjustment_end: self._apply_adverse_g(t)
                            
            # project nominal GDP
            self.ngdp[t] = self.ngdp[t-1] * (1 + self.ng[t] / 100) 

    def _apply_adverse_g(self, t):
        """
        Applies adverse growth conditions for adverse r-g scenario
        """
        self.ng[t] -= 0.5

    def _project_spb(self):
        """
        Project structural primary balance
        """
        for t in range(1, self.T):
            
            # Ageing cost adjustments are accounted for by spb adjustment during the adjustment period
            if t <= self.adjustment_end: 
                self.spb[t] = self.spb_bca[t]

            # After adjustment period ageing costs affect the SPB for duration of "ageing_cost_period"
            elif t > self.adjustment_end and t <= self.adjustment_end + self.ageing_cost_period: 
                self.ageing_component[t] = - (self.ageing_cost[t] - self.ageing_cost[self.adjustment_end]) 
                self.spb[t] = self.spb_bca[t] + self.ageing_component[t]
            
            # After ageing cost period, SPB is baseline
            elif t > self.adjustment_end + self.ageing_cost_period: 
                self.spb[t] = self.spb[t-1]

            # Total SPB for calcualtion of structural deficit
            self.SPB[t] = self.spb[t] / 100 * self.ngdp[t]
    
    def _project_pb(self):
        """
        Project primary balance adjusted as sum of SPB, cyclical component, and property income component
        """
        for t in range(1, self.T):
            
            # Equal to baseline until adjustment start
            if t < self.adjustment_start: self.pb[t] = self.pb_bl[t]

            # Calculate components
            self.output_gap[t] = (self.rgdp[t] / self.rgdp_pot[t] - 1) * 100
            self.cyclical_component[t] = self.pb_elasticity * self.output_gap[t]
            self.property_income_component[t] = self.property_income[t] - self.property_income[self.adjustment_start - 1]

            # Calculate primary balance ratio as sum of components and total primary balance
            self.pb[t] = self.spb[t] + self.cyclical_component[t] + self.property_income_component[t]
            self.PB[t] = self.pb[t] / 100 * self.ngdp[t]

    def _project_d(self):
        """
        Main loop for debt dynamics        
        """
        for t in range(1,self.T):
            
            # Apply financial stress scenario if specified
            if self.scenario == 'financial_stress' and t == self.adjustment_end+1: self._apply_financial_stress(t)

            # Calculate implicit interest rate, interestst, amortizations, gross financing needs, debt stock, overall balance, and debt ratio
            self._calculate_iir(t)
            self._calculate_interest(t)
            self._calculate_amortization(t)
            self._calculate_gfn(t)
            self._calculate_debt_stock(t)
            self._calculate_balance(t)
            self._calculate_debt_ratio(t)

    def _apply_financial_stress(self, t):
        """
        Adjust interest rates for financial stress scenario
        """
        # Adjust market rates for high debt countries financial stress scenario
        if self.d[self.adjustment_end] > 90:
            self.i_st[t] += (1 + (self.d[self.adjustment_end] - 90) * 0.06)
            self.i_lt[t] += (1 + (self.d[self.adjustment_end] - 90) * 0.06)

        # Adjust market rates for low debt countries financial stress scenario
        else:
            self.i_st[t] += 1
            self.i_lt[t] += 1

    def _calculate_iir(self, t):
        """
        Calculate implicit interest rate
        """
        # Calculate the shares of short term and long term debt in total debt
        self.alpha[t-1] = self.D_st[t-1] /  self.D[t-1]
        self.beta[t-1] = self.D_ltn[t-1] /  self.D_lt[t-1]

        # Use ameco implied interest in the short term and back out iir_lt
        if t <= self.ameco_end_t:
            self.iir_lt[t] = (self.iir[t] - self.alpha[t-1] * self.i_st[t]) / (1 - self.alpha[t-1])
            self.iir[t] = self.iir_bl[t]

        # Use DSM 2023 Annex A3 formulation after
        else:
            self.iir_lt[t] = self.beta[t-1] * self.i_lt[t] + (1 - self.beta[t-1]) * self.iir_lt[t-1]
            self.iir[t] = self.alpha[t-1] * self.i_st[t] + (1 - self.alpha[t-1]) * self.iir_lt[t]

        # Replace all 10 < iir < 0 with previous period value
        for iir in [self.iir, self.iir_lt]:
            if iir[t] < 0 or iir[t] > 10 or np.isnan(iir[t]):
                iir[t] = iir[t-1]
    
    def _calculate_interest(self, t):
        """
        Calculate interest payments
        """
        # Calculate interest payments on newly issued debt
        self.interest_st[t] = self.D_st[t-1] * self.i_st[t-1] / 100 # interest payments on newly issued short-term debt
        self.interest_lt[t] = self.iir_lt[t] / 100 * self.D_lt[t-1] # lt interest is t-1 lt debt times implicit lt interest rate
        self.interest[t] = self.interest_st[t] + self.interest_lt[t] # interest payments on newly issued debt and outstanding legacy debt
        self.interest_ratio[t] = self.interest[t] / self.ngdp[t] * 100

    def _calculate_amortization(self, t):
        """
        Calculate amortization payments
        """
        # Calculate amortization payments on newly issued debt
        self.amortization_st[t] = self.D_st[t-1] # amortization payments on short-term debt share in last years gross financing needs
        self.amortization_lt[t] = self.share_lt_maturing[t] * self.D_lt[t-1] + self.amortization_lt_inst[t] # lt amortization based on maturing share and inst debt
        self.amortization[t] = self.amortization_st[t] + self.amortization_lt[t] # amortization of newly issued st, lt debt

    def _calculate_debt_stock(self, t):
        """
        Calculate new debt stock and distribution of new short and long-term issuance
        """
        # Total debt stock is equal to last period stock minus amortization plus financing needs
        self.D[t] = np.max([self.D[t-1] - self.amortization[t] + self.gfn[t], 0])

        # Distribution of short-term and long-term debt in financing needs
        D_stn_theoretical = self.share_st * self.D[t] # st debt to keep share equal to share_st
        D_ltn_theoretical = (1 - self.share_st) * self.D[t] - self.D_lt[t-1] + self.amortization_lt[t] # lt debt to keep share equal to 1 - share_st
        share_st_issuance = D_stn_theoretical / (D_stn_theoretical + D_ltn_theoretical) # share of st in gfn
        
        # Calculate short-term and long-term debt issuance
        self.D_st[t] = share_st_issuance * self.gfn[t]
        self.D_ltn[t] = (1 - share_st_issuance) * self.gfn[t]
        self.D_lt[t] = self.D_lt[t-1] - self.amortization_lt[t] + self.D_ltn[t]

    def _calculate_gfn(self, t):
        """
        Calculate gross financing needs
        """
        self.gfn[t] = np.max([self.interest[t] + self.amortization[t] - self.PB[t] + self.SF[t], 0])

    def _calculate_balance(self, t):
        """
        Calculate overall balance and structural fiscal balance
        """
        self.ob[t] = (self.PB[t] - self.interest[t]) / self.ngdp[t] * 100
        self.sb[t] = (self.SPB[t] - self.interest[t]) / self.ngdp[t] * 100 
        
    def _calculate_debt_ratio(self, t):
        """
        Calculate debt ratio
        """
        # Calculate stock flow ratio
        self.sf[t] = self.SF[t] / self.ngdp[t] * 100
        
        # Calculate debt ratio (floor zero)
        self.d[t] = np.max([
            self.share_domestic * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) \
            + self.share_eur * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_eur[t] / self.exr_eur[t-1]) \
            + self.share_foreign * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_usd[t] / self.exr_usd[t-1]) \
            - self.pb[t] + self.sf[t], 0
            ])
            
    #--------------------------------------------------------------------------------#
    #----------------------------- OPTIMIZATION METHODS -----------------------------# 
    #--------------------------------------------------------------------------------#
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

        # Define EDP threshold, set to 3% of GDP unless investment shock
        self.edp_target = np.full(self.adjustment_period, -3, dtype=float)
        if self.inv_exception: self.edp_target[:np.min([self.inv_period,self.adjustment_period+1])] -= 0.5

        # If deficit excessive, increase spb by 0.5 annually until deficit below 3% 
        if self.ob[self.adjustment_start] < self.edp_target[0]:
            
            # Set start indices for spb and pb adjustment parts of EDP
            self.edp_spb_index = 0
            self.edp_sb_index = 3

            # Calculate EDP adjustment steps for spb, sb, and final periods
            self._calculate_edp_spb()
            self._calculate_edp_sb()
            self._calculate_edp_end(spb_target = spb_target)

        # If deficit not excessive, set EDP period to 0
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 1 
                    
    def _save_edp_period(self):
        """
        Saves EDP period and end period
        """
        self.edp_period = np.where(~np.isnan(self.edp_steps))[0][-1] + 1
        self.edp_end = self.adjustment_start + self.edp_period

    def _calculate_edp_spb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral primary balance adjustment
        """
        # Loop for SPB part of EDP: min. 0.5 spb adjustment while deficit > 3 and in spb adjustmet period
        while (self.ob[self.adjustment_start + self.edp_spb_index] < self.edp_target[self.edp_spb_index]
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

    def _calculate_edp_sb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral balance adjustment
        """
        # Loop for SB balance part of EDP: min. 0.5 ob adjustment while deficit > 3 and before last period
        while (self.ob[self.adjustment_start + self.edp_sb_index] < self.edp_target[self.edp_sb_index]
                and self.edp_sb_index+1 < self.adjustment_period):

            # Initiate sb step at current adjustment_step value, increase by 0.001
            self.edp_steps[self.edp_sb_index] = self.adjustment_steps[self.edp_sb_index]
            self.edp_steps[self.edp_sb_index] += 0.001
            
            # Project using last periods SPB as target, move to next period
            self.project(
                spb_target=self.spb_target,
                edp_steps=self.edp_steps
                )
            self._save_edp_period()

            # If sb adjustment reaches min. 0.5, move to next period
            if self.sb[self.adjustment_start + self.edp_sb_index] - self.sb[self.adjustment_start + self.edp_sb_index - 1] >= 0.5:
                self.edp_sb_index += 1

    def _calculate_edp_end(self, spb_target):
        """
        Calculate EDP adjustment steps or SPB target ensuring deficit below 3% at adjustment end
        """   
        # If EDP lasts until penultimate adjustmet period, increase EDP steps to ensure deficit < 3
        if self.edp_period == self.adjustment_period:
            while self.ob[self.adjustment_end] <= self.edp_target[-1]:           

                # Aim for linear adjustment path by increasing smallest EDP steps first
                min_edp_steps = np.min(self.edp_steps[~np.isnan(self.edp_steps)])
                min_edp_indices = np.where(self.edp_steps == min_edp_steps)[0]
                self.edp_steps[min_edp_indices] += 0.001
                self.project(
                    spb_target=self.spb_target,
                    edp_steps=self.edp_steps
                    )
                self._save_edp_period()
        
        # If last EDP period has deficit < 3, we do not impose additional adjustment
        if self.ob[self.adjustment_start-1 + self.edp_period] >= self.edp_target[-1]:
            self.edp_steps[self.edp_sb_index:] = np.nan
            self._save_edp_period()
        
        # If no spb_target was specified, calculate to ensure deficit < 3 until adjustment end
        if spb_target is None:
            print('No SPB target specified, calculating to ensure deficit < 3')
            while np.any(self.ob[self.edp_end:self.adjustment_end+1] <= self.edp_target[-1]):
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

    def _deterministic_optimization(self, criterion, bounds, steps): #TODO find error 
        """
        Main loop of optimizer for debt safeguard
        """
        # If debt safeguard and EDP lasts until penultimate adjustment year, debt safeguard satisfied by default
        if (criterion == 'debt_safeguard'
            and self.edp_period >= self.adjustment_period - 1):
            self.spb_target = self.spb_bca[self.edp_end-1]
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
                self._get_adjustment_steps(criterion=criterion, spb_target=spb_target)
                self.project(
                    edp_steps=self.edp_steps,
                    adjustment_steps=self.adjustment_steps,
                    scenario=self.scenario
                    )
                
                # If condition is met, enter nested loop and decrease spb_target in small steps
                if self._deterministic_condition(criterion=criterion):  
                    while (self._deterministic_condition(criterion=criterion) 
                           and spb_target >= bounds[0]):
                        current_spb_target = spb_target
                        spb_target -= steps[1]
                        self._get_adjustment_steps(criterion=criterion, spb_target=spb_target)
                        self.project(
                            edp_steps=self.edp_steps,
                            adjustment_steps=self.adjustment_steps,
                            scenario=self.scenario
                            )
                    break

                # If condition is not met, increase spb_target in large steps
                current_spb_target = spb_target
                spb_target += steps[0]

            except:
                raise #Exception(f'No solution found for {criterion}')
            
        # If spb_target exceeds upper bound, raise exception    
        if spb_target > bounds[1] - steps[1]:
            raise #Exception(f'No solution found for {criterion}')

        # Return last valid spb_target as optimal spb and project with target
        self.spb_target = current_spb_target
        spb_target -= steps[1]
        self._get_adjustment_steps(criterion=criterion, spb_target=spb_target)

        return self.spb_bca[self.adjustment_end]

    def _get_adjustment_steps(self, criterion, spb_target):
        """
        Get adjustment steps for debt safeguard after EDP
        """
        # If debt safeguard, apply adjustment to period after EDP
        if criterion == 'debt_safeguard':
            num_steps = self.adjustment_period - self.edp_period
            step_size = (spb_target - self.spb_bca[self.edp_end-1]) / num_steps
            non_edp_steps = np.full(num_steps, step_size)
            edp_steps_nonan = self.edp_steps[~np.isnan(self.edp_steps)]
            self.adjustment_steps = np.concatenate([edp_steps_nonan, non_edp_steps])
        
        # Otherwise apply adjustment to all periods
        else:
            num_steps = self.adjustment_period
            step_size = (spb_target - self.spb_bca[self.adjustment_start-1]) / num_steps
            self.adjustment_steps = np.full(num_steps, step_size)

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
        return (np.all(np.diff(self.d[self.adjustment_end:self.adjustment_end+11]) < 0) 
                or self.d[self.adjustment_end+10] <= 60)

    def _deficit_reduction_criterion(self):
        """
        Checks the deficit reduction criterion for <3% deficit for 10 years after adjustment end.
        """
        return np.all(self.ob[self.adjustment_end:self.adjustment_end+11] >= -3)

    def _debt_safeguard_criterion(self):
        """
        Checks the debt safeguard criterion.
        """
        debt_safeguard_decline = 1 if self.d[self.adjustment_start - 1] > 90 else 0.5

        # Exception for investment shock
        if self.inv_exception:
            self._calculate_d_non_inv()
            
            # If EDP period is 0, need debt ratio from 2024, pre-investment
            if self.edp_period == 0:

                return (self.d[self.edp_end] - self.d_non_inv[-1] 
                        >= debt_safeguard_decline * (self.adjustment_end - self.edp_end))
            
            # If EDP period is not 0, need debt ratio from EDP period, post-investment
            else:
                return (self.d_non_inv[self.edp_period] - self.d_non_inv[-1] 
                        >= debt_safeguard_decline * (self.adjustment_end - self.edp_end))
        
        # Non-investment criteria with baseline debt ratio
        else:
            return (self.d[self.edp_end] - self.d[self.adjustment_end] 
                    >= debt_safeguard_decline * (self.adjustment_end - self.edp_end))

    def _calculate_d_non_inv(self):
        """
        Calculate the non-investment shock debt ratio.
        """
        # Calculate yearly investment based on nominal GDP
        D_inv = np.cumsum(0.005 * self.ngdp[self.adjustment_start:self.adjustment_start+self.inv_period])

        # Extend D_inv to adjustment_end+1
        D_inv = np.concatenate([D_inv, np.full(self.adjustment_period - self.inv_period, D_inv[-1])])

        # Add interest costs based on implicit rate in given year
        D_inv += D_inv * self.iir[self.adjustment_start:self.adjustment_end+1] / 100

        # Calculate ratio
        self.d_non_inv = ((self.D[self.adjustment_start:self.adjustment_end+1] - D_inv)
                    / self.ngdp[self.adjustment_start:self.adjustment_end+1]) * 100 

    def find_spb_deficit_resilience(self):
        """
        Apply the deficit resilience targets that sets min. annual spb adjustment if structural deficit exceeds 1.5%.
        """
        # Initialize deficit_resilience_steps and spb_post_adjustment_period
        self.deficit_resilience_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)
        self.post_adjustment_steps = np.full((self.T - self.adjustment_end-1,), 0, dtype=np.float64)

        # Define structural deficit target and correct for investment shock 
        self.deficit_resilience_target = np.full(self.adjustment_period, -1.5, dtype=float)
        self.post_adjustment_target = -1.5
        if self.inv_exception: self.deficit_resilience_target[:np.min([self.inv_period,self.adjustment_period+1])] -= 0.5

        # Define deficit resilience step size
        if self.adjustment_period == 4: 
            self.deficit_resilience_step = 0.4
        elif self.adjustment_period == 7:
            self.deficit_resilience_step = 0.25
        
        # Project baseline
        self.project(
            spb_target=self.spb_target, 
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps
            )
                
        # If exception for the investment shock is specified, apply deficit resilience from second year only
        self.deficit_resilience_start = self.adjustment_start
        
        # Loop for adjustment period violations of deficit resilience
        self._deficit_resilience_loop_adjustment()    

        # Loop for post-adjustment period violations of deficit resilience
        self._deficit_resilience_loop_post_adjustment()

        return self.spb_bca[self.adjustment_end]
    
    def _deficit_resilience_loop_adjustment(self):
        """
        Loop for adjustment period violations of deficit resilience
        """
        for t in range(self.deficit_resilience_start, self.adjustment_end+1):
            if (self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start] and self.adjustment_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8): # 1e-8 tolerance for floating point errors
                self.deficit_resilience_steps[t - self.adjustment_start] = self.adjustment_steps[t - self.adjustment_start]
                while (self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start] 
                    and self.deficit_resilience_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8): # 1e-8 tolerance for floating point errors
                    self.deficit_resilience_steps[t - self.adjustment_start] += 0.001
                    self.project(
                        spb_target=self.spb_target, 
                        edp_steps=self.edp_steps,
                        deficit_resilience_steps=self.deficit_resilience_steps
                        )
                
    def _deficit_resilience_loop_post_adjustment(self):
        """
        Loop for post-adjustment period violations of deficit resilience
        """
        for t in range(self.adjustment_end+1, self.adjustment_end+11):
            while (self.sb[t] <= self.post_adjustment_target 
                    and self.post_adjustment_steps[t - self.adjustment_end-1] < self.deficit_resilience_step - 1e-8): # 1e-8 tolerance for floating point errors
                self.post_adjustment_steps[t - self.adjustment_end-1] += 0.001
                self.project(
                    spb_target=self.spb_target, 
                    edp_steps=self.edp_steps,
                    deficit_resilience_steps=self.deficit_resilience_steps,
                    post_adjustment_steps=self.post_adjustment_steps
                    )

    # def _calculate_deficit_elasticity(self):   
    #     """
    #     Calculate elasticity of overall (structural) balance to strucutral porimary balance
    #     """
    #     # Set adjustment steps to baseline and get ob and sb
    #     adjustment_steps = np.full(self.adjustment_period, 0, dtype=float)
    #     self.inv_shock = False
    #     self.project(adjustment_steps=adjustment_steps)
    #     ob_bl = self.ob[self.adjustment_start:self.adjustment_end+1].copy()
    #     sb_bl = self.sb[self.adjustment_start:self.adjustment_end+1].copy()

    #     # Increase adjustment by investment shock size and get ob and sb
    #     adjustment_steps[0] -= self.inv_size
    #     self.project(adjustment_steps=adjustment_steps)
    #     ob_adj = self.ob[self.adjustment_start:self.adjustment_end+1].copy()
    #     sb_adj = self.sb[self.adjustment_start:self.adjustment_end+1].copy()

    #     # Calculate and safe difference in each adjustment period
    #     self.ob_inv_elasticity = ob_adj - ob_bl
    #     self.sb_inv_elasticity = sb_adj - sb_bl

    #     # Reset vars
    #     self.adjustment_steps = None
    #     self.spb_target = None
    #     self.inv_shock = True

    #-----------------------------------------------------------------------------#
    #----------------------------- AUXILIARY METHODS -----------------------------#
    #-----------------------------------------------------------------------------#

    def df(self, *vars, all=False):
        """
        Return a dataframe with the specified variables as columns and years as rows.
        Takes a variable name (string) or a list of variable names as input.
        Alternatively takes a dictionary as input, where keys are variables (string) and values are variable names.
        """
        # if no variables specified, return spb, ob, d
        if not vars and all==False:
            vars = ['d', 'ob', 'sb', 'spb_bca', 'spb_bca_adjustment']
        
        # if all option True specified, return all variables
        elif not vars and all==True:
            vars = ['d', # debt ratio
                    'spb_bca', # ageing-cost adjusted structural primary balance
                    'spb_bca_adjustment', # adjustment to ageing-cost adjusted structural primary balance
                    'spb', # structural primary balance
                    'pb', # primary balance
                    'ob', # overall balance
                    'sb', # structural balance
                    'ageing_component', # ageing component of primary balance
                    'cyclical_component', # cyclical component of primary balance
                    'interest_ratio', # interest payments as share of GDP
                    'ageing_cost', # ageing cost
                    'rg', # real GDP growth
                    'rg_pot', # potential real GDP growth
                    'ng',  # nominal GDP growth
                    'output_gap', # output gap
                    'pi', # inflation
                    'rgdp_pot', # potential real GDP
                    'rgdp', # real GDP
                    'ngdp', # nominal GDP
                    'i_st', # short-term interest rate
                    'i_lt', # long-term interest rate
                    'iir_lt', # implicit long-term interest rate
                    'iir', # implicit interest rate
                    'sf', # stock flow adjustment
                    'D', # debt level
                    'D_lt_inst', # long-term instirutional debt level 
                    'D_st', # short-term debt level
                    'D_lt', # long-term debt level
                    'amortization', # amortization
                    'amortization_lt', # long-term amortization
                    'amortization_lt_inst', # long-term institutional amortization
                    'interest', # interest payments
                    'interest_lt', # long-term interest payments
                    'interest_st', # short-term interest payments
                    ]
        
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

        var_values = [getattr(self, var) if isinstance(var, str) else var for var in vars]

        df = pd.DataFrame(
            {vars[i]: var for i, var in enumerate(var_values)},
            index=range(self.start_year, self.end_year + 1)
            )
        
        if var_names: df.columns = var_names
        df.reset_index(names='y', inplace=True)
        df.reset_index(names='t', inplace=True)
        df.set_index(['t', 'y'], inplace=True)
        
        return df
