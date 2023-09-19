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
# 1. Data Methods: These methods clean and combine input data.
# 2. Projection Methods: These methods handle the projection of economic variables such as GDP
# 3. Optimization and Auxiliary Methods: These methods include functions to optimize the primary
#    balance to meet specific criteria, check deterministic conditions, and the creation of DataFrames.
# 
# In addition, the Stochastic Model Subclass, a specialized subclass building upon this base class,
# provides additional features for stochastic projection of economic variables under uncertainty. The
# stochastic model subclass is defined in the file EcStochasticModelClass.py.
#
# For comments and suggestions please contact lennard.welslau[at]bruegel[dot]org
#
# Author: Lennard Welslau
# Date: 01/09/2023
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

class EcDsaModel:
    #-------------------------#
    #--- INIITIALIZE MODEL ---# 
    #-------------------------#
    def __init__(self, 
                 country, # ISO code of country
                 start_year=2022, # start year of projection, first year is baseline value
                 end_year=2053, # end year of projection
                 adjustment_period=4, # number of years for linear spb_bcoa adjustment
                 adjustment_start=2025, # start year of linear spb_bcoa adjustment
                 ):

        ## Initialize model parameters
        self.country = country
        self.start_year = start_year
        self.end_year = end_year
        self.T = self.end_year - start_year + 1
        self.adjustment_period = adjustment_period 
        self.adjustment_start = adjustment_start - start_year
        self.adjustment_end = adjustment_start + adjustment_period - start_year - 1

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
        self.spb_bcoa = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP
        self.pb_cyclical_adj = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP adjusted for cyclical_component component
        self.pb_cyclical_ageing_adj = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP adjusted for cyclical and ageing_component
        self.pb = np.full(self.T, np.nan, dtype=np.float64) # primary balance over GDP

        # Implicit interest rate projection
        self.share_lt_maturing = np.full(self.T, np.nan, dtype=np.float64) # share of long-term debt maturing in the current year
        self.interest_st = np.full(self.T, np.nan, dtype=np.float64) # interest payment on short-term debt
        self.interest_lt = np.full(self.T, np.nan, dtype=np.float64) # interest payment on long-term debt
        self.interest = np.full(self.T, np.nan, dtype=np.float64) # interest payment on total debt
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
    
    #-------------------------------#
    #--- DATA METHODS (INTERNAL) ---#
    #-------------------------------#  
    def _import_data(self):
        """
        Import data from Excel input data file.
        """

        # ESM ESFS loans and interest payments if country in ESM data
        self.df_debt_inst = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='esm_data')
        if self.country in self.df_debt_inst['ISO'].unique():
            self.df_debt_inst = self.df_debt_inst.loc[self.df_debt_inst['ISO'] == self.country]
            self.D_lt_inst[0] = self.df_debt_inst['amortization'].sum()
            self.esm = True
        else:
            self.esm = False

        # Ameco projections
        self.df_ameco = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='ameco_data')
        self.pi_ea_2024 = self.df_ameco.loc[(self.df_ameco['year'] == 2024) & (self.df_ameco['ISO'] == 'EA20'), 'gdp_def_pch'].values[0]
        self.df_ameco = self.df_ameco.loc[self.df_ameco['ISO'] == self.country]
        self.ameco_end_y = self.df_ameco['year'].max()
        self.ameco_end_t = self.ameco_end_y - self.start_year

        # GDP t+5 projections from output gap working group
        self.df_output_gap_working_group = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='output_gap_working_group')
        self.df_output_gap_working_group = self.df_output_gap_working_group.loc[self.df_output_gap_working_group['ISO'] == self.country]
        self.growth_projection_end = self.df_output_gap_working_group['year'].max()

        # Commission provided data
        df_commission = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='commission_data')
        self.df_ageing_cost = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['ageing_cost']
        self.df_real_growth = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['real_growth']
        self.df_property_income = df_commission.loc[df_commission['ISO'] == self.country].set_index('year')['property_income']

        # Market rates
        df_gov_rates = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='bbg_data')
        self.df_2022_benchmark_rates = df_gov_rates.loc[df_gov_rates['year'] == 2022, ['ISO', '3M', '10Y']]
        self.df_2023_benchmark_rates = df_gov_rates.loc[df_gov_rates['year'] == 2023, ['ISO', '3M', '10Y']]
        self.df_fwd_rates = df_gov_rates.loc[df_gov_rates['year'] == 2023, ['ISO', '3M10Y', '10Y10Y']]
        
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
        self.benchmark_rate_lt_2022 = self.df_2022_benchmark_rates.loc[(self.df_2022_benchmark_rates['ISO'] == self.country), '10Y'].values[0]
        self.benchmark_rate_lt_2023 = self.df_2023_benchmark_rates.loc[(self.df_2023_benchmark_rates['ISO'] == self.country), '10Y'].values[0]

        # Inflation rate
        self.df_infl_fwd = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name= 'inflation_fwd')
        self.infl_fwd = self.df_infl_fwd.loc[(self.df_infl_fwd['maturity'] == 10), 'infl_expectation'].values[0]

        # ECB debt data
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

        # Budget semi-elasticities
        self.df_elasticity = pd.read_excel('../data/InputData/deterministic_model_data.xlsx', sheet_name='budget_elasticity')
        self.pb_elasticity = self.df_elasticity.loc[(self.df_elasticity['ISO'] == self.country), 'budget_elasticity'].values[0]
        
    def _clean_data(self):
        """
        Clean data, extrapolate missing values, and save as vectors.
        """
        self._clean_rgdp_pot()
        self._clean_rgdp()
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
        Uses T+2 from Ameco for totals, T+5 for growth rates then use commission august real gdp growth rates
        """
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            if y <= self.ameco_end_y:
                self.rg_pot[t] = self.df_output_gap_working_group.loc[self.df_output_gap_working_group['year'] == y, 'gdp_pot_pch'].values[0]
                self.rgdp_pot[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'gdp_pot'].values[0]
            elif y > self.ameco_end_y and y <= self.growth_projection_end:
                self.rg_pot[t] = self.df_output_gap_working_group.loc[self.df_output_gap_working_group['year'] == y, 'gdp_pot_pch'].values[0]
                self.rgdp_pot[t] = self.rgdp_pot[t-1] * (1 + self.rg_pot[t] / 100)
            elif y > self.growth_projection_end:
                self.rg_pot[t] = self.df_real_growth.loc[y]
                self.rgdp_pot[t] = self.rgdp_pot[t-1] * (1 + self.rg_pot[t] / 100)

    def _clean_rgdp(self):
        """
        Clean baseline real growth. 
        Uses T+2 from Ameco for totals, T+5 for growth rates, then equal to potential growth.
        """        
        for t, y in enumerate(range(self.start_year, self.end_year+1)):
            self.rg_bl[t] = self.df_real_growth.loc[y] 
            
            if y <= self.ameco_end_y:
                self.rgdp_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'rgdp'].values[0]
            elif y > self.ameco_end_y:
                self.rgdp_bl[t] = self.rgdp_bl[t-1] * (1 + self.rg_bl[t] / 100)
            
        # Set initial values to baseline
        self.rg = np.copy(self.rg_bl)
        self.rgdp = np.copy(self.rgdp_bl)

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
            self.df_pi.loc[2033, 'gdp_def_pch'] = self.infl_fwd + (self.df_pi.loc[2024].values[0] - self.pi_ea_2024) / 2
            self.df_pi.loc[2053, 'gdp_def_pch'] = 2.5
            self.df_pi.loc[2033] = self.infl_fwd + (self.df_pi.loc[2024].values[0] - self.pi_ea_2024) / 2
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
                self.ng_bl[t] = self.df_output_gap_working_group.loc[self.df_output_gap_working_group['year'] == y, 'gdp_nom_pch']
                self.ngdp_bl[t] = self.df_ameco.loc[self.df_ameco['year'] == y, 'ngdp'].values[0]
            elif y > self.ameco_end_y and y <= self.growth_projection_end:
                self.ng_bl[t] = self.df_output_gap_working_group.loc[self.df_output_gap_working_group['year'] == y, 'gdp_nom_pch']
                self.ngdp_bl[t] = self.ngdp_bl[t-1] * (1 + self.ng_bl[t] / 100)
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
            self.spb_bcoa[t] = self.spb_bl[t]
    
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
            ## Loop over years and calculate amortization and interest payments of inst debt
            for t, y in enumerate(range(self.start_year+1, self.end_year+1)):
                t += 1 # start in t = 1
                self.amortization_lt_inst[t] = self.df_debt_inst.loc[self.df_debt_inst['year'] == y, 'amortization'].values[0]
                self.D_lt_inst[t] = self.D_lt_inst[t-1] - self.amortization_lt_inst[t]
            
        
        # Calculate share of lt debt maturing each year
        # Set t and t + 10 value of maturing lt debt share
        self.share_lt_maturing[2023-self.start_year] = self.share_lt_maturing_t0 
        self.share_lt_maturing[(2033-self.start_year):] = self.share_lt_maturing_t10

        # Interpolate missing values
        x = np.arange(len(self.share_lt_maturing))
        mask = np.isnan(self.share_lt_maturing)
        self.share_lt_maturing[mask] = np.interp(x[mask], x[~mask], self.share_lt_maturing[~mask])

    #--------------------------#
    #--- PROJECTION METHODS ---# 
    #--------------------------#
    def project(self,
                spb_target=None,
                spb_initial_adjustment_period=0,
                spb_initial_adjustment_step=0.5,
                scenario=None):
        """
        Project debt dynamics
        """
        # Clear starting values from memory
        self.D_ltn = np.full(self.T, 0, dtype=np.float64) 
        self._clean_iir()

        # Set spb_target if specified
        self.spb_target = spb_target

        # Set spb_initial_adjustment_period if specified
        self.spb_initial_adjustment_period = spb_initial_adjustment_period
        self.spb_initial_adjustment_step = spb_initial_adjustment_step
        
        # Set scenario parameter
        if scenario == 'main_adjustment':
            self.scenario = None
        else:
            self.scenario = scenario

        ## Project debt dynamics
        self._project_market_rate()
        self._project_spb_bcoa()
        self._project_gdp()
        self._project_pb()
        self._project_d()
   
    def _project_market_rate(self):
        """
        Project market rate data, needs to be in projection method because of scenario parameter.
        Uses BBG forward rates upo to T+10, then linearly interpolates to long-term values.
        """
        # Clean vectors in case of repeated projection with differen scenarios
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

        # Interpolate missing values
        x_st = np.arange(len(self.i_st))
        mask_st = np.isnan(self.i_st)
        self.i_st[mask_st] = np.interp(x_st[mask_st], x_st[~mask_st], self.i_st[~mask_st])

        x_lt = np.arange(len(self.i_lt))
        mask_lt = np.isnan(self.i_lt)
        self.i_lt[mask_lt] = np.interp(x_lt[mask_lt], x_lt[~mask_lt], self.i_lt[~mask_lt])

    def _project_spb_bcoa(self):
        """
        Project structural primary balance
        """
        for t in range(1, self.T):
            
            # If no spb_target specified, spb_bcoa is constant after adjustment start
            if not self.spb_target and self.spb_initial_adjustment_period == 0:
                if t > self.adjustment_start:
                    self.spb_bcoa[t] = self.spb_bcoa[t-1]

            # If spb_target specified and not spb_initial_adjustment_period, increase linearly within adjustment period
            elif self.spb_target and self.spb_initial_adjustment_period == 0:
                # Within adjustment period, spb_bcoa increases linearly to spb_target
                if t in range(self.adjustment_start, self.adjustment_end+1):
                    self.spb_bcoa_step = (self.spb_target - self.spb_bl[self.adjustment_start-1]) / (self.adjustment_period)
                    self.spb_bcoa[t] = self.spb_bcoa[t-1] + self.spb_bcoa_step
            
                # After adjustment period, spb_bcoa is constant
                elif t > self.adjustment_end:
                    if self.scenario == 'lower_spb' and self.adjustment_period == 4 and t <= self.adjustment_end + 2:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1] - 0.5 / 2
                    elif self.scenario == 'lower_spb' and self.adjustment_period == 7 and t <= self.adjustment_end + 3:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1] - 0.5 / 3
                    else:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1]

            # If spb_initial_adjustment_period specified, increase spb by 0.5 for given periods
            elif self.spb_initial_adjustment_period > 0:

                # Within spb_initial_adjustment_period, spb_bcoa increases by 0.5
                if t in range(self.adjustment_start, self.adjustment_start + self.spb_initial_adjustment_period):
                    self.spb_bcoa[t] = self.spb_bcoa[t-1] + self.spb_initial_adjustment_step
                
                # After spb_initial_adjustment_period, if spb_target is defined spb_bcoa increases linearly to spb_target
                elif t in range(self.adjustment_start + self.spb_initial_adjustment_period, self.adjustment_end+1) and self.spb_target:
                    self.spb_bcoa_step = (self.spb_target - self.spb_bcoa[self.adjustment_start + self.spb_initial_adjustment_period - 1]) / (self.adjustment_period - self.spb_initial_adjustment_period)
                    self.spb_bcoa[t] = self.spb_bcoa[t-1] + self.spb_bcoa_step

                # After spb_initial_adjustment_period, if spb_target is not defined spb_bcoa is constant
                elif t in range(self.adjustment_start + self.spb_initial_adjustment_period, self.adjustment_end+1) and not self.spb_target:
                    self.spb_bcoa[t] = self.spb_bcoa[t-1]

                # After adjustment period, spb_bcoa is constant
                elif t > self.adjustment_end:
                    if self.scenario == 'lower_spb' and self.adjustment_period == 4 and t <= self.adjustment_end + 2:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1] - 0.5 / 2
                    elif self.scenario == 'lower_spb' and self.adjustment_period == 7 and t <= self.adjustment_end + 3:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1] - 0.5 / 3
                    else:
                        self.spb_bcoa[t] = self.spb_bcoa[t-1]

    def _project_gdp(self):
        """
        Project nominal GDP.
        """  
        for t in range(1, self.T):
            
            ## Real growth
            # fiscal multiplier effect from change in SPB relative to baseline
            self.fm_effect[t] = self.fm * ((self.spb_bcoa[t] - self.spb_bcoa[t-1]) - (self.spb_bl[t] - self.spb_bl[t-1]))   
            
            # Effect on output gap
            self.output_gap[t] = self.output_gap_bl[t] - self.fm_effect[t] - 2/3 * self.fm_effect[t-1] - 1/3 * self.fm_effect[t-2]
            
            # Real growth and real GDP
            self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]
            self.rg[t] = (self.rgdp[t] - self.rgdp[t-1]) / self.rgdp[t-1] * 100

            ## Nominal growth
            # Before adjustment period, nominal growth is baseline
            if t < self.adjustment_start:
                self.ng[t] = self.ng_bl[t]
                self.ngdp[t] = self.ngdp_bl[t]

            # After adjustment period, nominal growth based on real growth and inflation
            elif t >= self.adjustment_start:
                self.ng[t] = (1 + self.rg[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100 

                # Adjust nominal growth for adverse r-g scenario
                if self.scenario == 'adverse_r_g' and t > self.adjustment_end:
                    self.ng[t] -= 0.5
            
                # project nominal GDP
                self.ngdp[t] = self.ngdp[t-1] * (1 + self.ng[t] / 100) 

    def _project_pb(self):
        """
        Project primary balance adjusted for cyclical_component component and ageing costs
        """
        for t in range(1, self.T):
            # Get real-potential gdp growth gap and cyclical_component component of primary balance
            self.output_gap[t] = (self.rgdp[t] / self.rgdp_pot[t] - 1) * 100

            # Compute components
            self.cyclical_component[t] = self.pb_elasticity * self.output_gap[t]
            self.ageing_component[t] = - (self.ageing_cost[t] - self.ageing_cost[self.adjustment_end])
            self.property_income_component[t] = self.property_income[t] - self.property_income[self.adjustment_start - 1]

            # Up to adjustment_start, use baseline pb
            if t < self.adjustment_start:
                self.pb[t] = self.pb_bl[t]

            # During adjustment period, primary balance only corrected for cyclical_component
            elif t >= self.adjustment_start and t <= self.adjustment_end:
                self.pb_cyclical_adj[t] = self.spb_bcoa[t] + self.cyclical_component[t] + self.property_income_component[t]
                self.pb[t] = self.pb_cyclical_adj[t]

            # After adjustment period, primary balance corrected for cyclical_component and ageing costs
            if t > self.adjustment_end:
                
                self.pb_cyclical_ageing_adj[t] = self.spb_bcoa[t] + self.ageing_component[t] + self.cyclical_component[t] + self.property_income_component[t]
                self.pb[t] = self.pb_cyclical_ageing_adj[t]

            # calculate stock flow over GDP
            self.sf[t] = self.SF[t] / self.ngdp[t] * 100

            # Total primary balance in model
            self.PB[t] = self.pb[t] / 100 * self.ngdp[t]

            # Total SPB for reference
            self.SPB[t] = self.spb_bcoa[t] / 100 * self.ngdp[t]

    def _project_d(self):
        """
        Loop for debt dynamics        
        """
        ## Adjust interest rates for adverse r-g scenario
        if self.scenario == 'adverse_r_g':
            self._adjustment_adverse_r_g()

        for t in range(1,self.T):
            
            ## Calculate implicit interest rate
            self._calculate_iir(t)

            ## Adjust interest rates for adverse financial stress scenario
            if self.scenario == 'financial_stress' and t == self.adjustment_end+1:
                self._adjustment_financial_stress(t)

            ## Interest and amortization payments 
            # Short-term debt
            self.interest_st[t] = self.D_st[t-1] * self.i_st[t-1] / 100 # interest payments on newly issued short-term debt
            self.amortization_st[t] = self.D_st[t-1] # amortization payments on short-term debt share in last years gross financing needs

            # Long-term debt
            self.interest_lt[t] = self.iir_lt[t] / 100 * self.D_lt[t-1] # lt interest is t-1 lt debt times implicit lt interest rate
            self.amortization_lt[t] = self.share_lt_maturing[t] * self.D_lt[t-1] + self.amortization_lt_inst[t] # lt amortization based on maturing share and inst debt

            # Aggregate interest and amortization payments
            self.interest[t] = self.interest_st[t] + self.interest_lt[t] # interest payments on newly issued debt and outstanding legacy debt
            self.amortization[t] = self.amortization_st[t] + self.amortization_lt[t] # amortization of newly issued st, lt debt

            ## Gross financing needs based on old and new financing costs and primary balance
            self.gfn[t] = np.max([self.interest[t] + self.amortization[t] - self.PB[t] + self.SF[t], 0])

            ## New debt stock based on new issuance and debt stock last year
            # Total debt
            self.D[t] = np.max([self.D[t-1] - self.amortization[t] + self.gfn[t], 0])

            # Distribution of short-term and long-term debt in financing needs
            D_stn_theoretical = self.share_st * self.D[t] # st debt to keep share equal to share_st
            D_ltn_theoretical = (1 - self.share_st) * self.D[t] - self.D_lt[t-1] + self.amortization_lt[t] # lt debt to keep share equal to 1 - share_st
            share_st_issuance = D_stn_theoretical / (D_stn_theoretical + D_ltn_theoretical) # share of st in gfn
            
            # Calculate short-term and long-term debt issuance
            self.D_st[t] = share_st_issuance * self.gfn[t]
            self.D_ltn[t] = (1 - share_st_issuance) * self.gfn[t]
            self.D_lt[t] = self.D_lt[t-1] - self.amortization_lt[t] + self.D_ltn[t]

            ## Calculate the overall balance
            self.ob[t] = (self.PB[t] - self.interest[t]) / self.ngdp[t] * 100
            
            ## Calculate the debt to GDP ratio
            self.d[t] = np.max([
                self.share_domestic * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) \
                + self.share_eur * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_eur[t] / self.exr_eur[t-1]) \
                + self.share_foreign * self.d[t-1] * (1 + self.iir[t] / 100) / (1 + self.ng[t] / 100) * (self.exr_usd[t] / self.exr_usd[t-1]) \
                - self.pb[t] + self.sf[t], 0
                ])

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
            self.iir_lt[t] = self.beta[t-1] * self.i_lt[t] + (1 - self.beta[t-1]) * self.iir[t-1]
            self.iir[t] = self.alpha[t-1] * self.i_st[t] + (1 - self.alpha[t-1]) * self.iir_lt[t]

            # Alternative formulation    
            # self.iir[t] = self.interest[t] / self.D[t-1] * 100

        # Replace all 10 < iir < 0 with previous period value
        for iir in [self.iir, self.iir_lt]:
            if iir[t] < 0 or iir[t] > 10 or np.isnan(iir[t]):
                iir[t] = iir[t-1]

    def _adjustment_adverse_r_g(self):
        """ 
        Adjust interest rates for adverse r-g scenario
        """
        # Adjust market rates for adverse r-g scenario
        self.i_st[self.adjustment_end+1:] += 0.5
        self.i_lt[self.adjustment_end+1:] += 0.5

    def _adjustment_financial_stress(self, t):
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
            
    #----------------------------#
    #--- OPTIMIZATION METHODS ---# 
    #----------------------------#
    
    def find_spb_deterministic(self, bounds=(-10, 10), steps=[0.01, 0.0001], scenario='main_adjustment', criterion='debt_decline'):
        """
        Find the primary balance that ensures a decline in the debt ratio after the adjustment period.
        """
        # Check input parameters
        if scenario not in [None, 'main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']:
            raise ValueError("scenario must be None/'main_adjustment', 'lower_spb', 'financial_stress', or 'adverse_r_g'")
        if criterion not in ['debt_decline', 'debt_safeguard', 'expenditure_safeguard', 'deficit_reduction', 'debt_safeguard_deficit']:
            raise ValueError("criterion must be 'debt_decline', 'debt_safeguard', 'expenditure_safeguard', 'debt_safeguard_deficit', or 'deficit_reduction'")
        if not hasattr(self, 'spb_initial_adjustment_period'):
           self.spb_initial_adjustment_period = 0
           self.spb_initial_adjustment_step = 0.5

        # Initialize spb_target to the lower bound
        spb_target = bounds[0]  

        while spb_target <= bounds[1]:
            try:

                # Project the model with the current spb_target
                self.project(spb_target=spb_target, spb_initial_adjustment_period=self.spb_initial_adjustment_period, spb_initial_adjustment_step=self.spb_initial_adjustment_step, scenario=scenario)

                # If condition is met, enter nested loop and decrease spb_target in small steps
                if self._deterministic_condition(criterion=criterion):  
                    while self._deterministic_condition(criterion=criterion) and spb_target >= bounds[0]:
                        previous_spb_target = spb_target
                        spb_target -= steps[1]
                        self.project(spb_target=spb_target, spb_initial_adjustment_period=self.spb_initial_adjustment_period, spb_initial_adjustment_step=self.spb_initial_adjustment_step, scenario=scenario)
                    break

                # If condition is not met, increase spb_target in large steps
                previous_spb_target = spb_target
                spb_target += steps[0]
            except:
                raise

        if spb_target > bounds[1] - steps[1]:
            raise Exception(f'No solution found for {scenario} {criterion}')

        self.project(spb_target=previous_spb_target, spb_initial_adjustment_period=self.spb_initial_adjustment_period, spb_initial_adjustment_step=self.spb_initial_adjustment_step, scenario=scenario)
        return previous_spb_target
    
    def _deterministic_condition(self, criterion):
        """
        Check if the deterministic or safeguard condition is met.
        """

        # Define conditions
        low_debt = self.d[self.adjustment_start - 1] <= 60
        low_deficit = self.ob[self.adjustment_start - 1] >= -3

        debt_decline_criterion = np.all(np.diff(self.d[self.adjustment_end:self.adjustment_end+11]) < 0) or self.d[self.adjustment_end+10] <= 60 
        
        debt_safeguard_criterion = (self.d[self.adjustment_start+3] < self.d[self.adjustment_start - 1])
        
        debt_safeguard_deficit = debt_safeguard_criterion and np.all(self.ob[self.adjustment_start+np.max([self.spb_initial_adjustment_period,4]):self.adjustment_end+1] >= -3)
        
        deficit_reduction_criterion = np.all(self.ob[self.adjustment_end:self.adjustment_end+11] >= -3)
        
        expenditure_growth = self.spb_bcoa[self.adjustment_start-1] <= self.spb_bcoa[self.adjustment_end]

        # For 2024 deficit <3% and debt <60% only ensure <3% deficit for 10 years after adjustment end
        if low_debt and low_deficit and criterion != 'deficit_reduction':
            raise Exception("Only 'deficit_reduction' criterion for countries with debt ratio <= 60% and deficit <= 3%")
        
        # Debt decline from adjustment end to 10 years after adjustment end
        elif criterion == 'debt_decline' and debt_decline_criterion:
            return True
        
        # Safeguard ensures debt ratio is lower at end of adjustment than at start
        elif criterion == 'debt_safeguard' and debt_safeguard_criterion:
            return True
        
        # Safeguard ensures debt ratio is lower at end of adjustment than at start and deficit is <3% within adjustment_period
        elif criterion == 'debt_safeguard_deficit' and debt_safeguard_deficit:
            return True
        
        # Deficit criterion for <3% deficit for 10 years after adjustment end
        elif criterion == 'deficit_reduction' and deficit_reduction_criterion:
            return True
        
        # Expenditure safeguard ensures that expenditure growth remains below output growth
        elif criterion == 'expenditure_safeguard' and  expenditure_growth:
            return True
        
        # Otherwise, condition is not met
        else:
            return False        
        
    #-------------------------#
    #--- AUXILIARY METHODS ---# 
    #-------------------------#

    def df(self, *vars):
        """
        Return a dataframe with the specified variables as columns and years as rows.
        Takes a variable name (string) or a list of variable names as input.
        Alternatively takes a dictionary as input, where keys are variables (string) and values are variable names.
        """
        if isinstance(vars[0], dict):
            var_dict = vars[0]
            var_names = list(var_dict.values())
            vars = list(var_dict.keys())
        elif isinstance(vars[0], list):
            vars = vars[0]
            var_names = None
        else:
            var_names = None

        var_values = [getattr(self, var) if isinstance(var, str) else var for var in vars]

        df = pd.DataFrame({vars[i]: var for i, var in enumerate(var_values)},
                        index=range(self.start_year, self.end_year + 1))
        if var_names:
            df.columns = var_names
        df.reset_index(names='y', inplace=True)
        df.reset_index(names='t', inplace=True)
        df.set_index(['t', 'y'], inplace=True)
        return df
