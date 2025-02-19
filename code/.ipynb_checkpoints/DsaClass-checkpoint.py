import pandas as pd
import numpy as np
import os
import pycountry
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

class DsaClass:

    # ------------------------------------------------------------------------------ #
    # --------------------------------- Class init --------------------------------- #
    # ------------------------------------------------------------------------------ #    
    
    def __init__(self, iso, t=2023, T=2060, weo=False):
        
        self.t = t # set start of projection
        self.T = T # set end of projection
        self.iso = iso # Country
        self.weo = weo # Boolean to choose use of weo pb and debt projection

        # Load input data
        self.df_weo = pd.read_csv('../data/InputData/weo_data.csv') # IMF WEO
        self.df_weo = self.df_weo.loc[(self.df_weo['ISO'] == iso)].replace('--',np.nan)
    
        self.df_term_structure = pd.read_csv('../data/InputData/term_structure_data.csv') # EIKOS term structure of gpv debt
        self.df_term_structure = self.df_term_structure.loc[(self.df_term_structure['ISO'] == iso)]
        
        self.df_ecb_st_shares = pd.read_csv('../data/InputData/ecb_debt_data.csv') # ECB debt data
        self.df_ecb_st_shares = self.df_ecb_st_shares.loc[(self.df_ecb_st_shares['ISO'] == iso)]
                
        self.df_ecb_st_rates = pd.read_csv('../data/InputData/ecb_st_rate_data.csv') # Avg ECB short term gov bond rate
        self.df_ecb_st_rates = self.df_ecb_st_rates.loc[(self.df_ecb_st_rates['ISO'] == iso)]
        
        self.df_fwd_rates = pd.read_csv('../data/InputData/fwd_rates_data.csv') # Bloomberg gov fwd rates
        self.df_fwd_rates = self.df_fwd_rates.loc[(self.df_fwd_rates['ISO'] == iso)]
        
        self.df_infl_expectation = pd.read_csv('../data/InputData/infl_expectatation.csv') # Swap implied infl expecations
        
        self.df_esm = pd.read_csv('../data/InputData/esm_efsf_data.csv') # ESM ESFS loans and interest payments
        self.df_esm = self.df_esm.loc[(self.df_esm['ISO'] == iso)]
        
        # Initialize dictionaries 
        self.ngdp = {}
        self.pi = {}
        self.ng = {}
        self.pb_pct_ngdp = {}
        self.pb = {}
        self.fb = {}
        self.debt = {}
        self.debt_ratio = {}
        self.net_lb = {}
        self.implicit_rate = {}
        self.amortization_old = {}
        self.interest_old = {}
        self.debt_old = {}
        self.gfn = {}
        self.gfn_pct_ngdp = {}
        self.amortization_new = {}
        self.amortization_new_st = {}
        self.amortization_new_lt = {i: {} for i in range(1, 31)}        
        self.interest_new = {}
        self.interest_new_st = {}
        self.interest_new_lt = {i: {} for i in range(1, 31)}      
        self.interest_nm = {}
        self.fwd_rate = {}
        self.debt_new_st = {}
        self.debt_new_lt = {i: {} for i in range(1, 31)}        
        self.debt_new = {}
        self.g_error = 0
        self.i_error = 0
        
        # get additional data with methods
        self._get_weo()
        self._get_term_structure()
        self._get_fwd_rates()
        self._get_ecb()
        self._get_alpha()
         
    # ------------------------------------------------------------------------------ #
    # ------------------------ Init data extraction methods ------------------------ #
    # ------------------------------------------------------------------------------ #
    
    # Define method to extract WEO data and populates dictionaries
    def _get_weo(self):
        self.weo_start = self.df_weo['year'].iloc[0]
        self.weo_end = self.df_weo['year'].iloc[-1]
        self.adjustment_end = self.weo_end+1 # Define the adjustment period for primary balance used in the pb projection 
        self.rg = float(self.df_weo.loc[self.df_weo['year'] == self.weo_end,'NGDP_RPCH']) # ngdp growth for future values is inflated rgdp growth
        for t in range(self.weo_start, self.weo_end+1):
            self.ngdp[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'NGDP'])
            if t > self.weo_start: # ngdp growth from t+1
                self.ng[t] = (self.ngdp[t] - self.ngdp[t-1]) / self.ngdp[t-1] * 100
            self.pb_pct_ngdp[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'GGXONLB_NGDP'])
            self.pb[t] = self.pb_pct_ngdp[t] / 100 * self.ngdp[t]
        self.pb_pct_ngdp[self.weo_end+1] = float(self.df_weo.loc[self.df_weo['year'] == self.weo_end,'GGXONLB_NGDP']) # Set initial value for 2029 pb to be projected
        if self.weo:  # debt and implicit rate only used from weo until 2028 if specified, else only untill prjection start
            for t in range(self.weo_start, self.weo_end+1):
                self.debt[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'GGXWDG'])
                self.debt_old[t] =  self.debt[t]
                self.debt_ratio[t] = self.debt[t] / self.ngdp[t] * 100
                self.net_lb[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'GGXCNL_NGDP'])
                self.implicit_rate[t] = (self.pb_pct_ngdp[t] - self.net_lb[t]) * self.ngdp[t] / self.debt[t]
        else:
            for t in range(self.weo_start, self.t+1):
                self.debt[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'GGXWDG'])
                self.debt_old[t] =  self.debt[t]
                self.debt_ratio[t] = self.debt[t] / self.ngdp[t] * 100
                self.net_lb[t] = float(self.df_weo.loc[self.df_weo['year'] == t,'GGXCNL_NGDP'])
                self.implicit_rate[t] = (self.pb_pct_ngdp[t] - self.net_lb[t]) * self.ngdp[t] / self.debt[t]
        
    # Define method to get relevant data from df_term_structure and populates dictioanries
    def _get_term_structure(self):
        for t in range(self.weo_start, self.T+1):
            amortization_private = float(self.df_term_structure.loc[(self.df_term_structure['year'] == t) & (self.df_term_structure['Event'] == 'Principal'), 'amount_lc_bn'].sum())
            interest_private = float(self.df_term_structure.loc[(self.df_term_structure['year'] == t) & (self.df_term_structure['Event'] != 'Principal'), 'amount_lc_bn'].sum())
            if t in list(self.df_esm['year']):
                amortization_esm = float(self.df_esm.loc[self.df_esm['year'] == t, 'amortization'])
                interest_esm = float(self.df_esm.loc[self.df_esm['year'] == t, 'interest'])
            else: 
                amortization_esm = 0
                interest_esm = 0
            self.amortization_old[t] = amortization_private + amortization_esm
            self.interest_old[t] = interest_private + interest_esm
        
    # Define method to get forward rates from Bloomberg data and interpolates across time and maturities
    def _get_fwd_rates(self):
        df_long = pd.melt(self.df_fwd_rates, id_vars=['ISO'], var_name='term', value_name='rate')  # Convert long format to wide format
        df_long['m'] = df_long['term'].str.extract(r'^(\d+)').astype(int)  # Extract the maturity column
        df_long['t'] = df_long['term'].str.extract(r'(\d+)Y$').astype(int)  # Extract the tenor column
        df_wide = df_long.pivot(index='m', columns='t', values='rate')  # Convert back to wide format
        df_wide = df_wide.reindex(sorted(df_wide.columns, key=int), axis=1)  # Sort columns by tenor
        df_wide = df_wide.reindex(sorted(df_wide.index, key=int), axis=0)  # Sort rows by maturity
        df_wide = df_wide.reset_index().set_index('m')  # Reset the index to maturity
        new_index = pd.RangeIndex(df_wide.index.min(), df_wide.index.max() + 1)  # Create a new index
        df_wide = df_wide.reindex(new_index).interpolate()  # Interpolate missing values
        df_wide.index.names = ['m']  # Rename index to maturity
        df_wide = df_wide.T  # Transpose the DataFrame
        new_index = pd.RangeIndex(df_wide.index.min(), df_wide.index.max() + 1)  # Create a new index
        df_wide = df_wide.reindex(new_index).interpolate()  # Interpolate missing values
        df_wide.index.names = ['t']  # Rename index to tenor
        rates_t30 = df_wide.iloc[-1]  # Extract the 30-year rates
        rates_T = pd.concat([rates_t30] * (self.T-self.t-30), axis=1, ignore_index=True).T.rename(index=lambda x: x + 31)  # Create a DataFrame of constant 30-year rates for remaining t
        df_wide = pd.concat([df_wide, rates_T], ignore_index=False)  # Concatenate the DataFrames
        self.df_fwd_rates_new = df_wide
        self.df_yield_curve = df_wide.rename(index=lambda x: x + self.weo_end-5)  # Rename the index to years
        self.fwd_rate = self.df_yield_curve.to_dict()  # Convert DataFrame to a dictionary
        for m in self.fwd_rate.keys(): # Use 1Y forward for 2023 rates
            self.fwd_rate[m][self.weo_end-5] =  self.fwd_rate[m][self.weo_end-4]
            
    # Define method to get share of short term debt from ECB data and compute ST interest/ ST-LT amortization in t=0
    def _get_ecb(self):
        try:
            ecb_base_year = self.df_ecb_st_shares['year'].iloc[-1] # Last year with ECB data
            ecb_total = float(self.df_ecb_st_shares.loc[(self.df_ecb_st_shares['year'] == ecb_base_year) & (self.df_ecb_st_shares['type'] == 'total_securities'), 'value']) # Total security debt
            ecb_lst = float(self.df_ecb_st_shares.loc[(self.df_ecb_st_shares['year'] == ecb_base_year) & (self.df_ecb_st_shares['type'] == 'long_short_term'), 'value']) # Total security debt        
            ecb_sst = float(self.df_ecb_st_shares.loc[(self.df_ecb_st_shares['year'] == ecb_base_year) & (self.df_ecb_st_shares['type'] == 'short_term'), 'value'])  # short-term security debt (m<1y)
            share_sst_lst =  (ecb_sst + ecb_lst) / ecb_total
            self.share_st = ecb_sst / ecb_total
            self.share_lt = 1 - self.share_st
            ecb_st_rate = float(self.df_ecb_st_rates.loc[self.df_ecb_st_rates['year'] == self.t-1, 'value']) # set t-1 rate for ST bonds equal to realized st rate
        except: # If country is not in ecb data, use average from all countries which is caluclated in the dsa_prepare_data
            share_sst_lst =  0.13429386763844137
            self.share_st = 0.045167253581310865
            self.share_lt = 1 - self.share_st
            ecb_st_rate = self.implicit_rate[self.t-1] # Use avg weo rate if not in dataset (UK)
        # Use ecb data to compute ST-LT amortizations in t and ST interest payment 
        self.amortization_old[self.t] = share_sst_lst * self.debt[self.t-1] # Amortization t replaced by ecb share with res maturity <1y
        self.debt_new_st[self.t-1] = self.share_st * self.debt[self.t-1] # ST interest in t calculated based on new_st debt and the 1Y1Y rate in t-1
        self.fwd_rate[1][self.t-1] = ecb_st_rate

    # Define method to approximate original maturity  
    def _get_alpha(self):
        t = self.weo_end - 5  # Calculate t as the difference between self.weo_end and 5
        self.df_alpha = self.df_term_structure.loc[self.df_term_structure['Event'] == 'Principal', ['year','amount_lc_bn']].reset_index(drop=True)  # Filter self.df_term_structure by 'Principal' event and select 'year' and 'amount_lc_bn' columns
        self.df_alpha = self.df_alpha.rename(columns={'year':'m','amount_lc_bn':'a'})  # Rename 'year' column to 'm' and 'amount_lc_bn' column to 'a'
        self.df_alpha['m'] = self.df_alpha['m'] - t  # Subtract t from 'm' column
        self.df_alpha.loc[self.df_alpha['m'] == 1, 'a'] = self.df_alpha.loc[self.df_alpha['m'] <= 1, 'a'].sum()  # If 'm' is 1, sum all 'a' where 'm' is less than or equal to 1
        self.df_alpha.loc[self.df_alpha['m'] == 30, 'a'] = self.df_alpha.loc[self.df_alpha['m'] >= 30, 'a'].sum()  # If 'm' is 1, sum all 'a' where 'm' is less than or equal to 1        
        self.df_alpha = self.df_alpha.loc[(self.df_alpha['m'] >= 1) & (self.df_alpha['m'] <= 30), ['m', 'a']]  # Filter self.df_alpha by 'm' between 1 and 30 and select 'm', 'a', and 'rho' columns
        self.df_alpha['rho'] = self.df_alpha['a'] / self.df_alpha['a'].sum()
        self.df_alpha = self.df_alpha.set_index('m')  # Set 'm' column as the index of self.df_alpha
        new_index = pd.RangeIndex(1, 30 + 1)  # Create a new range index from the minimum to maximum value of 'm' plus one
        self.df_alpha = self.df_alpha.reindex(new_index).fillna(0).reset_index(names='m')  # Reindex self.df_alpha with the new index and interpolate the missing values using linear interpolation. Reset the index and rename the 'index' column to 'm'
        def func(x, a, b): # Define a function to fit the data
            return 1 / (1 + a * x) + b
        m = np.array(self.df_alpha['m'])
        rho = np.array(self.df_alpha['rho'])
        popt, pcov = curve_fit(func, m, rho)
        self.df_alpha['rho_fit'] = func(m, *popt)
        self.df_alpha['rho_fit'] = self.df_alpha['rho_fit'].clip(lower=0)
        self.df_alpha['alpha'] = self.df_alpha['rho_fit'] - self.df_alpha['rho_fit'].shift(-1)
        self.df_alpha.loc[self.df_alpha.index[-1], 'alpha'] = self.df_alpha.loc[self.df_alpha.index[-1], 'rho_fit']
        self.df_alpha['alpha'] = self.df_alpha['alpha'] / self.df_alpha['alpha'].sum()
        self.alpha = self.df_alpha.set_index('m')['alpha'].to_dict()  # Convert alpha values to dictionary with maturity as keys 
    
    # ------------------------------------------------------------------------------ #
    # ----------------------------- Projection methods ----------------------------- #
    # ------------------------------------------------------------------------------ #
    
    # Call methods that projects forward and saves dataframe 
    def projection(self):   
        self._project_gdp()
        self._project_pb()
        self._calc_debt_nm()
        self._project_debt_ratio()
        #self._save_projection()
        
    # Define method to project ngdp based on real growth and inflation expecatation
    def _project_gdp(self):
        for i,t in enumerate(range(self.weo_end+1, self.T+1)): # Iterate through period after WEO
            if i+1+5 <= 10: # Get corresponding inflation expectation for maturity 5years fwd
                self.pi[t] = float(self.df_infl_expectation.loc[self.df_infl_expectation['maturity'] == i+1+5, 'infl_expectation'])
            elif i+1+5 > 10: # set equal to 10y fwd for future iterations
                self.pi[t] = float(self.df_infl_expectation.loc[self.df_infl_expectation['maturity'] == 10, 'infl_expectation'])    
            self.ng[t] = self.rg + self.g_error + self.pi[t]
            self.ngdp[t] = self.ngdp[t-1] * (1 + self.ng[t] / 100) # Apply inflated g to ngdp and project forward

    # Define method to project ngdp based on real growth and inflation expecatation
    def _project_pb(self):
        for t in range(self.t+1, self.T+1): # Iterate through period after WEO
            if t > self.adjustment_end:
                self.pb_pct_ngdp[t] = self.pb_pct_ngdp[t-1]
            self.pb[t] = self.pb_pct_ngdp[t] / 100 * self.ngdp[t] # Derive primary balance
    
    # Defin method to calculate non-private debt _nm between bond and weo data
    def _calc_debt_nm(self):
        debt_total = self.debt[self.t]
        debt_m = sum(self.amortization_old.values())
        self.debt_nm = debt_total - debt_m
    
    # Define method that projects forward and populates dictionaries
    def _project_debt_ratio(self):
        
        for t in range(self.weo_start, self.t): # Set initial values for backlooking vars
            self.gfn[t] = 0
            self.amortization_new_st[t] = 0 
            for i, m in enumerate(self.interest_new_lt.keys()):
                if i > 0: # don't set m=1 fwd rate to zero since it contains the realized st rate from get_ecb
                    self.fwd_rate[m][t] = 0
                self.interest_new_lt[m][t] = 0
                self.amortization_new_lt[m][t] = 0
                self.debt_new_lt[m][t] = 0
            self.interest_new[t] = 0
            self.amortization_new[t] = 0 
            self.debt_new[t] = 0
            self.debt_old[t] = self.debt[t]
                    
        for t in range(self.t, self.T+1):
                
            # Interest and amortization of new st debt
            self.interest_new_st[t] = self.debt_new_st[t-1] * max(self.fwd_rate[1][t-1] + self.i_error, 0) / 100 # Interest on new st debt based on new stock and fwd rate
            if t == self.t: # in t=0 debt_new_st[t-1] is ecb st stock and not s.t. i_error
                self.interest_new_st[t] = self.debt_new_st[t-1] * self.fwd_rate[1][t-1] / 100
            self.amortization_new_st[t] = self.share_st * self.gfn[t-1] # Amortization is based on gfn at t-maturity 
            
            # Interest and amortization of new lt debt
            for m in self.interest_new_lt.keys():
                self.interest_new_lt[m][t] = sum(self.share_lt * self.alpha[m] * self.gfn[y] * max(self.fwd_rate[m][y] + self.i_error, 0) / 100 for y in range(t-m, t)) # Interest on new debt in each year is based on fwd rates of maturity buckets
                self.amortization_new_lt[m][t] = self.share_lt * self.alpha[m] * self.gfn[t-m] # Amortization is based on gfn at t-maturity
            
            # Aggregate new financing costs
            self.interest_new[t] = self.interest_new_st[t] + sum(self.interest_new_lt[m][t] for m in self.interest_new_lt.keys()) 
            self.amortization_new[t] = self.amortization_new_st[t] + sum(self.amortization_new_lt[m][t] for m in self.amortization_new_lt.keys()) 
            
            # Interest on non-marketable debt
            self.implicit_rate[t] = (self.interest_old[t] + self.interest_new[t]) / self.debt[t-1] * 100 # avg rate is calcluated on private debt only, then applied to debt _nm in future period
            self.interest_nm[t] = self.debt_nm * self.implicit_rate[t] / 100 # We assume that unaccounted debt is rolled over infinitely and faces avg rate
            
            # GFN based on financing costs old/new, debt _nm, and pb
            self.gfn[t] = self.amortization_old[t] + self.interest_old[t] + self.amortization_new[t] + self.interest_new[t] + self.interest_nm[t] - self.pb[t]
            self.gfn_pct_ngdp[t] = self.gfn[t] / self.ngdp[t] * 100

            # Get st/lt debt based on GFN
            self.debt_new_st[t] = self.debt_new_st[t-1] - self.amortization_new_st[t] + self.share_st * self.gfn[t]
            for m in self.debt_new_lt.keys():
                self.debt_new_lt[m][t] = self.debt_new_lt[m][t-1] - self.amortization_new_lt[m][t] + self.share_lt * self.alpha[m] * self.gfn[t]
            self.debt_new[t] = self.debt_new_st[t] + sum(self.debt_new_lt[m][t] for m in self.debt_new_lt.keys()) 
            
            if  self.weo and t < self.weo_end+1: # If we rely on weo debt projection, use them for first 5 years
                self.debt_old[t] = self.debt[t] - self.debt_new[t]
            elif t < self.t+1: # Otherwise still use weo 2023 stock projection
                self.debt_old[t] = self.debt[t] - self.debt_new[t]
            else:
                self.debt_old[t] = np.max([self.debt_old[t-1] - self.amortization_old[t], 0]) # Old debt is old debt last year minus amortization
                self.debt[t] = self.debt_old[t] + self.debt_new[t]
            
            self.debt_ratio[t] =  self.debt[t] / self.ngdp[t] * 100
            self.fb[t] = (self.pb[t] - self.interest_old[t] - self.interest_new[t] - self.interest_nm[t]) / self.ngdp[t] * 100
                          
    # Define method that converts dictioanries to dataframe 
    def save_projection(self):
        data_dict = {'debt_ratio': self.debt_ratio, 
                     'ngdp': self.ngdp, 
                     'debt': self.debt, 
                     'pi': self.pi,
                     'ng': self.ng,
                     'debt_old': self.debt_old, 
                     'debt_new': self.debt_new, 
                     'debt_new_st': self.debt_new_st, 
                     'amortization_old': self.amortization_old, 
                     'amortization_new': self.amortization_new, 
                     'interest_old': self.interest_old, 
                     'interest_new': self.interest_new, 
                     'interest_nm': self.interest_nm,
                     'gfn': self.gfn, 
                     'gfn_pct_ngdp': self.gfn_pct_ngdp, 
                     'implicit_rate': self.implicit_rate,
                     'pb':self.pb,
                     'pb_pct_ngdp': self.pb_pct_ngdp,
                     'fb': self.fb
                    }
        dfs = []
        for key in data_dict:
            df = pd.DataFrame({'year': list(data_dict[key].keys()), key: list(data_dict[key].values())})
            dfs.append(df)

        # Merge the DataFrames on the "year" column with an "outer" join
        df = pd.merge(dfs[0], dfs[1], on='year', how='outer')
        for i in range(2, len(dfs)):
            df = pd.merge(df, dfs[i], on='year', how='outer')
        # Add the ISO column to the DataFrame
        df['iso'] = self.iso
        # Reorder the columns
        self.df_projection = df[['iso', 'year', 'debt_ratio', 'debt', 'ngdp', 'pb', 'pb_pct_ngdp', 'fb', 'gfn', 'gfn_pct_ngdp', 'pi', 'ng', 'debt_old', 'debt_new', 'debt_new_st',
                                 'amortization_old', 'amortization_new', 'interest_old', 'interest_new', 'interest_nm', 'implicit_rate']]
    
    # ------------------------------------------------------------------------------ #
    # ----------------------------- Simulation methods ----------------------------- #
    # ------------------------------------------------------------------------------ #
    
    # Define method that updates interest rates based on errors
    def update_pb(self, value, adjustment_end=2029):
        self.adjustment_end = adjustment_end
        self.pb_pct_ngdp[adjustment_end] = value # Set pb at end of adjustment period
        for t in range(self.t+1, self.adjustment_end): # Delete existing weo entries for new in adjustment period
            if t in self.pb_pct_ngdp.keys(): # Can only delete existing entries
                del self.pb_pct_ngdp[t]
        self.pb_pct_ngdp = pd.Series(self.pb_pct_ngdp).reindex(range(self.weo_start, self.adjustment_end+1)).interpolate().to_dict() # Interpolate adj period values     

    # Define method to optimze for optimal pb path
    def find_pbstar(self, path='0', adjustment_end=2029, bounds=(-5,5), steps=[0.1,0.01,0.001], print_status=True):
        if path == '0': # Set conditions based on input path
            condition_a, condition_b = self._condition_0, None
        elif path == '0.5':
            condition_a, condition_b = self._condition_0_sgp, self._condition_5
        elif path == '1':
            condition_a, condition_b = self._condition_0_sgp, self._condition_10
        elif path == 'ec_new':
            condition_a, condition_b = self._condition_ec_deficit_new, self._condition_ec_debt_new
        elif path == 'ec_old':
            condition_a, condition_b = self._condition_ec_deficit_old, self._condition_ec_debt_old
        elif path == 'ec_0.5':
            condition_a, condition_b = self._condition_ec_5, None
        elif path == 'ec_1':
            condition_a, condition_b = self._condition_ec_10, None

        values = np.arange(bounds[0], bounds[1] , steps[0])  # Initiate a list of pbs to iterate over
        best_values = np.full(len(steps), bounds[0], dtype=float)
        for value in values:
            debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end) # Calculate the debt ratios and their debt_derivatives for the current country and pb
            if condition_a(debt_path, debt_derivatives, pb_path, pb_derivatives): # Check if the current pb satisfies the first condition function
                best_values[0] = value   
                if len(steps) > 1: # iterate downwards if step 2 is defined
                    while condition_a(debt_path, debt_derivatives, pb_path, pb_derivatives) and value >= bounds[0]: # Continue decrementing the `best_value` until it no longer satisfies the first condition function or out of bounds
                        best_values[1] = value
                        value -= steps[1]
                        debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end)    
                if len(steps) > 2: # If third step compute last valid values and decend further in smaller steps
                    value = best_values[1] # Get last valid values
                    debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end) # debt_derivatives for last valud value
                    while condition_a(debt_path, debt_derivatives, pb_path, pb_derivatives) and value >= bounds[0]:
                        best_values[2] = value
                        value -= steps[2]
                        debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end)
                break # Break once smallest step violates condition
            elif condition_b: # If a second condition function is specified, check if the current pb satisfies it
                if condition_b(debt_path, debt_derivatives, pb_path, pb_derivatives): # Check if the current pb satisfies the first condition function
                    best_values[0] = value   
                    if len(steps) > 1: # iterate downwards if step 2 is defined
                        while condition_b(debt_path, debt_derivatives, pb_path, pb_derivatives) and value >= bounds[0]: # Continue decrementing the `best_value` until it no longer satisfies the first condition function
                            best_values[1] = value
                            value -= steps[1]
                            debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end)    
                    if len(steps) > 2: # If third step compute last valid values and decend further in smaller steps
                        value = best_values[1] # Get last valid values
                        debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end) # debt_derivatives for last valud value
                        while condition_b(debt_path, debt_derivatives, pb_path, pb_derivatives) and value >= bounds[0]:
                            best_values[2] = value
                            value -= steps[2]
                            debt_path, debt_derivatives, pb_path, pb_derivatives = self._get_derivatives(value, adjustment_end)
                    break # Break once smallest step violates condition
        if print_status:
            print(f'{self.iso} pb*({path}): {np.min(best_values):.3f}')
        return np.min(best_values)
    
    # Define internal debt path condition functions for optimization of pbs
    def _condition_0(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        return debt_derivatives.iloc[-1] < 0.01 # final derivative is 0, stable path
    
    def _condition_0_sgp(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        return (debt_path.loc[self.weo_end+1:] <= 60).all() and (debt_derivatives.iloc[-1] < 0.01) # Stable path if below 60
    def _condition_5(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        return (debt_derivatives.loc[self.weo_end+1:] <= -0.5).all() # Path from 2029 declining by at least 0.5 annually
    def _condition_10(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        return (debt_derivatives.loc[self.weo_end+1:] <= -1).all() # Path from 2029 declining by at least 1 annually

    # Define conditions for comparisson of new ec proposal
    def _condition_ec_5(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -0.5).all()
        return debt_decrease_lt # ec proposal for d
    def _condition_ec_10(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -1).all()
        return debt_decrease_lt # ec proposal
    
    def _condition_ec_deficit_new(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        deficit_greater_3 = (pb_path.loc[self.t] < -3)
        deficit_decrease = (pb_derivatives.loc[self.t+1:self.adjustment_end] >= 0.5).all()
        debt_decrease_adjustment = (debt_path.loc[self.t+1] > debt_path.loc[self.adjustment_end])
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -0.5).all()
        return deficit_greater_3 and deficit_decrease and debt_decrease_adjustment and debt_decrease_lt # ec proposal for deficit below 3  
    def _condition_ec_debt_new(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        debt_decrease_adjustment = (debt_path.loc[self.t+1] > debt_path.loc[self.adjustment_end])
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -0.5).all()
        return debt_decrease_adjustment and debt_decrease_lt # ec for deficit above 3

    def _condition_ec_deficit_old(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        deficit_greater_3 = (pb_path.loc[self.t] < -3)
        deficit_decrease = (pb_derivatives.loc[self.t+1:self.adjustment_end] >= 0.5).all()
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -0.5).all()
        return deficit_greater_3 and deficit_decrease and debt_decrease_lt # ec proposal for deficit below 3  
    def _condition_ec_debt_old(self, debt_path, debt_derivatives, pb_path, pb_derivatives):
        debt_decrease_lt = (debt_derivatives.loc[self.adjustment_end+1:self.adjustment_end+10] < -0.5).all()
        return debt_decrease_lt # ec for deficit above 3
    
    # Define internal method to calculate debt_derivatives and debt path with new pb
    def _get_derivatives(self, value, adjustment_end):
        self.update_pb(value=value, adjustment_end=adjustment_end) 
        self.projection()     
        debt_path = pd.Series(self.debt_ratio.values(), index=self.debt_ratio.keys()).loc[self.t:]
        debt_derivatives = debt_path - debt_path.shift(1)
        pb_path = pd.Series(self.pb_pct_ngdp.values(), index=self.debt_ratio.keys()).loc[self.t:]
        pb_derivatives = pb_path - pb_path.shift(1) 
        return debt_path, debt_derivatives, pb_path, pb_derivatives
    
    # Define Monte Carlo simulation that calculates pbstars for draws of g and i errors
    def monte_carlo(self, kde_dict, n=500, path='0', bounds=(-5,5), steps=[1, 0.1, 0.01], joint=False):
        if joint:
            kde_xy = kde_dict[self.iso]['kde_xy']
            errors = kde_xy.sample(n_samples=n)
            g_errors = errors[:,0]
            i_errors = errors[:,1]
        else:
            kde_x = kde_dict[self.iso]['kde_x'] # Extract KDEs from KDE dictionary
            kde_y = kde_dict[self.iso]['kde_y']
            g_errors = kde_x.sample(n_samples=n).flatten() # Draw samples from the KDEs
            i_errors = kde_x.sample(n_samples=n).flatten()
        self.pbstar_mc = np.zeros(n) # Initiate pbstar_array for optimization results and loop through sampled errors
        start_time = time.time()
        for i, (g_error, i_error) in enumerate(zip(g_errors,i_errors)):
            self.g_error = g_error
            self.i_error = i_error
            remaining_time = (time.time() - start_time) / (i + 1) * (n - i - 1) 
            remaining_time_min, remaining_time_sec = divmod(int(remaining_time), 60)
            print(f'\rSolving {self.iso} pb*({path}) - Draw: {i+1}/{n} - Estimated time remaining: {remaining_time_min} min {remaining_time_sec} sec', end='')
            try:
                self.pbstar_mc[i] = self.find_pbstar(path=path, bounds=bounds, steps=steps, print_status=False)
            except:
                print(f'\nOptimization failed: {self.iso} pb*({path}), g_error: {g_error:.2f}, i_error: {i_error:.2f}')
                continue
        print('')
        return self.pbstar_mc

