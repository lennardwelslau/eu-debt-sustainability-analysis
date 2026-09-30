# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Base Class               #
# ========================================================================================= #
#
# The DsaModel class projects baseline and deterministic scenario debt paths following the
# methodology of the European Commission's Debt Sustainability Monitor (Annex A3). It has four parts:
#
# 1. Data methods: read the country data from the Excel input workbook (data/InputData, built with
#    the data_pipeline package) and fill methodological anchors (e.g. T+10 and T+30 market rates).
# 2. Projection methods: project growth (incl. fiscal multiplier effects), the primary balance,
#    interest rates and debt dynamics for given adjustment steps and scenarios.
# 3. Optimization methods: find the SPB target that meets a deterministic DSA criterion
#    (find_spb_deterministic).
# 4. Auxiliary methods: results as DataFrames.
#
# The StochasticDsaModel subclass adds stochastic projections and the EU fiscal rules (FiscalRules:
# find_spb_binding, EDP, debt sustainability and deficit resilience safeguards).
#
# For comments and suggestions please contact lennard.welslau[at]gmail[dot]com
#
# Author: Lennard Welslau
# Updated: 2026-09-29
#
# ========================================================================================= #

# Import libraries and modules
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from data_pipeline import read_country, latest_input_file, apply_overrides

# Default input workbook: the most recent 'dsa_inputs_<yyyy_mm>.xlsx' built in 'api' mode (latest public data).
# Use 'dsa_inputs_commission_2024.xlsx' to reproduce the Commission reference trajectories of the 2024 prior guidance,
# or 'dsa_inputs_2025_10.xlsx' to replicate results based on the legacy October 2025 input data.
DEFAULT_INPUT_FILE = None


class DsaModel:

    # ========================================================================================= #
    #                                   INITIALIZE MODEL                                        #
    # ========================================================================================= #

    def __init__(
            self,
            country,  # ISO code of country
            start_year=None,  # start year of projection, first year is baseline value; None uses input file reference year
            end_year=2070,  # end year of projection
            adjustment_period=4,  # number of years for linear spb_bca adjustment
            adjustment_start_year=None,  # start year of linear spb_bca adjustment; None uses start_year + 1
            ageing_cost_period=10,  # number of years for ageing cost adjustment after adjustment period
            fiscal_multiplier=None, # fiscal multiplier for fiscal adjustment, None uses input file value
            fiscal_multiplier_persistence=3, # persistence of fiscal multiplier in years
            fiscal_multiplier_type=None, # 'pers' (persistent) or 'ec' (Commission prior guidance rule), None uses input file
            bond_data=False, # Use bond level data for repayment profile
            input_file=DEFAULT_INPUT_FILE,  # input workbook, file name in data/InputData or full path
            overrides=None,  # user inputs on top of the input workbook: rows with CODE, VALUE and YEAR (None for parameters)
        ):

        # Default input file is the latest API-mode workbook
        if input_file is None:
            input_file = latest_input_file()

        # Default start year is the reference year of the input file, adjustment starts the year after
        if start_year is None:
            start_year = int(read_country(input_file, country)[1]['REFERENCE_YEAR'])
        if adjustment_start_year is None:
            adjustment_start_year = start_year + 1

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
        self.frontloading = True # EDP and deficit resilience steps front-load the adjustment (see find_spb_binding)
        self.scenario = None # scenario parameter
        self.debt_condition = 'declines_or_below_60' # debt criterion of the deterministic scenarios (see find_spb_deterministic)
        self.input_file = input_file  # input workbook
        self.overrides = overrides  # user inputs applied on top of the input workbook

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
        Read the input data and set up baseline arrays for the projection.
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
        self._clean_revenue()

    def _load_input_data(self):
        """
        Load country time series and parameters from the input workbook.
        """
        # Time series indexed by year (reindexed to cover the projection horizon) and parameter dictionary
        series, self.params = read_country(self.input_file, self.country)
        if self.overrides is not None:
            series, self.params = apply_overrides(series, self.params, self.overrides, self.country)
        years = range(min(series.index.min(), self.start_year), max(series.index.max(), self.end_year) + 1)
        self.df_deterministic_data = series.reindex(years)

        # Check if data is available in start year
        available = series.index[series[['DEBT_RATIO', 'NOMINAL_GDP']].notna().all(axis=1)]
        if self.start_year not in available:
            raise ValueError(f"No debt and GDP data for {self.country} in start year {self.start_year}. "
                             f"Available years: {list(available)}")

        # Forecast data are used up to T+2 where available (T+1 for spring forecasts), later years are projected
        self.last_data = max(t for t in range(3) if self.start_year + t in available
                             and all(self.start_year + k in available for k in range(t + 1)))

        # The fiscal balance before the start year is needed for the EDP abrogation rule
        if pd.isna(series['FISCAL_BALANCE'].get(self.start_year - 1, np.nan)):
            raise ValueError(f"No fiscal balance for {self.country} in {self.start_year - 1}, the year before the start "
                             f"year. Input workbooks must include historical data from T-1 (see data_pipeline).")

        # Fiscal multiplier from input file unless specified
        if self.fiscal_multiplier is None:
            self.fiscal_multiplier = self.params['FISCAL_MULTIPLIER']

        # Multiplier type from input file unless specified: persistent effect or Commission output gap closure rule
        if self.fiscal_multiplier_type is None:
            persistent = self.params.get('FISCAL_MULTIPLIER_PERSISTENT', np.nan)
            self.fiscal_multiplier_type = 'ec' if persistent == 0 else 'pers'

        # Last forecast year (T+x), defaults to T+1
        last_forecast_year = self.params.get('LAST_FORECAST_YEAR', np.nan)
        self.last_forecast = 1 if np.isnan(last_forecast_year) else int(last_forecast_year) - self.start_year

    def _input_path(self, var):
        """
        Return input series for the projection years as array, NaN where no value is provided.
        """
        return self.df_deterministic_data.loc[self.start_year:self.end_year, var].to_numpy(dtype=np.float64, copy=True)

    def _set_anchor(self, path, t, value, to_end=False):
        """
        Set methodological anchor value at T+t (and all later years if to_end) where the input file provides no value.
        """
        if t >= len(path):
            return
        idx = slice(t, None) if to_end else slice(t, t + 1)
        segment = path[idx]
        segment[np.isnan(segment)] = value

    def _clean_rgdp_pot(self):
        """
        Clean baseline real potential growth.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):

            # potential growth is based on OGWG up to T+5, long-run estimates from 2033, interpoalted in between
            self.rg_pot_bl[t] = self.df_deterministic_data.loc[y, 'POTENTIAL_GDP_GROWTH']

            # potential GDP up to T+5 are from OGWG, after that projected based on growth rate
            self.rgdp_pot_bl[t] = self.df_deterministic_data.loc[y, 'POTENTIAL_GDP']

            # after T+5, potential GDP is projected based on growth rates form AWG
            if pd.isna(self.df_deterministic_data.loc[y, 'POTENTIAL_GDP']):
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
            self.rgdp_bl[t] = self.df_deterministic_data.loc[y, 'REAL_GDP']
            self.rg_bl[t] = self.df_deterministic_data.loc[y, 'REAL_GDP_GROWTH']

            # if real GDP is missing not available from forecast, set it to potential GDP
            if pd.isna(self.df_deterministic_data.loc[y, 'REAL_GDP']):
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
        # Up to T+2 from Commission forecast GDP deflator (any further years provided in the input file are used as well)
        self.pi = self._input_path('GDP_DEFLATOR_PCH')

        # T+10 value based on market expectations (inflation swaps), T+30 value is the inflation target
        # Country specifics (e.g. higher inflation targets in POL, ROU, HUN) are set in the input file
        self._set_anchor(self.pi, 10, self.params['INFLATION_T10'])
        self._set_anchor(self.pi, 30, self.params['INFLATION_T30'])

        # Interpolate missing values
        x = np.arange(len(self.pi))
        mask_pi = np.isnan(self.pi)
        self.pi[mask_pi] = np.interp(x[mask_pi], x[~mask_pi], self.pi[~mask_pi])

    def _clean_ngdp(self):
        """
        Clean baseline nominal growth.
        """
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):

            # Forecast nominal GDP where available, after that projected based on real growth and inflation
            if t <= self.last_data:
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
        # Baseline debt from the forecast
        for t, y in enumerate(range(self.start_year, self.start_year + self.last_data + 1)):
            self.d[t] = self.df_deterministic_data.loc[y, 'DEBT_RATIO']
            self.D[t] = self.df_deterministic_data.loc[y, 'DEBT_TOTAL']

        # Set maturity shares and average maturity
        self.D_share_st = self.params['DEBT_ST_SHARE']
        self.D_share_lt = 1 - self.D_share_st
        self.D_share_lt_maturing_T = self.params['DEBT_LT_MATURING_SHARE']
        self.D_share_lt_mat_avg = self.params['DEBT_LT_MATURING_AVG_SHARE']
        self.avg_res_mat = np.min([round((1 / self.D_share_lt_mat_avg)), 30])

        # Set share of domestic, euro and usd debt (euro share is zero for euro area members, set in input file)
        self.D_share_domestic = np.round(self.params['DEBT_DOMESTIC_SHARE'], 4)
        self.D_share_eur = np.round(self.params['DEBT_EUR_SHARE'], 4)
        self.D_share_usd = np.round(1 - self.D_share_domestic - self.D_share_eur, 4)

        # Set initial values for long and short-term debt
        self.D_st[0] = self.D_share_st * self.D[0]
        self.D_lt[0] = self.D_share_lt * self.D[0]

    def _clean_esm_repayment(self):
        """
        Clean institutional debt data.
        """
        # Set to zero if missing
        self.df_deterministic_data['ESM_REPAYMENT'] = self.df_deterministic_data['ESM_REPAYMENT'].fillna(0)

        # Initial stock of institutional debt at end of start year is the sum of all later repayments
        self.D_lt_esm[0] = self.df_deterministic_data.loc[self.start_year + 1:, 'ESM_REPAYMENT'].sum()

        # Import ESM institutional debt repayments, repayment_lt_esm[t] is the repayment in year start_year + t
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            if t == 0:
                continue
            self.repayment_lt_esm[t] = self.df_deterministic_data.loc[y, 'ESM_REPAYMENT']
            self.D_lt_esm[t] = self.D_lt_esm[t - 1] - self.repayment_lt_esm[t]

    def _clean_debt_redemption(self):
        """
        Clean debt redemption data for institutional debt.
        """
        # Path from input file where given, otherwise T value converging to historical average by T+10
        self.D_share_lt_maturing = self._input_path('DEBT_LT_MATURING_SHARE_PATH')
        self._set_anchor(self.D_share_lt_maturing, 0, self.D_share_lt_maturing_T)
        self._set_anchor(self.D_share_lt_maturing, 10, self.D_share_lt_mat_avg, to_end=True)

        # Interpolate missing values
        x = np.arange(len(self.D_share_lt_maturing))
        mask = np.isnan(self.D_share_lt_maturing)
        self.D_share_lt_maturing[mask] = np.interp(x[mask], x[~mask], self.D_share_lt_maturing[~mask])

    def _clean_bond_repayment(self):
        """
        Clean long-term bond repayment data.
        """
        # Import bond repayment data
        if self.df_deterministic_data.loc[self.start_year + 1:, 'BOND_REPAYMENT'].isna().all():
            raise ValueError(f'bond_data=True requires BOND_REPAYMENT data for {self.country} in {self.input_file}')
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            self.repayment_lt_bond[t] = self.df_deterministic_data.loc[y, 'BOND_REPAYMENT']

    def _clean_pb(self):
        """
        Clean structural primary balance.
        """
        # Baseline SPB and balances from the forecast, SPB and primary balance constant thereafter
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            if t <= self.last_data:
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

        # Interest expenditure in the start year (primary minus overall balance), used by the Commission EDP rule
        self.interest_ratio[0] = self.pb[0] - self.ob[0]

        # Set initial values to baseline
        self.spb_bca = np.copy(self.spb_bl)
        self.spb = np.copy(self.spb_bl)

        # One-off and temporary measures, part of the primary balance but not of the structural primary balance
        self.one_off = np.nan_to_num(self._input_path('ONE_OFF_MEASURES'))

        # Get budget balance semi-elasticity
        self.budget_balance_elasticity = self.params['BUDGET_BALANCE_ELASTICITY']

        # Get primary expenditure share in start year
        self.expenditure_share = self.df_deterministic_data.loc[self.start_year, 'PRIMARY_EXPENDITURE_SHARE']

    def _clean_implicit_interest_rate(self):
        """
        Clean implicit interest rate.
        """
        # Implicit interest rate from the forecast, projected where not provided
        for t, y in enumerate(range(self.start_year, self.start_year + self.last_data + 1)):
            self.iir_bl[t] = self.df_deterministic_data.loc[y, 'IMPLICIT_INTEREST_RATE']

        # Exogenous adjustment to the projected implicit interest rate (zero where not provided)
        self.iir_adjustment = np.nan_to_num(self._input_path('IMPLICIT_INTEREST_RATE_ADJ'))

        # Set initial values to baseline
        self.iir = np.copy(self.iir_bl)

        # Initial lt baseline
        self.iir_lt[0] = self.iir[0] * (1 - self.D_share_st)

    def _clean_market_rates(self):
        """
        Clean benchmark and forward market rates. Interpolate missing values.
        """
        # Benchmark rates for the first years (any further years provided in the input file are used as well)
        self.i_st_bl = self._input_path('INTEREST_RATE_ST')
        self.i_lt_bl = self._input_path('INTEREST_RATE_LT')

        # T+10 values from market forward rates
        self.fwd_rate_st = self.params['INTEREST_RATE_ST_T10']
        self.fwd_rate_lt = self.params['INTEREST_RATE_LT_T10']
        self._set_anchor(self.i_st_bl, 10, self.fwd_rate_st)
        self._set_anchor(self.i_lt_bl, 10, self.fwd_rate_lt)

        # T+30 values and beyond: long-run convergence values (inflation target + 2, short-term rate at half of long-term rate)
        self._set_anchor(self.i_lt_bl, 30, self.params['INTEREST_RATE_LT_T30'], to_end=True)
        self._set_anchor(self.i_st_bl, 30, self.params['INTEREST_RATE_ST_T30'], to_end=True)

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
        # Stock-flow adjustment from the forecast (zero afterwards unless an exogenous path is given)
        for t, y in enumerate(range(self.start_year, self.start_year + self.last_data + 1)):
            self.SF[t] = self.df_deterministic_data.loc[y, 'STOCK_FLOW']

        # Exogenous stock-flow path in % of GDP (country-specific, e.g. pension fund balances), NaN where not given
        self.sf_exogenous = self._input_path('STOCK_FLOW_RATIO')

    def _clean_exchange_rate(self):
        """
        Clean exchange rate data for non-euro countries.
        """
        # Exchange rates from the forecast, constant thereafter
        for t, y in enumerate(range(self.start_year, self.end_year + 1)):
            if t <= self.last_data:
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

    def _clean_revenue(self):
        """
        Clean property income and pension revenue data (changes relative to the reference year of the source).
        """
        self.revenue = np.nan_to_num(self._input_path('TAX_AND_PROPERTY_INCOME'))

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
        self._project_adjustment_path()
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
        # Copy step inputs as float arrays so that the caller's arrays are not modified in place
        spb_steps, edp_steps, deficit_resilience_steps, post_spb_steps = [
            None if x is None else np.array(x, dtype=np.float64)
            for x in (spb_steps, edp_steps, deficit_resilience_steps, post_spb_steps)
        ]

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

    def _project_adjustment_path(self):
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
        if self.frontloading:  # later steps reduced to keep the SPB target (front-loading), otherwise added on top
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
        if self.frontloading:  # later steps reduced to keep the SPB target (front-loading), otherwise added on top
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
        if self.fiscal_multiplier_type == 'ec':
            self._calc_rgdp_ec()
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
            self.fiscal_multiplier_effect[t] = (self.fiscal_multiplier
                                                * ((self.spb_bca[t] - self.spb_bca[t - 1])
                                                   - (self.spb_bl[t] - self.spb_bl[t - 1]))
                                                )

            # Add spillover effect to fiscal_multiplier effect if defined
            if hasattr(self, 'fiscal_multiplier_spillover'):
                self.fiscal_multiplier_effect[t] += self.fiscal_multiplier_spillover[t]

            # Calculate persistence term of multiplier effect
            persistence_term = sum([self.fiscal_multiplier_effect[t - i]
                                    * (self.fiscal_multiplier_persistence - i)
                                    / self.fiscal_multiplier_persistence
                                    for i in range(1, self.fiscal_multiplier_persistence)]
                                    )

            # Fiscal multiplier effect on output gap
            self.output_gap[t] = self.output_gap_bl[t] - self.fiscal_multiplier_effect[t] - persistence_term

            # Real growth and real GDP
            self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]
            self.rg[t] = (self.rgdp[t] - self.rgdp[t - 1]) / self.rgdp[t - 1] * 100

    def _calc_rgdp_ec(self):
        """
        Calculates real GDP and real growth following the Commission methodology. In the year of an SPB change,
        the fiscal multiplier effect lowers growth relative to baseline. The resulting output gap closes with the
        2/3 and 1/3 rule over the following years (fiscal_multiplier_persistence = 3), during which new SPB changes
        add further effects. Without SPB changes, growth follows the baseline. In forecast years, only the deviation
        from the forecast output gap decays.
        """
        P = self.fiscal_multiplier_persistence
        self.multiplier_counter = np.zeros(self.projection_period)
        counter = self.multiplier_counter
        for t in range(1, self.projection_period):
            # Fiscal multiplier effect from change in SPB relative to baseline
            spb_change = self.spb_bca[t] - self.spb_bca[t - 1]
            spb_change_bl = self.spb_bl[t] - self.spb_bl[t - 1]
            self.fiscal_multiplier_effect[t] = self.fiscal_multiplier * (spb_change - spb_change_bl)

            # Add spillover effect to fiscal_multiplier effect if defined
            if hasattr(self, 'fiscal_multiplier_spillover'):
                self.fiscal_multiplier_effect[t] += self.fiscal_multiplier_spillover[t]

            # Counter: P + 1 in years with an SPB change, counting down to zero thereafter
            active = abs(spb_change - spb_change_bl) > 1e-4 and abs(spb_change) > 1e-4
            if hasattr(self, 'fiscal_multiplier_spillover') and abs(self.fiscal_multiplier_spillover[t]) > 1e-8:
                active = True
            counter[t] = P + 1 if active else max(counter[t - 1] - 1, 0)

            # Output gap closes gradually after an SPB change (2/3, 1/3 rule for P = 3)
            if counter[t] > 1 and counter[t - 1] > 1:
                j = int(P + 1 - counter[t - 1])  # years since the last SPB change before t - 1
                weight = (P - 1 - j) / P
                if t <= self.last_forecast:
                    gap = self.output_gap_bl[t] + weight * (self.output_gap[t - 1 - j] - self.output_gap_bl[t - 1 - j])
                else:
                    gap = weight * self.output_gap[t - 1 - j]
                self.output_gap[t] = gap - self.fiscal_multiplier_effect[t]
                self.rgdp[t] = (self.output_gap[t] / 100 + 1) * self.rgdp_pot[t]

            # Output gap closed at the end of the closing period
            elif counter[t] == 1:
                self.rgdp[t] = self.rgdp_pot[t]

            # Otherwise baseline growth net of the multiplier effect
            else:
                self.rgdp[t] = self.rgdp[t - 1] * (1 + (self.rg_bl[t] - self.fiscal_multiplier_effect[t]) / 100)

            # Real growth and output gap
            self.output_gap[t] = (self.rgdp[t] / self.rgdp_pot[t] - 1) * 100
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
        # From adjustment start, nominal growth based on real growth and inflation
        for t in range(self.adjustment_start, self.projection_period):
            self.ng[t] = (1 + self.rg[t] / 100) * (1 + self.pi[t] / 100) * 100 - 100
            self.ngdp[t] = self.ngdp[t - 1] * (1 + self.ng[t] / 100)

    def _project_stock_flow(self):
        """
        Calculate stock-flow adjustment as share of NGDP.
        Country-specific exceptions (see DSM 2023, e.g. pension fund balances in Finland and Luxembourg,
        or Greek programme-related flows) are provided as exogenous path STOCK_FLOW_RATIO in the input file.
        """
        for t in range(self.projection_period):

            # Where an exogenous path in % of GDP is given, it is used directly
            if not np.isnan(self.sf_exogenous[t]):
                self.sf[t] = self.sf_exogenous[t]

            # Otherwise stock flow is based on the forecast (levels, zero thereafter)
            else:
                self.sf[t] = self.SF[t] / self.ngdp[t] * 100

            # Stock-flow adjustment in levels
            self.SF[t] = self.sf[t] / 100 * self.ngdp[t]

    def _project_spb(self):
        """
        Project structural primary balance
        """
        for t in range(1, self.projection_period):
            # Ageing costs affect the SPB for duration of "ageing_cost_period" if there is policy change
            if ((t > self.adjustment_end and t <= self.adjustment_end + self.ageing_cost_period)
                and self.policy_change):
                self.ageing_component[t] = self.ageing_cost[t] - self.ageing_cost[self.adjustment_end]
                self.revenue_component[t] = self.revenue[t] - self.revenue[self.adjustment_end]

            # After the ageing cost period, the ageing component is kept constant
            elif (t > self.adjustment_end + self.ageing_cost_period
                  and self.policy_change):
                self.ageing_component[t] = self.ageing_component[t-1]
                self.revenue_component[t] = self.revenue_component[t-1]

            # In a no fiscal policy change scenario, spb is kept constant
            elif not self.policy_change:
                self.ageing_component[t] = 0
                self.revenue_component[t] = 0

            # SPB: SPB before ageing costs, net of the change in ageing costs and property income
            self.spb[t] = self.spb_bca[t] - self.ageing_component[t] + self.revenue_component[t]

            # Total SPB for calculation of the structural balance
            self.SPB[t] = self.spb[t] / 100 * self.ngdp[t]

            # Net expenditure growth: potential growth plus inflation minus the SPB change relative to expenditure (EC formula)
            self.net_expenditure_growth[t] = self.rg_pot[t] + self.pi[t] - (self.spb_bca[t] - self.spb_bca[t - 1]) / self.expenditure_share * 100

    def _project_pb_from_spb(self):
        """
        Project primary balance as sum of SPB, cyclical component and one-off measures.
        """
        for t in range(self.projection_period):

            # Calculate components
            self.cyclical_component[t] = self.budget_balance_elasticity * self.output_gap[t]

            # Calculate primary balance ratio as sum of components and total primary balance
            self.pb[t] = self.spb[t] + self.cyclical_component[t] + self.one_off[t]
            self.PB[t] = self.pb[t] / 100 * self.ngdp[t]

    def _project_debt_ratio(self):
        """
        Main loop for debt dynamics
        """
        for t in range(1, self.projection_period):

            # Apply financial stress scenario if specified
            if self.scenario == 'financial_stress' and t == self.adjustment_end + 1:
                self._apply_financial_stress(t)

            # Implicit interest rate, interest, repayments, gross financing needs, debt stock, balances and debt ratio
            self._calc_iir(t)
            self._calc_interest(t)
            self._calc_repayment(t)
            self._calc_gfn(t)
            self._calc_debt_stock(t)
            # Balances and debt ratio before the adjustment start are forecast data where available
            if t >= self.adjustment_start or t > self.last_data:
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

        # Use forecast implicit interest rate where available and derive iir_lt
        if t <= self.last_data and not np.isnan(self.iir_bl[t]):
            self.iir_lt[t] = (self.iir[t] - self.alpha[t - 1] * self.i_st[t]) / (1 - self.alpha[t - 1])
            self.iir[t] = self.iir_bl[t]

        # Use DSM 2023 Annex A3 formulation after, plus exogenous adjustment if provided
        else:
            self.iir_lt[t] = self.beta[t - 1] * self.i_lt[t] + (1 - self.beta[t - 1]) * self.iir_lt[t - 1]
            self.iir[t] = self.alpha[t - 1] * self.i_st[t] + (1 - self.alpha[t - 1]) * self.iir_lt[t] + self.iir_adjustment[t]

        # Replace all 10 < iir < 0 with previous period value to avoid implausible values
        for iir in [self.iir, self.iir_lt]:
            if iir[t] < 0 or iir[t] > 10 or np.isnan(iir[t]):
                iir[t] = iir[t - 1]

    def _calc_interest(self, t):
        """
        Calculate interest payments on newly issued debt
        """
        # Total interest is t-1 debt times the implicit interest rate, consistent with the debt ratio equation
        self.interest[t] = self.iir[t] / 100 * self.D[t - 1]
        self.interest_st[t] = self.D_st[t - 1] * self.i_st[t] / 100  # interest on short-term debt issued in t-1
        self.interest_lt[t] = self.interest[t] - self.interest_st[t]  # interest on long-term debt
        self.interest_ratio[t] = self.interest[t] / self.ngdp[t] * 100

    def _calc_repayment(self, t):
        """
        Calculate repayment of newly issued debt
        """
        self.repayment_st[t] = self.D_st[t - 1]  # repayment payments on short-term debt share in last years gross financing needs

        # With bond data, repayment of new issuance is added to the repayment of existing bonds (_clean_bond_repayment)
        if self.bond_data:
            self.repayment_lt[t] = np.sum(self.D_new_lt[np.max([0, t - 20]) : t] / 20) # Average maturity of new issuance is 10 years, spread evenly over 20 years

        # If bond data is false, repayment share is a function of last periods market debt stock (excluding ESM/EFSF loans)
        else:
            self.repayment_lt[t] = self.D_share_lt_maturing[t] * np.max([self.D_lt[t - 1] - self.D_lt_esm[t - 1], 0])

        # Calculate total repayment
        self.repayment[t] = self.repayment_st[t] + self.repayment_lt[t] + self.repayment_lt_bond[t] + self.repayment_lt_esm[t]

    def _calc_gfn(self, t):
        """
        Calculate gross financing needs
        """
        self.GFN[t] = self.interest[t] + self.repayment[t] - self.PB[t] + self.SF[t]

    def _calc_debt_stock(self, t):
        """
        Calculate new debt stock and distribution of new short and long-term issuance
        """
        # Total debt stock is equal to last period stock minus repayment plus financing needs
        self.D[t] = np.max([self.D[t - 1] - self.repayment[t] + self.GFN[t], 1e-8])  # floor to avoid division by zero

        # Distribution of short-term and long-term debt in financing needs
        D_theoretical_issuance_st = self.D_share_st * self.D[t]  # st debt to keep share equal to D_share_st
        D_theoretical_issuance_lt = np.max([(1 - self.D_share_st) * self.D[t] - (self.D_lt[t - 1] - self.repayment_lt[t] - self.repayment_lt_bond[t] - self.repayment_lt_esm[t]), 1e-8]) # lt debt to keep share equal to 1 - D_share_st, non-negative
        D_issuance_share_st = D_theoretical_issuance_st / (D_theoretical_issuance_st + D_theoretical_issuance_lt)  # share of st in gfn

        # Calculate short-term and long-term debt issuance
        self.D_st[t] = np.max([D_issuance_share_st * self.GFN[t], 1e-8])
        self.D_new_lt[t] = np.max([(1 - D_issuance_share_st) * self.GFN[t], 1e-8])
        self.D_lt[t] = np.max([self.D_lt[t - 1] - self.repayment_lt[t] - self.repayment_lt_bond[t] - self.repayment_lt_esm[t] + self.D_new_lt[t] , 1e-8])

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
            - self.pb[t] + self.sf[t], 1e-8
        ])

    # ========================================================================================= #
    #                               OPTIMIZATION METHODS                                        #
    # ========================================================================================= #

    def find_spb_deterministic(self, criterion, bounds=(-10, 10), tol=0.0001, debt_condition='declines_or_below_60'):
        """
        Find the smallest SPB target at the end of the adjustment period (linear adjustment) that meets a
        deterministic criterion. For the debt scenarios, debt_condition sets the criterion over the 10 years after
        the adjustment period: 'declines_or_below_60' (default), 'declines' or 'below_60'.
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
        assert debt_condition in ['declines_or_below_60', 'declines', 'below_60'], 'Unknown debt condition'
        self.debt_condition = debt_condition

        # Set scenario parameter
        if criterion in [None, 'main_adjustment', 'debt_safeguard']:
            self.scenario = 'main_adjustment'
        else:
            self.scenario = criterion

        # The debt safeguard criterion is part of the fiscal rules (FiscalRules, available in StochasticDsaModel)
        if criterion == 'debt_safeguard' and not hasattr(self, '_debt_safeguard_criterion'):
            raise NotImplementedError('The debt safeguard is part of FiscalRules: use StochasticDsaModel')

        # Precalculate EDP for the debt safeguard if not done yet
        if criterion == 'debt_safeguard' and not hasattr(self, 'edp_period'):
            print('Precalculating EDP steps for debt safeguard')
            self.find_edp()
        elif not hasattr(self, 'edp_steps'):
            self.edp_steps = None

        # Run deterministic optimization
        return self._deterministic_optimization(criterion=criterion, bounds=bounds, tol=tol)

    def _deterministic_optimization(self, criterion, bounds, tol):
        """
        Main loop of optimizer using a bisection method.
        Finds the smallest spb_target that satisfies the given criterion.
        """
        low, high = bounds[0], bounds[1]

        # Check lower bound. If the criterion already holds there (e.g. the debt safeguard if the EDP lasts until the
        # end of the adjustment period), the lower bound is returned
        self._get_spb_steps(criterion=criterion, spb_target=low)
        self.project(
            edp_steps=self.edp_steps,
            spb_steps=self.spb_steps,
            scenario=self.scenario
        )
        if self._deterministic_condition(criterion=criterion):
            self.spb_target = low
            return self.spb_bca[self.adjustment_end]

        # Check upper bound
        self._get_spb_steps(criterion=criterion, spb_target=high)
        self.project(
            edp_steps=self.edp_steps,
            spb_steps=self.spb_steps,
            scenario=self.scenario
        )

        if not self._deterministic_condition(criterion=criterion):
            raise ValueError(f'Deterministic criterion {criterion} not satisfied at upper bound ({high})')

        # Initialize result with the satisfying upper bound
        result_spb_target = high

        # Bisection loop
        while (high - low) > tol:
            mid = (low + high) / 2
            self._get_spb_steps(criterion=criterion, spb_target=mid)
            self.project(
                edp_steps=self.edp_steps,
                spb_steps=self.spb_steps,
                scenario=self.scenario
            )
            if self._deterministic_condition(criterion=criterion):
                result_spb_target = mid
                high = mid
            else:
                low = mid

        # Set final spb_target and return the relevant value
        self.spb_target = result_spb_target

        self._get_spb_steps(criterion=criterion, spb_target=self.spb_target)
        self.project(
            edp_steps=self.edp_steps,
            spb_steps=self.spb_steps,
            scenario=self.scenario
        ) # Project with the final spb_target to ensure self.spb_bca is updated

        return self.spb_bca[self.adjustment_end]

    def _get_spb_steps(self, criterion, spb_target):
        """
        Linear adjustment steps to reach spb_target. EDP and deficit resilience minimum steps are applied in project().
        """
        # If adjustment steps are predefined, keep them and adjust the remaining steps linearly
        if hasattr(self, 'predefined_spb_steps'):
            num_predefined_steps = len(self.predefined_spb_steps)
            self.spb_steps = np.full(self.adjustment_period, np.nan, dtype=np.float64)
            self.spb_steps[:num_predefined_steps] = self.predefined_spb_steps
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
            if self.debt_condition == 'declines':
                return self._debt_declines()
            if self.debt_condition == 'below_60':
                return self._debt_below_60()
            return self._debt_declines() or self._debt_below_60()
        elif criterion == 'deficit_reduction':
            return self._deficit_below_3()
        elif criterion == 'debt_safeguard':
            return self._debt_safeguard_criterion()
        else:
            return False

    def _debt_declines(self):
        """
        Debt ratio declines in each of the 10 years after the adjustment period (or is zero).
        """
        e = self.adjustment_end
        return np.all((np.diff(self.d[e:e + 11]) < 0) | (self.d[e + 1:e + 11] < 1e-3))

    def _debt_below_60(self):
        """
        Debt ratio is at or below 60% of GDP 10 years after the adjustment period.
        """
        return self.d[self.adjustment_end + 10] <= 60

    def _deficit_below_3(self):
        """
        Deficit is at or below 3% of GDP in the 10 years after the adjustment period.
        """
        return np.all(self.ob[self.adjustment_end:self.adjustment_end + 11] >= -3)

    def project_fr(self, coefs, smooth_period=1):
        """
        Project the model with a fiscal reaction function, given reaction coefficients.
        FR function can be linear, quadratic or cubic. First coef is the intercept.
        """
        # Extract intercept and fr coefficients
        fr_coefs = np.zeros(4)
        fr_coefs[:len(coefs)] = coefs

        # Define fiscal reaction function
        def fr_func(t):
            spb = (
                fr_coefs[0]
                + fr_coefs[1] * self.d[t-1]
                + fr_coefs[2] * self.d[t-1]**2
                + fr_coefs[3] * self.d[t-1]**3
                )
            return spb

        # Project with fiscal reaction function to calculate initial smooth step guess
        self.project()
        initial_step = fr_func(self.adjustment_start) - self.spb_bca[self.adjustment_start]
        smooth_step_guess = initial_step / smooth_period

        # Adjust the initial steps until they match the fr at the end of smoothing
        if smooth_period > 1:
            step_diff = 10 # arbitrary value to start
            while abs(step_diff) > 1e-3:
                self.spb_steps[:smooth_period] = smooth_step_guess
                self.project(spb_steps=self.spb_steps)
                actual_step = fr_func(self.adjustment_start + smooth_period) - self.spb_bca[self.adjustment_start]
                step_diff = actual_step - smooth_step_guess * smooth_period
                smooth_step_guess += step_diff / smooth_period

        # Project with fiscal reaction function after smooth period
        for i, t in enumerate(
            range(self.adjustment_start + smooth_period, self.adjustment_end + 1),
            start=smooth_period-1
            ):
            spb = fr_func(t)
            self.spb_steps[i] = spb - self.spb_bca[t]
            self.project(spb_steps=self.spb_steps)

    # ========================================================================================= #
    #                                   AUXILIARY METHODS                                       #
    # ========================================================================================= #

    def key_results(self, variables=None):
        """
        Labelled table of key variables by year for the current projection (see ResultsTables.KEY_VARIABLES).
        """
        from classes.ResultsTables import key_variables
        return key_variables(self.df(all=True), variables)

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
