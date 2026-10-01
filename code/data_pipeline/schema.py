# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Input Schema             #
# ========================================================================================= #
#
# Single definition of every input the DSA model reads from the input workbook. The workbook
# writer, the workbook reader and the data pipeline all use this registry, so variable codes,
# descriptions and units are defined in one place only.
#
# Two kinds of inputs exist:
#   - PARAMETERS: one value per country (debt structure, market anchors, elasticities, ...)
#   - SERIES:     one value per country and year (forecasts, ageing costs, repayments, ...)
#
# Author: Lennard Welslau
# ========================================================================================= #

from pathlib import Path

# Repository root and default data locations, resolved relative to this file (not the cwd)
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / 'data' / 'InputData'
RAW_DIR = REPO_ROOT / 'data' / 'RawData'

EU27 = [
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL',
    'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
]

COUNTRY_NAMES = {
    'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia', 'CYP': 'Cyprus',
    'CZE': 'Czechia', 'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France',
    'DEU': 'Germany', 'GRC': 'Greece', 'HUN': 'Hungary', 'IRL': 'Ireland', 'ITA': 'Italy',
    'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta', 'NLD': 'Netherlands',
    'POL': 'Poland', 'PRT': 'Portugal', 'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
    'ESP': 'Spain', 'SWE': 'Sweden',
}

# Parameters: code -> (description, unit)
PARAMETERS = {
    'REFERENCE_YEAR': ('Reference year T of the input data (last year before the adjustment); default model start year', 'year'),
    'LAST_FORECAST_YEAR': ('Last year of the Commission forecast (output gap deviations from the forecast decay until then)', 'year'),
    'EXCESSIVE_DEFICIT_PROCEDURE': ('Country subject to an excessive deficit procedure in T (1 = yes, 0 = no)', 'flag'),
    'FISCAL_MULTIPLIER': ('Fiscal multiplier on changes in the structural primary balance', 'ratio'),
    'FISCAL_MULTIPLIER_PERSISTENT': ('Multiplier effect: 1 = persistent effect relative to baseline output gap, 0 = Commission prior guidance output gap closure rule', 'flag'),
    'BUDGET_BALANCE_ELASTICITY': ('Budget balance semi-elasticity to the output gap', 'ratio'),
    'DEBT_ST_SHARE': ('Share of short-term debt in total debt', 'share'),
    'DEBT_LT_MATURING_SHARE': ('Share of long-term debt maturing in T+1 (starting value)', 'share'),
    'DEBT_LT_MATURING_AVG_SHARE': ('Share of long-term debt maturing each year (T+10 convergence value)', 'share'),
    'DEBT_AVG_RESIDUAL_MATURITY': ('Average residual maturity of debt (stochastic interest rate shocks; optional, default 1 / DEBT_LT_MATURING_AVG_SHARE)', 'years'),
    'DEBT_DOMESTIC_SHARE': ('Share of debt in domestic currency', 'share'),
    'DEBT_EUR_SHARE': ('Share of debt in euro (0 for euro area members, counted as domestic)', 'share'),
    'INTEREST_RATE_ST_T10': ('Short-term market interest rate, T+10 convergence value', '%'),
    'INTEREST_RATE_LT_T10': ('Long-term market interest rate, T+10 convergence value', '%'),
    'INTEREST_RATE_ST_T30': ('Short-term market interest rate, T+30 convergence value', '%'),
    'INTEREST_RATE_LT_T30': ('Long-term market interest rate, T+30 convergence value', '%'),
    'INFLATION_T10': ('GDP deflator inflation, T+10 convergence value', '%'),
    'INFLATION_T30': ('GDP deflator inflation, T+30 convergence value', '%'),
}

# Series: code -> (description, unit, group)
SERIES = {
    # Macro
    'REAL_GDP': ('Real GDP', 'bn national currency', 'Macro'),
    'REAL_GDP_GROWTH': ('Real GDP growth', '%', 'Macro'),
    'POTENTIAL_GDP': ('Potential GDP', 'bn national currency', 'Macro'),
    'POTENTIAL_GDP_GROWTH': ('Potential GDP growth', '%', 'Macro'),
    'NOMINAL_GDP': ('Nominal GDP', 'bn national currency', 'Macro'),
    'NOMINAL_GDP_GROWTH': ('Nominal GDP growth', '%', 'Macro'),
    'GDP_DEFLATOR_PCH': ('GDP deflator inflation', '%', 'Macro'),
    'EA_GDP_DEFLATOR_PCH': ('Euro area GDP deflator inflation (reference only)', '%', 'Macro'),
    # Fiscal
    'PRIMARY_BALANCE': ('Primary balance', '% of GDP', 'Fiscal'),
    'STRUCTURAL_PRIMARY_BALANCE': ('Structural primary balance', '% of GDP', 'Fiscal'),
    'FISCAL_BALANCE': ('Overall fiscal balance', '% of GDP', 'Fiscal'),
    'ONE_OFF_MEASURES': ('One-off and other temporary measures (part of the primary balance, not the SPB)', '% of GDP', 'Fiscal'),
    'PRIMARY_EXPENDITURE_SHARE': ('Primary expenditure', '% of GDP', 'Fiscal'),
    'IMPLICIT_INTEREST_RATE': ('Implicit interest rate on government debt (forecast years; projected thereafter)', '%', 'Fiscal'),
    'IMPLICIT_INTEREST_RATE_ADJ': ('Adjustment added to the projected implicit interest rate (e.g. official loan terms)', 'pp', 'Fiscal'),
    # Debt
    'DEBT_TOTAL': ('Gross government debt', 'bn national currency', 'Debt'),
    'DEBT_RATIO': ('Gross government debt', '% of GDP', 'Debt'),
    'STOCK_FLOW': ('Stock-flow adjustment (T to T+2 forecast)', 'bn national currency', 'Debt'),
    'STOCK_FLOW_RATIO': ('Stock-flow adjustment, exogenous path (overrides STOCK_FLOW where given)', '% of GDP', 'Debt'),
    'DEBT_LT_MATURING_SHARE_PATH': ('Share of long-term debt maturing each year (optional path, overrides interpolation)', 'share', 'Debt'),
    'ESM_REPAYMENT': ('ESM/EFSF loan repayments', 'bn national currency', 'Debt'),
    'BOND_REPAYMENT': ('Repayments of long-term debt outstanding at the start (excl. ESM/EFSF loans; used if bond_data=True, scaled to long-term debt in T)', 'bn national currency', 'Debt'),
    # Markets
    'INTEREST_RATE_ST': ('Short-term market interest rate (3M)', '%', 'Markets'),
    'INTEREST_RATE_LT': ('Long-term market interest rate (10Y)', '%', 'Markets'),
    'EXR_EUR': ('Exchange rate, euro per national currency unit', 'ratio', 'Markets'),
    'EXR_USD': ('Exchange rate, US dollar per national currency unit', 'ratio', 'Markets'),
    # Long-term
    'AGEING_COST': ('Total cost of ageing', '% of GDP', 'Long-term'),
    'TAX_AND_PROPERTY_INCOME': ('Taxes on pensions and property income, change relative to reference year', '% of GDP', 'Long-term'),
}

# Provenance categories used to colour cells in the workbook: code -> (label, fill colour)
PROVENANCE = {
    'api': ('Fetched from public API (AMECO, ECB, Eurostat, ESM)', 'DDEBF7'),
    'commission': ('European Commission prior guidance calculation sheet', 'E7E6E6'),
    'reference': ('Published reference file (Ageing Report, DSM, OGWG)', 'E2EFDA'),
    'assumption': ('Methodological assumption or user input', 'FFF2CC'),
    'derived': ('Derived / interpolated by the data pipeline', 'FFFFFF'),
    'legacy': ('Converted from legacy CSV input file', 'F2F2F2'),
}

OVERRIDE_COLUMNS = ['COUNTRY', 'CODE', 'YEAR', 'VALUE', 'COMMENT']


def latest_input_file():
    """
    Most recent input workbook named dsa_inputs_<yyyy_mm>.xlsx in data/InputData (API builds and converted legacy files;
    Commission and user workbooks are not matched), the model's default input.
    """
    import re
    files = sorted(f.name for f in INPUT_DIR.glob('dsa_inputs_*.xlsx') if re.fullmatch(r'dsa_inputs_\d{4}_\d{2}\.xlsx', f.name))
    if not files:
        raise FileNotFoundError(f'No dsa_inputs_<yyyy_mm>.xlsx in {INPUT_DIR}. Build one with data_pipeline.build.build_inputs().')
    return files[-1]


def resolve_input_path(input_file, must_exist=True):
    """
    Resolve an input workbook path. Bare file names are looked up in data/InputData.
    """
    path = Path(input_file)
    if not path.is_absolute() and not path.exists() and (must_exist or path.parent == Path('.')):
        path = INPUT_DIR / path
    if must_exist and not path.exists():
        raise FileNotFoundError(f'Input file not found: {input_file} (also looked in {INPUT_DIR})')
    return path
