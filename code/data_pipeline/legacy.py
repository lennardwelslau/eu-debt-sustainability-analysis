# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Legacy Converter         #
# ========================================================================================= #
#
# Converts the legacy long-format CSV input files (deterministic_data_YYYY_MM.csv, with scalar
# parameters stored in a YEAR=0 row) into the Excel input workbook.
#
# Country-specific assumptions that were previously hard-coded in the model are written into
# the workbook instead, reproducing the old behaviour exactly:
#   - T+30 inflation and interest rate anchors (POL/ROU: 2.5 / 4.5, HUN: 3 / 5, others: 2 / 4)
#   - T+10 inflation premium for POL/ROU (half of the T+2 gap to euro area GDP deflator inflation)
#   - EUR debt share set to zero for euro area members
#   - Exogenous stock-flow paths for Finland, Greece (2024 plans) and Luxembourg (pension balance)
#
# Author: Lennard Welslau
# ========================================================================================= #

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from .schema import EU27, SERIES, INPUT_DIR
from .workbook import write_workbook

# Countries for which the legacy model used a non-zero euro debt share
NON_EURO_EUR_SHARE = ['BGR', 'CZE', 'DNK', 'HUN', 'POL', 'ROU', 'SWE']

# Previously hard-coded stock-flow paths (% of GDP)
LEGACY_SF_FIN = {  # Finland, based on the 2024 national medium-term fiscal-structural plan
    2024: 2.5, 2025: 1.3, 2026: 1.7, 2027: 1.4, 2028: 1.4, 2029: 1.0, 2030: 1.1, 2031: 1.5, 2032: 1.5,
    2033: 1.4, 2034: 1.2, 2035: 1.1, 2036: 0.9, 2037: 0.8, 2038: 0.6, 2039: 0.5, 2040: 0.3, 2041: 0.2
}
LEGACY_SF_GRC = {  # Greece, based on Commission numbers for the 2024 plan
    2024: -1.1, 2025: 1.5, 2026: -0.8, 2027: -0.9, 2028: -1.0, 2029: -1.0, 2030: -1.0, 2031: -1.1, 2032: -1.1,
    2033: 0.2, 2034: 0.2, 2035: 0.2, 2036: 0.2, 2037: 0.2, 2038: 0.2, 2039: 0.2, 2040: 0.2, 2041: 0.2
}

PARAM_MAP = {  # legacy column -> new parameter code
    'DEBT_ST_SHARE': 'DEBT_ST_SHARE',
    'DEBT_LT_MATURING_SHARE': 'DEBT_LT_MATURING_SHARE',
    'DEBT_LT_MATURING_AVG_SHARE': 'DEBT_LT_MATURING_AVG_SHARE',
    'DEBT_DOMESTIC_SHARE': 'DEBT_DOMESTIC_SHARE',
    'DEBT_EUR_SHARE': 'DEBT_EUR_SHARE',
    'FWD_RATE_3M10Y': 'INTEREST_RATE_ST_T10',
    'FWD_RATE_10Y10Y': 'INTEREST_RATE_LT_T10',
    'FWD_INFL_5Y5Y': 'INFLATION_T10',
    'BUDGET_BALANCE_ELASTICITY': 'BUDGET_BALANCE_ELASTICITY',
}

LEGACY_SOURCES = {
    'DEBT_TOTAL': 'AMECO (UDGG)', 'DEBT_RATIO': 'AMECO (UDGG)', 'NOMINAL_GDP': 'AMECO (UVGD)',
    'NOMINAL_GDP_GROWTH': 'AMECO (UVGD)', 'GDP_DEFLATOR_PCH': 'AMECO (PVGD)', 'EA_GDP_DEFLATOR_PCH': 'AMECO (PVGD, EA20)',
    'PRIMARY_BALANCE': 'AMECO (UBLGIE)', 'STRUCTURAL_PRIMARY_BALANCE': 'AMECO (UBLGBPS)', 'FISCAL_BALANCE': 'AMECO (UBLGE)',
    'STOCK_FLOW': 'AMECO (UDGGS)', 'PRIMARY_EXPENDITURE_SHARE': 'AMECO (UUTGI)', 'IMPLICIT_INTEREST_RATE': 'AMECO (AYIGD)',
    'REAL_GDP': 'Output Gaps Working Group', 'REAL_GDP_GROWTH': 'Output Gaps Working Group',
    'POTENTIAL_GDP': 'Output Gaps Working Group',
    'POTENTIAL_GDP_GROWTH': 'OGWG to T+5, Ageing Report 2024 from T+12, linearly interpolated in between',
    'AGEING_COST': 'Ageing Report 2024', 'TAX_AND_PROPERTY_INCOME': 'Debt Sustainability Monitor country annex, extrapolated',
    'INTEREST_RATE_ST': 'ECB (GFS 3M benchmark rate)', 'INTEREST_RATE_LT': 'ECB (IRS 10Y benchmark rate)',
    'EXR_EUR': 'AMECO (XNE)', 'EXR_USD': 'AMECO (XNE)', 'ESM_REPAYMENT': 'ESM repayment database',
    'BOND_REPAYMENT': 'Eikon bond-level data', 'STOCK_FLOW_RATIO': 'Country-specific assumption (see model documentation)',
}


def convert_legacy_csv(csv_file, out_file=None, countries=EU27, vintage=None, history_guidance='2024', shock_files=None):
    """
    Convert a legacy deterministic_data CSV to an input workbook.

    Parameters:
        csv_file (str): File name in data/InputData or full path.
        out_file (str): Output workbook path. Defaults to data/InputData/dsa_inputs_<vintage>.xlsx.
        countries (list): ISO3 codes to include.
        vintage (str): Vintage label, defaults to the CSV file suffix (e.g. '2025_10').
        history_guidance (str): Commission prior guidance vintage used for the fiscal balance in T-2 and T-1 and the EDP
            status in T, which legacy files do not contain. None skips both.
        shock_files (tuple): Legacy quarterly and annual shock CSVs (stochastic_data_quarterly.csv and
            stochastic_data_annual.csv in the folder of csv_file by default).
    """
    csv_path = Path(csv_file) if Path(csv_file).exists() else INPUT_DIR / csv_file
    df = pd.read_csv(csv_path)
    if vintage is None:
        vintage = csv_path.stem.replace('deterministic_data_', '')
    if out_file is None:
        out_file = INPUT_DIR / f'dsa_inputs_{vintage}.xlsx'

    sheets = None
    if history_guidance is not None:
        from .commission import load_prior_guidance
        sheets = load_prior_guidance(countries, vintage=history_guidance)

    data = {}
    reference_year = None
    for iso in countries:
        dfc = df.loc[df['COUNTRY'] == iso].set_index('YEAR').drop(columns='COUNTRY')
        if dfc.empty:
            raise ValueError(f'{iso} not found in {csv_file}')
        raw_params = dfc.loc[0]
        series = dfc.loc[dfc.index > 0].sort_index()
        T = int(series.index.min())
        reference_year = T if reference_year is None else min(reference_year, T)
        series = series.reindex(range(T, int(series.index.max()) + 1))
        data[iso] = _convert_country(iso, raw_params, series, T)
        if sheets is not None:
            _add_history(data[iso], sheets[iso], T, history_guidance)
            _add_edp_status(data[iso], sheets[iso], history_guidance)

    meta = {
        'vintage': vintage,
        'mode': 'legacy',
        'reference_year': reference_year,
        'description': f'Converted from legacy input file {csv_file}. Country-specific assumptions that were '
                       'hard-coded in earlier model versions are written into the parameters and STOCK_FLOW_RATIO.',
    }
    # Legacy stochastic shock files (first differences), if available
    from .shocks import legacy_shocks
    q, a = shock_files or (csv_path.parent / 'stochastic_data_quarterly.csv', csv_path.parent / 'stochastic_data_annual.csv')
    q, a = Path(q), Path(a)
    shocks = legacy_shocks(q, a) if q.exists() and a.exists() else None
    if shocks:
        meta['shock_sources'] = {'Shocks_Q/Shocks_A': f'Legacy files {q.name} and {a.name}'}
    else:
        warnings.warn(f'Legacy shock files not found ({q}, {a}): the workbook has no shock data for the stochastic '
                      f'model; pass shock_files or add Shocks_Q/Shocks_A sheets')
    write_workbook(out_file, data, meta, shocks=shocks)
    return out_file


def _add_edp_status(data, sheet, vintage):
    """
    EDP status in T from the Commission prior guidance sheet (legacy files do not record it).
    """
    from .commission import commission_parameters
    params, sources = commission_parameters(sheet)
    data['params']['EXCESSIVE_DEFICIT_PROCEDURE'] = params['EXCESSIVE_DEFICIT_PROCEDURE']
    data['param_sources']['EXCESSIVE_DEFICIT_PROCEDURE'] = ('commission', f'Commission {vintage} prior guidance sheet')


def _add_history(data, sheet, T, vintage, years=2):
    """
    Add the fiscal balance in T-years to T-1 (where available, T-1 required) from a Commission prior guidance sheet,
    as legacy files start in T.
    """
    history = sheet.row('Baseline NFPC', r'^Headline balance').reindex(range(T - years, T)).dropna()
    if T - 1 not in history.index:
        raise ValueError(f'No fiscal balance for {T - 1} in the Commission {vintage} prior guidance sheet')
    series, prov = data['series'], data['series_provenance']
    index = range(T - years, int(series.index.max()) + 1)
    data['series'], data['series_provenance'] = series.reindex(index), prov.reindex(index)
    data['series'].loc[history.index, 'FISCAL_BALANCE'] = history.values
    data['series_provenance'].loc[history.index, 'FISCAL_BALANCE'] = 'commission'
    data['series_sources']['FISCAL_BALANCE'] = (data['series_sources'].get('FISCAL_BALANCE', 'legacy')
                                                + f'; history before T from Commission {vintage} prior guidance sheet')


def _convert_country(iso, raw, series, T):
    params, psrc = {}, {}
    for old, new in PARAM_MAP.items():
        params[new] = raw.get(old, np.nan)
        psrc[new] = ('legacy', f'legacy {old}')

    # Euro debt share only used for non-euro countries
    if iso not in NON_EURO_EUR_SHARE:
        params['DEBT_EUR_SHARE'] = 0.0
        psrc['DEBT_EUR_SHARE'] = ('assumption', 'Euro area member: euro debt counted as domestic')

    # T+10 inflation: POL/ROU keep half of the T+2 gap to euro area inflation
    if iso in ['POL', 'ROU']:
        infl_t10 = params['INFLATION_T10']
        infl_t10 += (series.loc[T + 2, 'GDP_DEFLATOR_PCH'] - series.loc[T + 2, 'EA_GDP_DEFLATOR_PCH']) / 2
        params['INFLATION_T10'] = infl_t10
        psrc['INFLATION_T10'] = ('assumption', 'Euro area 5y5y inflation swap plus half of the T+2 gap to euro area inflation')
    else:
        psrc['INFLATION_T10'] = ('legacy', 'Euro area 5y5y inflation swap (Bloomberg)')

    # T+30 anchors: 2% inflation target (+0.5 POL/ROU, +1 HUN), long-term rate = target + 2, short-term = 0.5 x long-term
    if iso in ['POL', 'ROU']:
        params['INFLATION_T30'], params['INTEREST_RATE_LT_T30'] = 2.5, 4.5
    elif iso == 'HUN':
        params['INFLATION_T30'], params['INTEREST_RATE_LT_T30'] = 3.0, 5.0
    else:
        params['INFLATION_T30'], params['INTEREST_RATE_LT_T30'] = 2.0, 4.0
    params['INTEREST_RATE_ST_T30'] = params['INTEREST_RATE_LT_T30'] * 0.5
    for p in ['INFLATION_T30', 'INTEREST_RATE_LT_T30', 'INTEREST_RATE_ST_T30']:
        psrc[p] = ('assumption', 'DSM 2023 methodology: national inflation target; LT rate = target + 2; ST rate = 0.5 x LT rate')
    psrc['INTEREST_RATE_ST_T10'] = ('legacy', 'Bloomberg 3M10Y forward rate (spread to DEU if unavailable)')
    psrc['INTEREST_RATE_LT_T10'] = ('legacy', 'Bloomberg 10Y10Y forward rate (spread to DEU if unavailable)')

    params['LAST_FORECAST_YEAR'] = T + 1
    psrc['LAST_FORECAST_YEAR'] = ('assumption', 'Earlier model versions: output gap follows the forecast in T+1 only')
    params['EXCESSIVE_DEFICIT_PROCEDURE'] = np.nan
    psrc['EXCESSIVE_DEFICIT_PROCEDURE'] = ('assumption', 'Not recorded in legacy files')
    params['FISCAL_MULTIPLIER_PERSISTENT'] = 0
    psrc['FISCAL_MULTIPLIER_PERSISTENT'] = ('assumption', 'Earlier model versions used the Commission output gap closure rule by default')
    params['FISCAL_MULTIPLIER'] = 0.75
    psrc['FISCAL_MULTIPLIER'] = ('assumption', 'Carnot and de Castro (2015)')

    # Series in schema order, exogenous stock-flow paths
    out = pd.DataFrame(index=series.index, columns=list(SERIES), dtype=float)
    for code in SERIES:
        if code in series.columns:
            out[code] = series[code].astype(float)
    out['STOCK_FLOW_RATIO'] = _legacy_sf_ratio(iso, series, T)

    prov = pd.DataFrame('legacy', index=out.index, columns=out.columns)
    prov['STOCK_FLOW_RATIO'] = 'assumption'
    ssrc = dict(LEGACY_SOURCES)
    ssrc['STOCK_FLOW_RATIO'] = {
        'FIN': 'Finland 2024 medium-term fiscal-structural plan (previously hard-coded)',
        'GRC': 'Commission numbers for the Greek 2024 plan (previously hard-coded)',
        'LUX': 'Ageing Report pension balance T+3 to T+10, linear to zero by T+24 (DSM 2023)',
    }.get(iso, 'None: stock-flow is zero after T+2')

    return {'params': params, 'param_sources': psrc, 'series': out, 'series_provenance': prov, 'series_sources': ssrc,
            'reference_year': T}


def _legacy_sf_ratio(iso, series, T):
    sf = pd.Series(np.nan, index=series.index)
    if iso == 'FIN':
        for y, v in LEGACY_SF_FIN.items():
            if y in sf.index:
                sf[y] = v
    elif iso == 'GRC':
        for y, v in LEGACY_SF_GRC.items():
            if y in sf.index:
                sf[y] = v
    elif iso == 'LUX':
        for t in range(3, 11):
            sf[T + t] = series.loc[T + t, 'PENSION_BALANCE']
        sf10 = sf[T + 10]
        for t in range(11, 25):
            sf[T + t] = sf10 - (t - 10) * sf10 / 14
    return sf
