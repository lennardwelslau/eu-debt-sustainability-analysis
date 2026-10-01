# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - API Mode Builder         #
# ========================================================================================= #
#
# Assembles model inputs from up-to-date public sources, following the Commission methodology
# (Debt Sustainability Monitor, Annex A3):
#
#   T to L        Commission forecast (AMECO): GDP, deflator, balances, debt, stock-flow, IIR, where L is
#                 the last forecast year (T+1 for spring, T+2 for autumn forecasts); later years are
#                 projected by the model
#   L+1 to T+5    Output Gaps Working Group projections if supplied (ogwg_file); otherwise the
#                 output gap closes linearly by T+5 and potential growth is interpolated
#   T+10 onwards  Ageing Report potential growth; ageing costs from T
#   Markets       ECB benchmark rates in T and T+1; T+10 anchors from the most recent Bloomberg file in
#                 data/RawData/market (forward rates, euro area 5y5y inflation swap) if supplied, otherwise
#                 from the Commission prior guidance sheet; T+30 anchors from the Commission sheet
#   Debt          ECB debt structure; Eurostat (gov_10dd_rmd, gov_10dd_ggd) where ECB data are missing or
#                 implausible, then the Commission prior guidance sheet, then the EU median; maturing shares
#                 exclude ESM/EFSF loans (own schedule in the model); average residual maturity and repayment
#                 profile of long-term debt (BOND_REPAYMENT) from Eurostat; ESM repayments
#   Stock-flow    AMECO to L; country-specific paths after L from the latest DSM fiches
#
# As in the Commission prior guidance, the reference year T is the year of the forecast vintage, i.e.
# the year after the last outturn year in Eurostat's EDP notification data (spring forecast: T+1 is
# the last forecast year; autumn forecast: T+2).
#
# Author: Lennard Welslau
# ========================================================================================= #

import warnings
import numpy as np
import pandas as pd
import requests

from .schema import SERIES
from . import sources as S
from . import maturity as M
from .commission import commission_parameters

END_YEAR = 2070

# Inflation anchors (DSM 2024, Annex A3): at T+10 the euro area 5y5y inflation swap, plus half of the inflation spread to
# the euro area for non-euro area countries targeting inflation other than 2% (footnote 6; the spread observed in T+2,
# here in the last forecast year, which is T+2 for autumn and T+1 for spring forecasts); at T+30 the
# national inflation target (footnote 7, sources.INFLATION_TARGETS). T+30 interest rates: long-term rate = inflation
# target + 2, short-term rate = half of the long-term rate.
INFLATION_SPREAD_COUNTRIES = [c for c, target in S.INFLATION_TARGETS.items() if target != 2.0]


def _ameco_vintage():
    try:
        r = requests.get('https://api.db.nomics.world/v22/datasets/AMECO/UDGG', timeout=30).json()
        return r['datasets']['docs'][0].get('indexed_at', '')[:10]
    except Exception:
        return 'unknown'


def build_countries(countries, sheets, reference_year=None, ogwg_file=None, market_file='latest', verbose=True):
    """
    Build input data for all countries from public APIs and reference files.

    Parameters:
        countries (list): ISO3 codes.
        sheets (dict): Parsed Commission prior guidance sheets by country (market assumptions, fallbacks).
        reference_year (int): Reference year T. Defaults to the year after the last outturn year in Eurostat's EDP
            notification data (the vintage year of the forecast, as in the Commission prior guidance).
        ogwg_file (str/Path): Optional Output Gaps Working Group file with real and potential GDP to T+5.
        market_file (str/Path): Bloomberg market expectations for the T+10 anchors (sources.read_market_expectations);
            'latest' (default) uses the most recent file in data/RawData/market, None the Commission sheet values.

    Returns:
        (data dict for write_workbook, description string)
    """
    if verbose:
        print('Fetching AMECO (DBnomics) ...')
    ameco, _ = S.fetch_ameco(countries)
    ameco_vintage = _ameco_vintage()
    L = int(min(ameco[(c, 'DEBT_RATIO')].last_valid_index() for c in countries))  # last forecast year
    if reference_year is None:
        try:
            reference_year = S.fetch_last_outturn_year(countries) + 1
        except Exception as e:
            reference_year = L - 2
            warnings.warn(f'Eurostat EDP data not available ({e}); reference year set to last AMECO year minus two')
    T = reference_year
    if not T + 1 <= L <= T + 2:
        raise ValueError(f'AMECO forecast ends in {L}, expected T+1 or T+2 for reference year T = {T}')

    if verbose:
        print(f'Reference year T = {T}, last forecast year {L}. Fetching ECB rates and debt structure, ESM repayments ...')
    rates = S.fetch_benchmark_rates(countries, [T, T + 1])
    esm_history = S.fetch_esm_repayments(countries, 2010)  # incl. disbursements, for past ESM/EFSF loan stocks
    esm = esm_history.loc[T:]
    debt, debt_years = S.fetch_debt_structure(countries, esm=esm_history)
    if verbose:
        print('Fetching Eurostat debt maturity data ...')
    maturity = M.fetch_maturity_data(countries)
    eurostat = M.eurostat_debt_structure(maturity, countries, esm=esm_history)[0]
    debt_params, debt_sources, debt_median = _debt_structure(debt, eurostat, countries)
    profiles, profile_info = M.repayment_profiles(maturity, countries, T, esm, end_year=END_YEAR)

    if verbose:
        print('Reading Ageing Report and DSM reference files ...')
    ref = S.download_reference_files()
    awg_ageing = S.read_awg_table(ref['awg'], 'Table II.1.135')
    awg_growth = S.read_awg_table(ref['awg'], 'Table II.1.14')
    dsm = S.read_dsm_fiche_rows(ref['dsm'], countries, {'SF': r'^\(3\) Stock-flow', 'TPI': r'^\(1\.1\.3\) Others'})
    ogwg = S.read_ogwg(ogwg_file, countries) if ogwg_file else None
    market_file = S.latest_market_file() if market_file == 'latest' else market_file
    market, market_vintage = S.read_market_expectations(market_file, countries) if market_file else (None, None)

    ea_deflator = ameco[('EA20', 'GDP_DEFLATOR')].pct_change() * 100
    usd_per_eur = ameco[('USA', 'XNE')]

    data = {}
    for c in countries:
        data[c] = _build_country(c, T, L, ameco[c], ea_deflator, usd_per_eur, rates, debt.loc[c], esm[c],
                                 awg_ageing[c], awg_growth[c], dsm['SF'].get(c), dsm['TPI'].get(c),
                                 ogwg, sheets[c], ameco_vintage, debt_params.loc[c], debt_sources[c], debt_median,
                                 profiles.get(c), profile_info.get(c),
                                 market.loc[c] if market is not None else None, market_vintage)
        _check(c, T, L, data[c])

    description = (f'Up-to-date public data. AMECO via DBnomics (indexed {ameco_vintage}), ECB Data Portal '
                   f'(debt structure to {max(debt_years.values())}), Eurostat debt maturity (gov_10dd_rmd, gov_10dd_ggd), '
                   f'ESM repayment database, 2024 Ageing Report, '
                   f'DSM 2025 country fiches, '
                   f"{'OGWG file ' + str(ogwg_file) if ogwg_file else 'no OGWG file (output gap closes by T+5)'}. "
                   + (f'Market anchors at T+10 from Bloomberg ({market_vintage}); market anchors at T+30, ' if market is not None
                      else f'Market anchors (T+10, T+30), ')
                   + f'EDP status and semi-elasticities from the Commission prior '
                   f'guidance sheets; fiscal multiplier 0.75 with persistent effect on the output gap.')
    return data, description


# Debt structure parameters. Fallback order: ECB, Eurostat, Commission sheet; the maturing shares fall back to the EU
# median where the Commission sheet value is implausible too (e.g. 0.001 for Luxembourg)
DEBT_SHARES = ['DEBT_ST_SHARE', 'DEBT_LT_MATURING_SHARE', 'DEBT_LT_MATURING_AVG_SHARE']
MATURING = ['DEBT_LT_MATURING_SHARE', 'DEBT_LT_MATURING_AVG_SHARE']


def _plausible(v):
    return pd.notna(v) and 0.005 < v < 1


def _debt_structure(ecb, eurostat, countries):
    """
    Debt structure parameters from the ECB, with Eurostat where the ECB data are missing or implausible. The two
    maturing shares are taken from the same source (the ECB residual maturity series is implausible for some
    countries in all years); a share alone is taken from the ECB if neither pair is usable.
    Returns (DataFrame ISO3 x parameter, NaN where neither source is usable; dict ISO3 -> {parameter: (provenance,
    note)}; Series of EU medians of the parameters).
    """
    out = pd.DataFrame(np.nan, index=countries, columns=DEBT_SHARES + ['DEBT_AVG_RESIDUAL_MATURITY'])
    src = {c: {} for c in countries}
    for c in countries:
        e, u = ecb.loc[c], eurostat.loc[c]
        for group in [['DEBT_ST_SHARE'], MATURING]:
            if all(_plausible(e.get(p)) for p in group):
                for p in group:
                    out.loc[c, p], src[c][p] = e[p], ('api', 'ECB GFS debt structure')
            elif all(_plausible(u.get(p)) for p in group):
                for p in group:
                    out.loc[c, p], src[c][p] = u[p], ('api', 'Eurostat gov_10dd_rmd/ggd (ECB data missing or implausible)')
            else:
                for p in group:
                    if _plausible(e.get(p)):
                        out.loc[c, p], src[c][p] = e[p], ('api', 'ECB GFS debt structure')
        if pd.notna(u.get('DEBT_AVG_RESIDUAL_MATURITY')):
            out.loc[c, 'DEBT_AVG_RESIDUAL_MATURITY'] = u['DEBT_AVG_RESIDUAL_MATURITY']
            src[c]['DEBT_AVG_RESIDUAL_MATURITY'] = ('api', 'Eurostat gov_10dd_rmd, average residual maturity of debt')
    return out, src, out.median()


def _build_country(c, T, L, am, ea_deflator, usd_per_eur, rates, debt, esm, awg_ageing, awg_growth,
                   dsm_sf, dsm_tpi, ogwg, sheet, ameco_vintage, debt_params, debt_sources, debt_median,
                   profile=None, profile_info=None, market=None, market_vintage=None):
    years = range(T - 3, END_YEAR + 1)
    out = pd.DataFrame(index=pd.Index(years, name='YEAR'), columns=list(SERIES), dtype=float)
    prov = pd.DataFrame(None, index=out.index, columns=out.columns, dtype=object)
    src = {}
    fc = slice(T - 3, L)
    ameco_src = f'AMECO via DBnomics (indexed {ameco_vintage})'

    def put(code, values, provenance, note, rng=None):
        values = pd.Series(values, dtype=float).reindex(out.index)
        if rng is not None:
            values = values.loc[rng].reindex(out.index)
        mask = values.notna()
        out.loc[mask, code] = values[mask]
        prov.loc[mask, code] = provenance
        src[code] = note if code not in src else f'{src[code]}; {note}'

    # Commission forecast T-3 to L
    for code in ['NOMINAL_GDP', 'DEBT_TOTAL', 'DEBT_RATIO', 'PRIMARY_BALANCE', 'FISCAL_BALANCE',
                 'STRUCTURAL_PRIMARY_BALANCE', 'PRIMARY_EXPENDITURE_SHARE', 'IMPLICIT_INTEREST_RATE', 'STOCK_FLOW',
                 'ONE_OFF_MEASURES']:
        put(code, am[code], 'api', f'{ameco_src}: {S.AMECO[code]}', fc)
    put('NOMINAL_GDP_GROWTH', am['NOMINAL_GDP'].pct_change() * 100, 'api', f'{ameco_src}: growth of {S.AMECO["NOMINAL_GDP"]}', fc)
    put('GDP_DEFLATOR_PCH', am['GDP_DEFLATOR'].pct_change() * 100, 'api', f'{ameco_src}: growth of {S.AMECO["GDP_DEFLATOR"]}', fc)
    put('EA_GDP_DEFLATOR_PCH', ea_deflator, 'api', f'{ameco_src}: EA20 GDP deflator growth', fc)

    # Exchange rates: euro and US dollar per unit of national currency
    exr_eur = 1 / am['XNE']
    put('EXR_EUR', exr_eur, 'api', f'{ameco_src}: inverse of {S.AMECO["XNE"]}', fc)
    put('EXR_USD', exr_eur * usd_per_eur, 'api', f'{ameco_src}: EUR rate times USD per EUR', fc)

    # Real and potential GDP
    rgdp, pgdp = am['REAL_GDP'].loc[fc], am['POTENTIAL_GDP'].loc[fc]
    put('REAL_GDP', rgdp, 'api', f'{ameco_src}: {S.AMECO["REAL_GDP"]} to the last forecast year')
    put('POTENTIAL_GDP', pgdp, 'api', f'{ameco_src}: {S.AMECO["POTENTIAL_GDP"]} to the last forecast year')
    rgdp, pgdp = rgdp.copy(), pgdp.copy()
    if ogwg is not None and c in ogwg['REAL_GDP'].columns:
        for code, level, name in [('REAL_GDP', rgdp, 'real'), ('POTENTIAL_GDP', pgdp, 'potential')]:
            growth = ogwg[code][c].pct_change()
            for y in range(L + 1, T + 6):
                level[y] = level[y - 1] * (1 + growth[y])
            put(code, level.loc[L + 1:T + 5], 'reference', f'{L + 1}-{T + 5}: AMECO level grown with OGWG {name} growth')
        pot_growth_T5 = (pgdp[T + 5] / pgdp[T + 4] - 1) * 100
        last_short = T + 5
    else:
        # Output gap closes linearly by T+5, potential growth interpolated between L and T+10 (Ageing Report)
        g_L = (pgdp[L] / pgdp[L - 1] - 1) * 100
        g_T10 = awg_growth.get(T + 10, np.nan)
        gap_L = (rgdp[L] / pgdp[L] - 1) * 100
        n_close = T + 5 - L
        for k, y in enumerate(range(L + 1, T + 6), start=1):
            g = g_L + (g_T10 - g_L) * k / (T + 10 - L)
            pgdp[y] = pgdp[y - 1] * (1 + g / 100)
            rgdp[y] = pgdp[y] * (1 + gap_L * (n_close - k) / n_close / 100)
        put('REAL_GDP', rgdp.loc[L + 1:T + 5], 'derived', f'{L + 1}-{T + 5}: output gap closes linearly by T+5 (no OGWG file)')
        put('POTENTIAL_GDP', pgdp.loc[L + 1:T + 5], 'derived', f'{L + 1}-{T + 5}: potential growth interpolated to Ageing Report T+10 (no OGWG file)')
        pot_growth_T5 = (pgdp[T + 5] / pgdp[T + 4] - 1) * 100
        last_short = T + 5
    put('REAL_GDP_GROWTH', rgdp.pct_change() * 100, 'derived', 'Growth of REAL_GDP', slice(T - 2, last_short))

    # Potential growth: short term from levels, Ageing Report from T+10, linear interpolation in between
    pg = (pgdp.pct_change() * 100).loc[T - 2:last_short]
    put('POTENTIAL_GDP_GROWTH', pg, 'derived', 'Growth of POTENTIAL_GDP to T+5')
    lt = pd.Series(awg_growth).loc[T + 10:END_YEAR]
    put('POTENTIAL_GDP_GROWTH', lt, 'reference', 'Ageing Report 2024 (Table II.1.14) from T+10')
    between = pd.Series(np.nan, index=range(last_short, T + 11), dtype=float)
    between[last_short], between[T + 10] = pot_growth_T5, lt.get(T + 10, np.nan)
    put('POTENTIAL_GDP_GROWTH', between.interpolate().loc[last_short + 1:T + 9], 'derived', 'linear interpolation T+6 to T+9')

    # Market rates in T and T+1 where available (current year: year to date)
    for code in ['INTEREST_RATE_ST', 'INTEREST_RATE_LT']:
        note = 'ECB GFS 3M rate, last observation of year' if code.endswith('ST') else 'ECB IRS 10Y benchmark yield, annual mean'
        put(code, rates[code][c], 'api', f'{note} (T+1: year to date)')

    # Ageing costs and revenues
    put('AGEING_COST', pd.Series(awg_ageing).loc[T:END_YEAR], 'reference', 'Ageing Report 2024 (Table II.1.135), total cost of ageing')
    if dsm_tpi is not None:
        tpi = dsm_tpi.dropna()
        tpi = tpi.reindex(range(T, int(tpi.index.max()) + 1)).fillna(0.0)
        slope = np.polyfit(tpi.index, tpi.values, 1)[0] if len(tpi) > 2 else 0.0
        put('TAX_AND_PROPERTY_INCOME', tpi, 'reference', 'DSM 2025 country fiche (1.1.3) Others (taxes and property income), cumulative change')
        ext = pd.Series({y: tpi.iloc[-1] + slope * (y - tpi.index[-1]) for y in range(tpi.index[-1] + 1, END_YEAR + 1)})
        put('TAX_AND_PROPERTY_INCOME', ext, 'derived', 'linear extrapolation to 2070')

    # Stock-flow exceptions after the forecast from DSM, converging linearly to zero over 10 years after the last DSM year
    if dsm_sf is not None:
        sf = dsm_sf.loc[L + 1:].dropna()
        if (sf.abs() > 1e-6).any():
            put('STOCK_FLOW_RATIO', sf, 'reference', f'DSM 2025 country fiche (3) Stock-flow adjustments, from {L + 1}')
            last, v = int(sf.index.max()), sf.iloc[-1]
            tail = pd.Series({last + k: v * (1 - k / 10) for k in range(1, 11)})
            put('STOCK_FLOW_RATIO', tail, 'derived', 'converges linearly to zero over 10 years')

    # ESM repayments
    put('ESM_REPAYMENT', esm.loc[T:END_YEAR], 'api', 'ESM repayment database (RepaymentData.csv), bn EUR')

    # Repayments of long-term debt outstanding at the end of T-1 (used if bond_data=True)
    if profile is not None:
        put('BOND_REPAYMENT', profile, 'derived' if profile_info['estimated'] else 'api', profile_info['note'])

    # Parameters: market anchors, elasticities from Commission sheet; debt structure from ECB with fallback
    params, psrc = commission_parameters(sheet)
    params['LAST_FORECAST_YEAR'] = L
    psrc['LAST_FORECAST_YEAR'] = ('api', 'Last AMECO forecast year')
    params['FISCAL_MULTIPLIER_PERSISTENT'] = 1
    psrc['FISCAL_MULTIPLIER_PERSISTENT'] = ('assumption', 'Persistent multiplier effect relative to the baseline output gap')
    params['FISCAL_MULTIPLIER'] = 0.75
    psrc['FISCAL_MULTIPLIER'] = ('assumption', 'Carnot and de Castro (2015); Commission sheets from 2025 use 0.6')
    psrc['EXCESSIVE_DEFICIT_PROCEDURE'] = ('commission', psrc['EXCESSIVE_DEFICIT_PROCEDURE'][1] + ' - update if EDP status changed')
    # EDP decisions after the guidance sheet (the sheet's flag refers to its own reference year)
    if c in S.EDP_STATUS_UPDATES and sheet.T < S.EDP_STATUS_UPDATES[c][1] <= T:
        params['EXCESSIVE_DEFICIT_PROCEDURE'] = S.EDP_STATUS_UPDATES[c][0]
        psrc['EXCESSIVE_DEFICIT_PROCEDURE'] = ('assumption', S.EDP_STATUS_UPDATES[c][2] + ' (data_pipeline.sources.EDP_STATUS_UPDATES)')
    note_T10 = f' (T+10 refers to {sheet.T + 10} in the guidance, applied at T+10 = {T + 10} here)'
    for p in ['INTEREST_RATE_ST_T10', 'INTEREST_RATE_LT_T10', 'INFLATION_T10']:
        psrc[p] = (psrc[p][0], psrc[p][1] + note_T10)

    # T+10 anchors from a more recent Bloomberg file where supplied (replacing the Commission sheet values)
    if market is not None and market[['FWD_RATE_3M10Y', 'FWD_RATE_10Y10Y', 'FWD_INFL_5Y5Y']].notna().all():
        bbg = f'Bloomberg ({market_vintage}, data/RawData/market)'
        params['INTEREST_RATE_ST_T10'] = market['FWD_RATE_3M10Y']
        psrc['INTEREST_RATE_ST_T10'] = ('reference', f'{bbg}: 3M10Y forward rate')
        params['INTEREST_RATE_LT_T10'] = market['FWD_RATE_10Y10Y']
        psrc['INTEREST_RATE_LT_T10'] = ('reference', f'{bbg}: 10Y10Y forward rate')
        params['INFLATION_T10'] = market['FWD_INFL_5Y5Y']
        psrc['INFLATION_T10'] = ('reference', f'{bbg}: euro area 5y5y inflation swap')
        if c in INFLATION_SPREAD_COUNTRIES:
            params['INFLATION_T10'] += (out.loc[L, 'GDP_DEFLATOR_PCH'] - out.loc[L, 'EA_GDP_DEFLATOR_PCH']) / 2
            psrc['INFLATION_T10'] = ('reference', f'{bbg}: euro area 5y5y inflation swap plus half of the inflation spread '
                                                  f'to the euro area in {L} (last forecast year)')

    # T+30 anchors from the national inflation target (the Commission sheets use the same values)
    target = S.inflation_target(c)
    t30 = {'INFLATION_T30': target, 'INTEREST_RATE_LT_T30': target + 2, 'INTEREST_RATE_ST_T30': (target + 2) / 2}
    for p, v in t30.items():
        if pd.notna(params.get(p)) and abs(params[p] - v) > 1e-6:
            warnings.warn(f'{c}: {p} = {v} (inflation target) differs from the Commission sheet ({params[p]})')
        params[p] = v
        psrc[p] = ('assumption', f'DSM 2024 methodology: inflation target {target:g}% (national target of non-euro area '
                                 f'inflation targeters, 2% otherwise); long-term rate = target + 2; short-term rate = half '
                                 f'of the long-term rate')
    euro_member = S.EURO_ADOPTION.get(c, 9999) <= T + 1
    for p in DEBT_SHARES:
        if pd.notna(debt_params[p]):
            params[p], psrc[p] = debt_params[p], debt_sources[p]
        elif p == 'DEBT_ST_SHARE' or _plausible(params.get(p)):
            psrc[p] = ('commission', psrc[p][1] + ' (ECB and Eurostat data missing or implausible)')
        else:
            params[p] = debt_median[p]
            psrc[p] = ('assumption', 'EU median (ECB, Eurostat and Commission sheet data missing or implausible)')
    if pd.notna(debt_params['DEBT_AVG_RESIDUAL_MATURITY']):
        params['DEBT_AVG_RESIDUAL_MATURITY'] = debt_params['DEBT_AVG_RESIDUAL_MATURITY']
        psrc['DEBT_AVG_RESIDUAL_MATURITY'] = debt_sources['DEBT_AVG_RESIDUAL_MATURITY']
    else:
        psrc['DEBT_AVG_RESIDUAL_MATURITY'] = ('derived', 'Not available from Eurostat; the model uses 1 / DEBT_LT_MATURING_AVG_SHARE')
    dom, eur = debt.get('DEBT_DOMESTIC_SHARE', np.nan), debt.get('DEBT_EUR_SHARE', np.nan)
    eur = 0.0 if pd.isna(eur) else eur
    if euro_member:
        params['DEBT_DOMESTIC_SHARE'] = 1.0 if pd.isna(dom) else min(dom + eur, 1.0)
        params['DEBT_EUR_SHARE'] = 0.0
        psrc['DEBT_DOMESTIC_SHARE'] = ('api', 'ECB GFS: domestic plus euro-denominated debt (euro area member)')
        psrc['DEBT_EUR_SHARE'] = ('assumption', 'Euro area member: euro debt counted as domestic')
    else:
        params['DEBT_DOMESTIC_SHARE'], params['DEBT_EUR_SHARE'] = dom, eur
        psrc['DEBT_DOMESTIC_SHARE'] = psrc['DEBT_EUR_SHARE'] = ('api', 'ECB GFS currency composition of debt')

    return {'params': params, 'param_sources': psrc, 'series': out, 'series_provenance': prov,
            'series_sources': src, 'reference_year': T}


REQUIRED_FORECAST = [  # series that must be available from T to the last forecast year L
    'NOMINAL_GDP', 'DEBT_RATIO', 'DEBT_TOTAL', 'GDP_DEFLATOR_PCH', 'STRUCTURAL_PRIMARY_BALANCE', 'PRIMARY_BALANCE',
    'FISCAL_BALANCE', 'IMPLICIT_INTEREST_RATE', 'STOCK_FLOW', 'EXR_EUR', 'EXR_USD', 'REAL_GDP', 'POTENTIAL_GDP',
]
REQUIRED_T = ['INTEREST_RATE_ST', 'INTEREST_RATE_LT', 'PRIMARY_EXPENDITURE_SHARE']  # required in T
REQUIRED_HISTORY = ['FISCAL_BALANCE']  # required in T-1 (EDP abrogation rule)


def _check(c, T, L, d):
    s = d['series']
    missing = [f'{code} ({y})' for code in REQUIRED_FORECAST for y in range(T, L + 1) if pd.isna(s.loc[y, code])]
    missing += [f'{code} ({T})' for code in REQUIRED_T if pd.isna(s.loc[T, code])]
    missing += [f'{code} ({T - 1})' for code in REQUIRED_HISTORY if pd.isna(s.loc[T - 1, code])]
    missing += [f'{code} (T+..2070)' for code in ['POTENTIAL_GDP_GROWTH', 'AGEING_COST'] if s.loc[T + 1:, code].isna().any()]
    missing += [p for p, v in d['params'].items() if pd.isna(v)]
    if missing:
        warnings.warn(f'{c}: missing inputs {missing}. Fill them in the Overrides sheet.')
