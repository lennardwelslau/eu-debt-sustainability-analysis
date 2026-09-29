# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Stochastic Shock Data    #
# ========================================================================================= #
#
# Historical series for the stochastic DSA (DSM Annex A4). Shocks are first differences of
# quarterly (or annual) series of:
#
#   EXR_EUR             euro per unit of national currency     Eurostat ert_bil_eur (1 for euro area)
#   EXR_USD             US dollar per unit of national currency Eurostat ert_bil_eur (cross rate)
#   INTEREST_RATE_ST    3-month rate                            Eurostat irt_st (euro area), OECD (others)
#   INTEREST_RATE_LT    10-year government bond yield           Eurostat irt_lt_mcby (EST: ECB MIR)
#   NOMINAL_GDP_GROWTH  nominal GDP growth, y-o-y               Eurostat namq_10_gdp / nama_10_gdp
#   PRIMARY_BALANCE     net lending + interest payable, % GDP   Eurostat gov_10q_ggnfa / gov_10dd_edpt1
#
# Missing quarterly values are filled with the annual value in Q4 and interpolated linearly
# within the sample (no extrapolation beyond the last observation).
#
# Author: Lennard Welslau
# ========================================================================================= #

import io
import pandas as pd

from .sources import _get, fetch_ecb, EURO_ADOPTION, CURRENCY
from .commission import EC_ISO2

EUROSTAT_URL = 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
OECD_URL = 'https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/'
SHOCK_VARIABLES = ['EXR_EUR', 'EXR_USD', 'INTEREST_RATE_ST', 'INTEREST_RATE_LT', 'NOMINAL_GDP_GROWTH', 'PRIMARY_BALANCE']
SHOCK_SOURCES = {
    'EXR_EUR': 'Eurostat ert_bil_eur_q/a (average), inverted; 1 for euro area members (incl. BGN, pegged)',
    'EXR_USD': 'Eurostat ert_bil_eur_q/a: USD per EUR times EUR per national currency',
    'INTEREST_RATE_ST': 'Eurostat irt_st_q/a (3-month rate, euro area) / OECD IR3TIB (non-euro countries)',
    'INTEREST_RATE_LT': 'Eurostat irt_lt_mcby_q/a (EMU convergence yield); Estonia: ECB MIR lending rate',
    'NOMINAL_GDP_GROWTH': 'Eurostat namq_10_gdp (SCA, y-o-y) / nama_10_gdp, current prices',
    'PRIMARY_BALANCE': 'Eurostat gov_10q_ggnfa (B9 SCA + D41PAY) / gov_10dd_edpt1, % of GDP',
}


def _eurostat(dataset, key, start):
    r = _get(f'{EUROSTAT_URL}{dataset}/{key}', params={'format': 'SDMX-CSV', 'startPeriod': str(start)})
    df = pd.read_csv(io.StringIO(r.text))
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    return df


def _oecd(key, start):
    r = _get(OECD_URL + key, params={'startPeriod': str(start)},
             headers={'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8'})
    df = pd.read_csv(io.StringIO(r.text))
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    return df


def _to_period(s, freq):
    s = s.astype(str)
    if freq == 'Q':
        return pd.PeriodIndex(s.str.replace('-', ''), freq='Q')
    return pd.PeriodIndex(s.str[:4], freq='Y')


def _pivot(df, column, freq, rename=None):
    df = df.copy()
    df['period'] = _to_period(df['TIME_PERIOD'], freq)
    out = df.pivot_table(index='period', columns=column, values='OBS_VALUE')
    return out.rename(columns=rename) if rename else out


def _monthly_to(df, freq):
    df = df.copy()
    df['period'] = pd.PeriodIndex(df['TIME_PERIOD'].astype(str), freq='M').asfreq('Q' if freq == 'Q' else 'Y')
    return df.groupby('period')['OBS_VALUE'].mean()


def fetch_shock_history(countries, freq='Q', start=1998):
    """
    Fetch historical levels of the shock variables. Returns {variable: DataFrame (period x ISO3)}.
    """
    f = freq.lower()
    euro = [c for c in countries if EURO_ADOPTION.get(c, 9999) <= 2025]  # euro area rates for members before 2026
    fixed = [c for c in countries if EURO_ADOPTION.get(c, 9999) <= 2026]  # fixed to the euro (BGN pegged since 1999)
    iso2 = {c: EC_ISO2[c] for c in countries}  # Eurostat codes (Greece is EL)
    inv = {v: k for k, v in iso2.items()}
    geo = '+'.join(iso2.values())
    out = {}

    # Exchange rates (national currency per EUR)
    cur = {c: CURRENCY[c] for c in countries if c not in fixed}
    exr = _pivot(_eurostat(f'ert_bil_eur_{f}', f'{freq}.AVG.NAC.' + '+'.join(list(cur.values()) + ['USD']), start),
                 'currency', freq)
    eur = pd.DataFrame(index=exr.index)
    for c in countries:
        eur[c] = 1.0 if c in fixed else 1 / exr[cur[c]]
    out['EXR_EUR'] = eur
    out['EXR_USD'] = eur.mul(exr['USD'], axis=0)

    # Nominal GDP growth
    if freq == 'Q':
        gdp = _pivot(_eurostat('namq_10_gdp', f'Q.CP_MNAC.SCA.B1GQ.{geo}', start - 1), 'geo', freq, inv)
        out['NOMINAL_GDP_GROWTH'] = gdp.pct_change(4, fill_method=None) * 100
    else:
        gdp = _pivot(_eurostat('nama_10_gdp', f'A.CP_MNAC.B1GQ.{geo}', start - 1), 'geo', freq, inv)
        out['NOMINAL_GDP_GROWTH'] = gdp.pct_change(fill_method=None) * 100

    # Short-term rates: euro area rate for members, OECD 3-month interbank rates for others
    ea = _pivot(_eurostat(f'irt_st_{f}', f'{freq}.IRT_M3.EA', start), 'geo', freq)['EA']
    others = [c for c in countries if c not in euro]
    oecd = _pivot(_oecd('+'.join(others) + '.Q.IR3TIB.PA.....', start), 'REF_AREA', 'Q')
    if freq == 'A':
        oecd = oecd.groupby(oecd.index.asfreq('Y')).mean()
    st = pd.DataFrame(index=ea.index.union(oecd.index))
    for c in countries:
        st[c] = ea if c in euro else oecd.get(c)
    out['INTEREST_RATE_ST'] = st

    # Long-term rates (Estonia: ECB MIR lending rate, as no long-term government bond yield exists)
    lt = _pivot(_eurostat(f'irt_lt_mcby_{f}', f'{freq}.MCBY.{geo}', start), 'geo', freq, inv)
    if 'EST' in countries:
        lt['EST'] = _monthly_to(fetch_ecb('MIR', f'M.EE.B.A2C.I.R.A.2250.EUR.N?startPeriod={start}'), freq)
    out['INTEREST_RATE_LT'] = lt.reindex(columns=countries)

    # Primary balance = net lending/borrowing + interest payable
    if freq == 'Q':
        b9 = _pivot(_eurostat('gov_10q_ggnfa', f'Q.PC_GDP.SCA.S13.B9.{geo}', start), 'geo', freq, inv)
        d41 = _pivot(_eurostat('gov_10q_ggnfa', f'Q.PC_GDP.NSA.S13.D41PAY.{geo}', start), 'geo', freq, inv)
    else:
        df = _eurostat('gov_10dd_edpt1', f'A.PC_GDP.S13.B9+D41PAY.{geo}', start)
        b9 = _pivot(df.loc[df['na_item'] == 'B9'], 'geo', freq, inv)
        d41 = _pivot(df.loc[df['na_item'] == 'D41PAY'], 'geo', freq, inv)
    out['PRIMARY_BALANCE'] = (b9 + d41).reindex(columns=countries)

    return {k: v.loc[v.index >= pd.Period(str(start), freq='Q' if freq == 'Q' else 'Y')].reindex(columns=countries)
            for k, v in out.items()}


def build_shocks(countries, start=1998, verbose=True):
    """
    Build historical levels and shocks (first differences) at quarterly and annual frequency.
    Returns {'History_Q', 'History_A', 'Shocks_Q', 'Shocks_A'} as long DataFrames with
    columns COUNTRY, PERIOD and the shock variables.
    """
    if verbose:
        print('Fetching historical data for stochastic shocks (Eurostat, OECD, ECB) ...')
    annual = fetch_shock_history(countries, 'A', start)
    quarterly = fetch_shock_history(countries, 'Q', start)

    # Fill missing Q4 values with annual values, interpolate within the sample only
    for var, q in quarterly.items():
        a = annual[var]
        for c in countries:
            missing_q4 = q.index[(q.index.quarter == 4) & q[c].isna()]
            for p in missing_q4:
                y = pd.Period(str(p.year), freq='Y')
                if y in a.index and pd.notna(a.loc[y, c]):
                    q.loc[p, c] = a.loc[y, c]
            q[c] = q[c].interpolate(method='linear', limit_area='inside')

    result = {}
    for freq, data in [('Q', quarterly), ('A', annual)]:
        levels = pd.concat({v: data[v] for v in SHOCK_VARIABLES}, axis=1)
        long = levels.stack(level=1, future_stack=True).rename_axis(['PERIOD', 'COUNTRY']).reset_index()
        long = long[['COUNTRY', 'PERIOD'] + SHOCK_VARIABLES].sort_values(['COUNTRY', 'PERIOD'])
        shocks = long.copy()
        shocks[SHOCK_VARIABLES] = shocks.groupby('COUNTRY')[SHOCK_VARIABLES].diff()
        shocks = shocks.dropna()
        for df in (long, shocks):
            df['PERIOD'] = df['PERIOD'].astype(str)
        result[f'History_{freq}'] = long.dropna(how='all', subset=SHOCK_VARIABLES).reset_index(drop=True)
        result[f'Shocks_{freq}'] = shocks.reset_index(drop=True)
    return result


def legacy_shocks(quarterly_csv, annual_csv):
    """
    Read legacy stochastic_data_quarterly/annual CSV files (first differences) into the workbook format.
    """
    out = {}
    for freq, path in [('Q', quarterly_csv), ('A', annual_csv)]:
        df = pd.read_csv(path).rename(columns={'YEAR': 'PERIOD'})
        df['PERIOD'] = df['PERIOD'].astype(str)
        out[f'Shocks_{freq}'] = df[['COUNTRY', 'PERIOD'] + SHOCK_VARIABLES]
    return out
