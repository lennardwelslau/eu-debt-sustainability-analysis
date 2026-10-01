# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Debt Maturity Data       #
# ========================================================================================= #
#
# Maturity structure of general government debt from Eurostat (Maastricht debt, face value):
#
#   gov_10dd_rmd   debt by residual maturity: <= 1, 1-5, 5-10, 10-30 and > 30 years (sum to the
#                  total), and the average residual maturity in years
#   gov_10dd_ggd   debt by original maturity: short-term debt (<= 1 year)
#
# Uses:
#   eurostat_debt_structure   short-term share, share of long-term debt maturing within a year (excluding
#                             ESM/EFSF loans, as in the model) and average residual maturity (fallback for
#                             missing or implausible ECB data)
#   repayment_profiles        BOND_REPAYMENT: repayments of the long-term debt outstanding at the end
#                             of the base year, by year (used with bond_data=True)
#   add_repayment_profiles    BOND_REPAYMENT for the Commission-mode workbooks, scaled to their debt
#                             (levels there are indexed, not in bn)
#
# Repayment profile. The base year is the latest year with data before the reference year T (end of
# T-1 in API builds). Short-term debt (original maturity <= 1 year, repaid each year in the model) is
# deducted from the <= 1 year bucket and ESM/EFSF loans (ESM_REPAYMENT, own schedule in the model)
# from each bucket. Each bucket is spread evenly over its years, the > 30 year bucket over 20 years.
# Countries without the 5-10, 10-30 and > 30 year split split debt maturing after 5 years with the
# average shares of the countries with the full split (excluding Greece, whose EFSF loans make the
# > 30 year bucket unusually large). The profile covers the debt outstanding at the end of the base
# year only; the model scales it to the long-term debt in T (DsaModel._clean_bond_repayment).
#
# Author: Lennard Welslau
# ========================================================================================= #

import io
import time
import numpy as np
import pandas as pd

from .commission import EC_ISO2

EUROSTAT_URL = 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'

# Residual maturity buckets: Eurostat code -> (first, last) year after the base year in which the debt matures
BUCKETS = {'Y_LE1': (1, 1), 'Y1-5': (2, 5), 'Y5-10': (6, 10), 'Y10-30': (11, 30), 'Y_GT30': (31, 50)}
LONG_BUCKETS = ['Y5-10', 'Y10-30', 'Y_GT30']
POOL_EXCLUDE = ['GRC']  # excluded from the average split of debt maturing after 5 years


def _eurostat(dataset, key, start, retries=3):
    """
    Eurostat SDMX-CSV query. Large queries can be queued by Eurostat (XML response); these are retried.
    """
    from .sources import _get
    for attempt in range(retries):
        r = _get(f'{EUROSTAT_URL}{dataset}/{key}', params={'format': 'SDMX-CSV', 'startPeriod': str(start)})
        if r.text.startswith('DATAFLOW'):
            df = pd.read_csv(io.StringIO(r.text))
            df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
            return df
        time.sleep(10)
    raise RuntimeError(f'Eurostat query {dataset}/{key} did not return data: {r.text[:300]}')


def fetch_maturity_data(countries, start=2010):
    """
    Fetch Eurostat debt maturity data. Returns dict of DataFrames indexed by (ISO3, year):
        'residual': debt by residual maturity bucket (TOTAL, Y_LE1, Y1-5, Y_GT1, Y5-10, Y10-30, Y_GT30), bn national currency
        'original': total and short-term debt by original maturity (TOTAL, Y_LE1), bn national currency
        'avg_maturity': average residual maturity of debt in years (Series)
    """
    geo = '+'.join(EC_ISO2[c] for c in countries)
    inv = {v: k for k, v in EC_ISO2.items()}
    # All countries for gov_10dd_rmd: the query fails for countries that do not report it (e.g. HU, LU)
    rmd = _eurostat('gov_10dd_rmd', 'A.S13..GD.MIO_NAC+YR.', start)
    ggd = _eurostat('gov_10dd_ggd', f'A.GD.S1_S2.S13.TOTAL+Y_LE1.MIO_NAC.{geo}', start)
    rmd, ggd = [df.assign(country=df['geo'].map(inv)).dropna(subset=['country']) for df in (rmd, ggd)]
    rmd = rmd[rmd['country'].isin(countries)]
    amounts = rmd['unit'] == 'MIO_NAC'
    residual = rmd[amounts].pivot_table(index=['country', 'TIME_PERIOD'], columns='maturity', values='OBS_VALUE') / 1000
    original = ggd.pivot_table(index=['country', 'TIME_PERIOD'], columns='maturity', values='OBS_VALUE') / 1000
    avg = rmd[(rmd['unit'] == 'YR') & (rmd['maturity'] == 'TOTAL')].set_index(['country', 'TIME_PERIOD'])['OBS_VALUE']
    for df in (residual, original):
        df.index.names = ['country', 'year']
    avg.index.names = ['country', 'year']
    return {'residual': residual, 'original': original, 'avg_maturity': avg.dropna()}


def esm_outstanding(esm, country, years):
    """
    ESM/EFSF loans of a country outstanding at the end of each year and repaid in the following year (bn EUR), from the
    signed ESM schedule (disbursements negative, repayments positive; sources.fetch_esm_repayments from an early start
    year, so that past disbursements are included). Returns (stock, repaid next year) as Series indexed by years.
    """
    flows = esm[country].fillna(0).sort_index() if esm is not None and country in esm else pd.Series(dtype=float)
    stock = pd.Series({y: flows.loc[y + 1:].sum() for y in years}, dtype=float)
    repaid = pd.Series({y: max(flows.get(y + 1, 0.0), 0.0) for y in years}, dtype=float)
    return stock, repaid


def lt_maturing_share(lt, lt_maturing, esm, country, scale=1.0):
    """
    Share of long-term debt maturing within a year, excluding ESM/EFSF loans: the model applies the share to long-term
    debt excluding these loans, which follow their own schedule. lt and lt_maturing are Series by year in bn times
    scale (e.g. 1000 for millions).
    """
    stock, repaid = esm_outstanding(esm, country, lt.index)
    return ((lt_maturing - repaid * scale) / (lt - stock * scale)).dropna()


def _country(df, c):
    return df.xs(c, level='country') if c in df.index.get_level_values('country') else df.iloc[0:0].droplevel('country')


def eurostat_debt_structure(data, countries, esm=None):
    """
    Debt structure parameters from Eurostat, defined as the ECB-based parameters (sources.fetch_debt_structure):
        DEBT_ST_SHARE:              3-year average share of short-term debt (original maturity <= 1 year)
        DEBT_LT_MATURING_SHARE:     latest share of long-term debt with residual maturity <= 1 year, excluding ESM/EFSF
                                    loans (esm: signed ESM schedule by year, see esm_outstanding)
        DEBT_LT_MATURING_AVG_SHARE: 6-year average of the above (all available years)
        DEBT_AVG_RESIDUAL_MATURITY: latest average residual maturity of debt (years)
    Returns DataFrame (ISO3 x parameter) and dict of last years used per parameter.
    """
    rows, years = {}, {}
    for c in countries:
        res, orig = _country(data['residual'], c), _country(data['original'], c)
        avg = _country(data['avg_maturity'].to_frame(), c)['OBS_VALUE']
        empty = pd.Series(dtype=float)
        total, st_amount = orig.get('TOTAL', empty), orig.get('Y_LE1', empty)
        st = (st_amount / total).dropna()
        lt_mat = lt_maturing_share(total - st_amount, res.get('Y_LE1', empty) - st_amount, esm, c)
        rows[c] = {
            'DEBT_ST_SHARE': st.loc[st.index.max() - 2:].mean() if len(st) else np.nan,
            'DEBT_LT_MATURING_SHARE': lt_mat.iloc[-1] if len(lt_mat) else np.nan,
            'DEBT_LT_MATURING_AVG_SHARE': lt_mat.loc[lt_mat.index.max() - 5:].mean() if len(lt_mat) else np.nan,
            'DEBT_AVG_RESIDUAL_MATURITY': avg.iloc[-1] if len(avg) else np.nan,
        }
        years[c] = {'st': st.index.max() if len(st) else None, 'lt_mat': lt_mat.index.max() if len(lt_mat) else None,
                    'avg': avg.index.max() if len(avg) else None}
    return pd.DataFrame(rows).T, years


def _buckets(res, orig, y):
    """
    Long-term debt by residual maturity bucket at the end of year y (bn), short-term debt deducted from <= 1 year.
    Returns (dict bucket -> amount, list of buckets estimated) or (None, None) if data are missing.
    """
    if y not in res.index or y not in orig.index:
        return None, None
    r, o = res.loc[y], orig.loc[y]
    if pd.isna(r.get('Y_LE1')) or pd.isna(r.get('Y1-5')) or pd.isna(o.get('Y_LE1')):
        return None, None
    b = {'Y_LE1': max(r['Y_LE1'] - o['Y_LE1'], 0.0), 'Y1-5': r['Y1-5']}
    if all(pd.notna(r.get(k)) for k in LONG_BUCKETS):
        return {**b, **{k: r[k] for k in LONG_BUCKETS}}, []
    total = r.get('TOTAL', np.nan) if pd.notna(r.get('TOTAL', np.nan)) else o.get('TOTAL', np.nan)
    after5 = r['Y_GT1'] - r['Y1-5'] if pd.notna(r.get('Y_GT1', np.nan)) else total - r['Y_LE1'] - r['Y1-5']
    if pd.isna(after5):
        return None, None
    b['Y_GT5'] = max(after5, 0.0)
    return b, LONG_BUCKETS


def pooled_long_shares(data, year, exclude=POOL_EXCLUDE):
    """
    Average shares of the 5-10, 10-30 and > 30 year buckets in debt maturing after 5 years, across countries with the
    full split in the given year (simple mean, excluding the countries in exclude).
    """
    res = data['residual']
    full = res.xs(year, level='year')[LONG_BUCKETS].dropna()
    full = full.loc[~full.index.isin(exclude) & (full.sum(axis=1) > 0)]
    return full.div(full.sum(axis=1), axis=0).mean()


def repayment_profiles(data, countries, reference_year, esm=None, end_year=2070, gt30_years=20):
    """
    Repayment profile of the long-term debt outstanding at the end of the base year (latest year with data before T),
    excluding ESM/EFSF loans, for the years T+1 to end_year (bn national currency). Repayments after end_year are
    added to end_year.

    Parameters:
        data (dict): fetch_maturity_data output.
        reference_year (int or dict): Reference year T (per country if dict).
        esm (DataFrame): ESM/EFSF repayments (year x ISO3, bn), deducted bucket by bucket.
        gt30_years (int): Number of years over which the > 30 year bucket is spread.

    Returns:
        (DataFrame year x ISO3, dict ISO3 -> {'base_year', 'estimated' (list of estimated buckets), 'note'})
        Countries without data are omitted.
    """
    buckets = {**BUCKETS, 'Y_GT30': (31, 30 + gt30_years)}
    profiles, info = {}, {}
    for c in countries:
        T = reference_year[c] if isinstance(reference_year, dict) else reference_year
        res, orig = _country(data['residual'], c), _country(data['original'], c)
        candidates = sorted((y for y in res.index if y < T), reverse=True)
        base, b, estimated = None, None, None
        for y in candidates:
            b, estimated = _buckets(res, orig, y)
            if b is not None:
                base = y
                break
        if base is None:
            continue
        if 'Y_GT5' in b:
            shares = pooled_long_shares(data, base)
            after5 = b.pop('Y_GT5')
            b.update({k: after5 * shares[k] for k in LONG_BUCKETS})

        # Deduct ESM/EFSF loans bucket by bucket and spread each bucket evenly over its years
        years = range(base + 1, base + buckets['Y_GT30'][1] + 1)
        profile = pd.Series(0.0, index=years)
        for k, (first, last) in buckets.items():
            span = range(base + first, base + last + 1)
            esm_k = esm[c].reindex(span).fillna(0).clip(lower=0).sum() if esm is not None and c in esm else 0.0
            amount = max(b[k] - esm_k, 0.0)
            profile.loc[span] += amount / len(span)
        tail = profile.loc[end_year + 1:].sum()
        profile = profile.loc[T + 1:end_year].copy()
        profile.loc[end_year] = profile.get(end_year, 0.0) + tail
        profiles[c] = profile

        note = (f'Eurostat gov_10dd_rmd/ggd, long-term debt at end-{base} by residual maturity, net of short-term debt and '
                f'ESM/EFSF loans, buckets spread evenly (> 30 years over {gt30_years} years); scaled to long-term debt '
                f'in T by the model')
        if estimated:
            note += '; 5-10/10-30/> 30 year split estimated with the average shares of countries with full data'
        info[c] = {'base_year': base, 'estimated': estimated, 'note': note}
    return pd.DataFrame(profiles), info


def add_repayment_profiles(data, countries, end_year=2070):
    """
    Add Eurostat repayment profiles (BOND_REPAYMENT) to Commission-mode country data (commission.build_country_commission),
    in place. These workbooks index nominal GDP (debt is not in bn) and have no separate ESM/EFSF series, so the profile
    covers all long-term debt and is scaled to the long-term debt of the workbook in T (DEBT_TOTAL x (1 - DEBT_ST_SHARE));
    the model scales profiles to that debt in any case (DsaModel._clean_bond_repayment). Countries without Eurostat data
    get no profile. Needs internet access; on failure the workbook is built without profiles.
    """
    import warnings
    try:
        maturity = fetch_maturity_data(countries)
    except Exception as err:
        warnings.warn(f'Eurostat maturity data not available ({err}); workbook built without BOND_REPAYMENT')
        return
    T = {c: data[c]['reference_year'] for c in countries}
    profiles, info = repayment_profiles(maturity, countries, T, esm=None, end_year=end_year)
    for c in profiles:
        d = data[c]
        series, T_c = d['series'], T[c]
        stock = series.loc[T_c, 'DEBT_TOTAL'] * (1 - d['params']['DEBT_ST_SHARE'])
        profile = profiles[c].dropna()
        profile = profile / profile.sum() * stock
        years = [y for y in profile.index if y in series.index]
        series.loc[years, 'BOND_REPAYMENT'] = profile[years].to_numpy()
        d['series_provenance'].loc[years, 'BOND_REPAYMENT'] = 'derived' if info[c]['estimated'] else 'api'
        d['series_sources']['BOND_REPAYMENT'] = (
            info[c]['note'].replace('net of short-term debt and ESM/EFSF loans', 'net of short-term debt')
            .replace('scaled to long-term debt in T by the model', 'scaled to long-term debt of this workbook in T'))
