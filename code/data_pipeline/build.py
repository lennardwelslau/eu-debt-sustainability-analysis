# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Build Input Workbook     #
# ========================================================================================= #
#
# Entry point of the data pipeline. Builds the Excel input workbook read by the DSA model.
#
#   build_inputs(mode='commission')  All inputs from the Commission prior guidance calculation
#                                    sheets. Used to replicate and validate the Commission results.
#   build_inputs(mode='api')         Up-to-date data from public APIs (AMECO, ECB, ESM) and
#                                    published reference files (Ageing Report, DSM, OGWG); market
#                                    assumptions default to the Commission prior guidance sheets.
#
# Author: Lennard Welslau
# ========================================================================================= #

import datetime
import warnings

from .schema import EU27, INPUT_DIR
from .workbook import write_workbook
from . import commission


def build_inputs(mode='api', countries=EU27, vintage=None, prior_guidance='latest', out_file=None,
                 download=True, overrides=None, shocks=True, **kwargs):
    """
    Build the input workbook.

    Parameters:
        mode (str): 'api' (up-to-date public data) or 'commission' (Commission prior guidance sheets only).
        countries (list): ISO3 country codes.
        vintage (str): Label used in the file name, defaults to 'commission_<prior_guidance>' or today's date.
        prior_guidance (str): Which Commission prior guidance sheet to use per country, 'latest' (default) or '2024'.
        out_file (str/Path): Output path, defaults to data/InputData/dsa_inputs_<vintage>.xlsx.
        download (bool): Download missing Commission sheets before building.
        overrides (DataFrame): Optional table of user overrides (COUNTRY, CODE, YEAR, VALUE, COMMENT)
            written to the Overrides sheet.
        shocks (bool or dict): True fetches historical shock data for the stochastic model (Eurostat, OECD, ECB),
            a dict of shock tables (e.g. from a previous build) is written as is, False omits them.
        **kwargs: Passed to the mode-specific builder.

    Returns:
        Path of the workbook.
    """
    if download:
        commission.download_prior_guidance(countries, verbose=False)
    sheets = commission.load_prior_guidance(countries, vintage=prior_guidance)

    if mode == 'commission':
        vintage = vintage or f'commission_{prior_guidance}'
        data = {c: commission.build_country_commission(sheets[c]) for c in countries}
        description = (f'All inputs taken from the European Commission prior guidance calculation sheets '
                       f'({prior_guidance} vintage). Use to replicate the Commission DSA; start the model in each '
                       f"country's reference year T.")
    elif mode == 'api':
        from . import api_build
        vintage = vintage or datetime.date.today().strftime('%Y_%m')
        data, description = api_build.build_countries(countries, sheets, **kwargs)
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'api' or 'commission'")

    reference_years = {c: d['reference_year'] for c, d in data.items()}
    if len(set(reference_years.values())) > 1:
        warnings.warn(f'Reference years differ across countries: {reference_years}')
    meta = {
        'vintage': vintage,
        'mode': mode,
        'reference_year': min(reference_years.values()),
        'reference_years': reference_years,
        'description': description,
        'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    if shocks:
        from .shocks import build_shocks, SHOCK_SOURCES
        if shocks is True:
            shocks = build_shocks(countries)
        meta['shock_sources'] = SHOCK_SOURCES
    out_file = out_file or INPUT_DIR / f'dsa_inputs_{vintage}.xlsx'
    write_workbook(out_file, data, meta, overrides=overrides, shocks=shocks or None)
    return out_file
