# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Fiscal Rules             #
# ========================================================================================= #
#
# The FiscalRules class implements the EU fiscal rules (Regulation (EU) 2024/1263 and Regulation
# (EC) No 1467/97) on top of the DSA model. It is a mixin class: StochasticDsaModel inherits from
# FiscalRules and DsaModel, so that all methods are called on the model, e.g. model.find_spb_binding().
#
# 1. Integrated optimizer: find_spb_binding combines the DSA criteria, the EDP and the safeguards,
#    with the default rules (Darvas, Welslau and Zettelmeyer, 2024) or the rules of the Commission
#    prior guidance calculation sheets (see BINDING_RULES and COMMISSION_RULES).
# 2. Default rules: sequential optimization with front-loading of minimum steps.
# 3. Commission rules: constant annual adjustment with minimum steps added on top.
# 4. Excessive deficit procedure: EDP status, minimum steps and projected abrogation.
# 5. Debt sustainability safeguard and deficit resilience safeguard.
#
# Interface with the model: the methods use the projection and optimization methods of DsaModel and
# StochasticDsaModel (project, find_spb_deterministic, find_spb_stochastic, df) and read and write
# model attributes, in particular:
#   read:   adjustment_start, adjustment_end, adjustment_period, adjustment_start_year, start_year,
#           projection_period, params, country, d, ob, sb, spb_bca, pb, interest_ratio, spb_steps,
#           net_expenditure_growth, frontloading, df_deterministic_data
#   write:  rules, edp_status, edp_active, edp_steps, edp_period, edp_end, deficit_resilience_steps,
#           spb_target, spb_target_dict, pb_target_dict, binding_spb_target, binding_criterion,
#           binding_parameter_dict, binding_tables, df_dict, edp_binding, debt_safeguard_binding,
#           deficit_resilience_binding
#
# Author: Lennard Welslau
# Updated: 2026-09-30
# ========================================================================================= #

import warnings
import numpy as np
import pandas as pd
from classes.ResultsTables import display_tables, criterion_label, COUNTRY_NAMES

# EDP benchmark: minimum annual adjustment of 0.5% of GDP in structural balance terms (Regulation (EC) No 1467/97,
# Art. 3(4)). During the transition period 2025-2027, the benchmark may be adjusted for the increase in interest payments
# (Regulation (EU) 2024/1264, recital 23): until this year, the minimum step applies to the SPB.
EDP_SPB_TERMS_LAST_YEAR = 2027

# Rules for find_spb_binding: defaults (Darvas, Welslau and Zettelmeyer, 2024) and Commission prior guidance
BINDING_RULES = {
    'dsa_criteria': 'default',
    'stochastic_criteria': ['debt_declines', 'debt_below_60'],
    'edp': 'default',
    'edp_status': 'input',
    'debt_safeguard': 'default',
    'deficit_resilience': 'default',
    'frontloading': True,
    'grid': None,
}
COMMISSION_RULES = {
    'dsa_criteria': 'commission',
    'stochastic_criteria': ['debt_declines'],
    'edp': 'commission',
    'edp_status': 'input',
    'debt_safeguard': 'commission',
    'deficit_resilience': 'commission',
    'frontloading': False,
    'grid': 0.01,
}


class FiscalRules:
    """
    EU fiscal rules: integrated optimizer (find_spb_binding), EDP, debt sustainability safeguard and deficit
    resilience safeguard. Mixin class for StochasticDsaModel (see module header for the interface).
    """
    edp_status = 'input'  # EDP status in T: 'input' (input file), True, False or 'infer' (predicted by the model)
    edp_active = True  # EDP rules active (set by find_spb_binding)

    # ========================================================================================= #
    #   INTEGRATED OPTIMIZER                                                                  #
    # ========================================================================================= #

    def find_spb_binding(self,
                         edp=True,
                         debt_safeguard=True,
                         deficit_resilience=True,
                         stochastic=True,
                         print_results=True,
                         stochastic_criteria=None,
                         save_df=False,
                         rules=None,
                         **rule_options):
        """
        Find the structural primary balance that meets the DSA criteria, the EDP requirements and the safeguards.

        The default rules follow Darvas, Welslau and Zettelmeyer (2024). Each rule can be switched to the version used
        in the Commission prior guidance calculation sheets, individually (keyword arguments) or all at once
        (rules='commission'), see BINDING_RULES and COMMISSION_RULES:

            dsa_criteria        'default': per deterministic scenario, debt declines or stays below 60%, deficit below 3%,
                                stochastic criteria; 'commission': debt declines in all scenarios and stochastically, or
                                debt below 60% in all scenarios, deficit below 3%; technical information
                                (deficit and debt below 60% only) for countries with debt < 60% and deficit < 3% in T;
                                non-negative adjustment for the reference trajectory, at most -1 pp. per year otherwise
            stochastic_criteria list of 'debt_declines', 'debt_stable', 'debt_below_60'
            edp                 'default': minimum steps of 0.5 pp. while the deficit exceeds 3%, front-loaded;
                                'commission': min. 0.5 pp. step after a year with deficit above 3%, EDP abrogated after two
                                years with deficit below 3%. Both in SPB terms until 2027 and in SB terms from 2028
                                (EDP_SPB_TERMS_LAST_YEAR); None: no EDP
            edp_status          EDP status in T: 'input' (input file, as in the Commission prior guidance), True, False,
                                or 'infer' (predicted by the model from a projected deficit above 3%)
            debt_safeguard      'default': average decline from the year the EDP is projected to be abrogated (the year
                                after the deficit falls below 3%, if it stays below 3%, as in the Commission sheets) or T
                                to adjustment end, 1 pp. if debt in T > 90%, 0.5 pp. otherwise; 'commission': by
                                start-of-year debt band (> 90%, 60-90%), average over adjustment years outside the EDP;
                                None
            deficit_resilience  'default': steps raised (up to 0.4/0.25 pp.) until the structural deficit is below 1.5%
                                in the same year; 'commission': 0.4/0.25 pp. step after a year with a structural deficit
                                above 1.55%; None
            frontloading        True: EDP and deficit resilience steps front-load the adjustment, later steps are reduced
                                to keep the SPB target; False: minimum steps are added on top (Commission)
            grid                None: exact SPB target; 0.01: annual adjustment rounded up to 0.01 pp. (Commission)

        Arguments edp, debt_safeguard and deficit_resilience accept True (rule from rules), False, or a rule name.
        """
        # Set rules: defaults, preset, and individual options
        self.rules = self._set_rules(rules, rule_options, edp, debt_safeguard, deficit_resilience, stochastic_criteria)
        self.stochastic_criteria = self.rules['stochastic_criteria']

        # Remove results of earlier runs
        for attr in ['guidance', 'reference_trajectory', 'edp_binding', 'debt_safeguard_binding',
                     'deficit_resilience_binding', 'edp_min_steps', 'binding_criterion']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Model settings used by project() during the optimization, restored afterwards
        settings = {'frontloading': self.frontloading, 'edp_active': getattr(self, 'edp_active', True),
                    'edp_status': getattr(self, 'edp_status', 'input')}
        self.frontloading = self.rules['frontloading']
        self.edp_active = self.rules['edp'] is not None
        self.edp_status = self.rules['edp_status']

        # Default rules follow the sequential DWZ (2024) implementation, other rules a constant annual adjustment
        default = (all(self.rules[k] in (BINDING_RULES[k], None)
                       for k in ['dsa_criteria', 'edp', 'debt_safeguard', 'deficit_resilience'])
                   and self.rules['frontloading'] and self.rules['grid'] is None)
        try:
            if default:
                self._find_spb_binding_default(stochastic, print_results, save_df)
            else:
                self._find_spb_binding_rules(stochastic, print_results, save_df)
        finally:
            for k, v in settings.items():
                setattr(self, k, v)

    def _set_rules(self, rules, rule_options, edp, debt_safeguard, deficit_resilience, stochastic_criteria):
        """
        Combine default rules, preset and individual options.
        """
        if rules is None or rules == 'default':
            out = dict(BINDING_RULES)
        elif rules == 'commission':
            out = dict(COMMISSION_RULES)
        elif isinstance(rules, dict):
            out = {**BINDING_RULES, **rules}
        else:
            raise ValueError(f"Unknown rules '{rules}', use 'default', 'commission' or a dict")
        unknown = set(rule_options) - set(BINDING_RULES)
        if unknown:
            raise TypeError(f'Unknown rule options: {sorted(unknown)}')
        out.update(rule_options)
        for key, value in [('edp', edp), ('debt_safeguard', debt_safeguard), ('deficit_resilience', deficit_resilience)]:
            if value is False or value is None:
                out[key] = None
            elif isinstance(value, str):
                out[key] = value
        if stochastic_criteria is not None:
            out['stochastic_criteria'] = stochastic_criteria
        if not (isinstance(out['edp_status'], (bool, np.bool_)) or out['edp_status'] in ('input', 'infer')):
            raise ValueError(f"edp_status must be 'input', 'infer', True or False, got {out['edp_status']!r}")
        return out

    def _save_binding(self, edp, debt_safeguard, deficit_resilience, print_results):
        """
        Save binding SPB target, adjustment parameters and results tables.
        """
        # Save binding SPB and PB target
        self.spb_target_dict['binding'] = self.spb_bca[self.adjustment_end]
        self.pb_target_dict['binding'] = self.pb[self.adjustment_end]

        # Save binding parameters to reproduce adjustment path
        self.binding_parameter_dict['spb_steps'] = self.spb_steps
        self.binding_parameter_dict['spb_target'] = self.binding_spb_target
        self.binding_parameter_dict['criterion'] = self.binding_criterion

        # Save EDP and safeguards parameters
        if edp:
            self.binding_parameter_dict['edp_binding'] = self.edp_binding
            self.binding_parameter_dict['edp_steps'] = self.edp_steps
        if debt_safeguard:
            self.binding_parameter_dict['debt_safeguard_binding'] = self.debt_safeguard_binding
        if deficit_resilience:
            self.binding_parameter_dict['deficit_resilience_binding'] = self.deficit_resilience_binding
            self.binding_parameter_dict['deficit_resilience_steps'] = self.deficit_resilience_steps
        self.binding_parameter_dict['net_expenditure_growth'] = self.net_expenditure_growth[self.adjustment_start:self.adjustment_end+1]

        # Results tables, displayed if print_results
        self.binding_results()
        if print_results:
            display_tables(self.binding_tables)

        # Save dataframe
        if self.save_df:
            self.df_dict['binding'] = self.df(all=True)

    def binding_results(self, years_after=10):
        """
        Results of find_spb_binding as labelled tables:
            'Summary':          binding SPB target, annual adjustment, binding criterion and safeguards
            'SPB targets':      SPB at the end of the adjustment period required by each criterion
            'Adjustment path':  annual SPB steps, minimum steps, balances, debt and net expenditure growth by year
        """
        n, s, e = self.adjustment_period, self.adjustment_start, self.adjustment_end
        spb_start = self.spb_bca[s - 1]
        binding = self.spb_target_dict['binding']
        rules = getattr(self, 'rules', BINDING_RULES)
        changed = {k: v for k, v in rules.items() if BINDING_RULES.get(k) != v}
        rules_label = ('default' if not changed else 'default (no EDP)' if changed == {'edp': None}
                       else 'commission' if rules == COMMISSION_RULES
                       else ', '.join(f'{k}={v}' for k, v in changed.items()))
        dsa_keys = [k for k in self.spb_target_dict if k not in ['edp', 'debt_safeguard', 'deficit_resilience', 'binding']]
        dsa_target = (spb_start + n * self.binding_parameter_dict['annual_adjustment_dsa']
                      if 'annual_adjustment_dsa' in self.binding_parameter_dict
                      else max(self.spb_target_dict[k] for k in dsa_keys))

        summary = pd.Series({
            'Country': COUNTRY_NAMES.get(self.country, self.country),
            'Adjustment period': f'{self.adjustment_start_year}-{self.adjustment_end_year}',
            'Rules': rules_label,
            'Guidance': getattr(self, 'guidance', ''),
            'Binding criterion': criterion_label(self.binding_criterion),
            'SPB in T (% of GDP)': spb_start,
            'DSA-based SPB target (% of GDP)': dsa_target,
            'SPB at end of adjustment (% of GDP)': binding,
            'Average annual adjustment (pp.)': (binding - spb_start) / n,
            'Average net expenditure growth (%)': np.mean(self.net_expenditure_growth[s:e + 1]),
            'EDP abrogation year (projected)': self.edp_abrogation_year(),
            'EDP binding': getattr(self, 'edp_binding', None),
            'Debt safeguard binding': getattr(self, 'debt_safeguard_binding', None),
            'Deficit resilience binding': getattr(self, 'deficit_resilience_binding', None),
        }, name=self.country)
        summary = summary[[v is not None and v != '' for v in summary.values]]

        targets = pd.DataFrame({
            'SPB at end of adjustment': {criterion_label(k): v for k, v in self.spb_target_dict.items()},
            'Average annual adjustment': {criterion_label(k): (v - spb_start) / n for k, v in self.spb_target_dict.items()},
        })
        targets.index.name = 'Criterion'

        years = range(s - 1, min(e + years_after, self.projection_period - 1) + 1)
        edp_steps = np.full(self.projection_period, np.nan)
        resilience_steps = np.full(self.projection_period, np.nan)
        edp_min_steps = getattr(self, 'edp_min_steps', getattr(self, 'edp_steps', None))
        if edp_min_steps is not None:
            edp_steps[s:e + 1] = edp_min_steps
        if getattr(self, 'deficit_resilience_steps', None) is not None:
            resilience_steps[s:e + 1] = self.deficit_resilience_steps
        path = pd.DataFrame({
            'Annual SPB adjustment (pp.)': self.spb_bca_adjustment,
            'EDP minimum step (pp.)': edp_steps,
            'Deficit resilience minimum step (pp.)': resilience_steps,
            'SPB before change in ageing costs (% of GDP)': self.spb_bca,
            'Structural primary balance (% of GDP)': self.spb,
            'Overall balance (% of GDP)': self.ob,
            'Structural balance (% of GDP)': self.sb,
            'Debt (% of GDP)': self.d,
            'Net expenditure growth (%)': self.net_expenditure_growth,
        }, index=pd.Index(range(self.start_year, self.end_year + 1), name='Year')).iloc[list(years)]
        path.loc[path.index[0], ['Annual SPB adjustment (pp.)', 'Net expenditure growth (%)']] = np.nan
        path = path.dropna(axis=1, how='all')

        self.binding_tables = {'Summary': summary.to_frame(), 'SPB targets': targets, 'Adjustment path': path}
        return self.binding_tables

    # ========================================================================================= #
    #   DEFAULT RULES (Darvas, Welslau and Zettelmeyer, 2024)                                 #
    # ========================================================================================= #

    def _find_spb_binding_default(self, stochastic, print_results, save_df):
        """
        Binding SPB target with the default rules: DSA target, then EDP, debt safeguard and deficit resilience.
        """
        # EDP applies if the country is in EDP in T; with edp_status='infer', find_edp decides from the projected deficit
        edp = self.rules['edp'] is not None and (self.edp_status == 'infer' or self._in_edp_T())
        debt_safeguard = self.rules['debt_safeguard'] is not None
        deficit_resilience = self.rules['deficit_resilience'] is not None

        # Initiate spb_target and dataframe dictionary
        self.save_df = save_df
        self.spb_target_dict = {}
        self.pb_target_dict = {}
        self.binding_parameter_dict = {}
        if self.save_df:
            self.df_dict = {}

        # Run DSA and deficit criteria and project toughest under baseline assumptions
        self.project(spb_target=None, edp_steps=None) # clear projection
        self._run_dsa(stochastic=stochastic)
        self._get_binding()

        # Apply EDP (edp_end = adjustment_start - 2 marks no EDP, as in find_edp)
        if edp:
            self._apply_edp()
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 2
        self.edp_min_steps = np.copy(self.edp_steps)

        # Apply debt safeguard
        if debt_safeguard:
            self._apply_debt_safeguard()

        # Apply deficit resilience
        if deficit_resilience:
            self._apply_deficit_resilience()

        self._save_binding(edp, debt_safeguard, deficit_resilience, print_results)

    def _run_dsa(self, stochastic=True, criterion='all'):
        """
        Run the DSA: SPB targets for all deterministic criteria and the stochastic criterion (criterion='all'), or
        re-optimize a single criterion on the EDP path (used by _apply_edp).
        """
        # Define deterministic criteria
        deterministic_criteria_list = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g', 'deficit_reduction']

        # If all criteria, run all deterministic and stochastic
        if criterion == 'all':

            # Run all deterministic scenarios, skip if criterion not applicable
            for deterministic_criterion in deterministic_criteria_list:
                try:
                    self.find_spb_deterministic(criterion=deterministic_criterion)
                    self.spb_target_dict[deterministic_criterion] = self.spb_bca[self.adjustment_end]
                    self.pb_target_dict[deterministic_criterion] = self.pb[self.adjustment_end]
                    if self.save_df:
                        self.df_dict[deterministic_criterion] = self.df(all=True)
                except Exception as e:
                    raise ValueError(f'{deterministic_criterion} did not converge for {self.country}') from e

            # Run stochastic scenario, skip with a warning if not possible (e.g. lack of shock data)
            if stochastic:
                try:
                    self.find_spb_stochastic(stochastic_criteria=self.stochastic_criteria)
                    self.spb_target_dict['stochastic'] = self.spb_bca[self.adjustment_end]
                    self.pb_target_dict['stochastic'] = self.pb[self.adjustment_end]
                    if self.save_df:
                        self.df_dict['stochastic'] = self.df(all=True)
                except Exception as e:
                    warnings.warn(f'{self.country}: stochastic criterion skipped ({e})')

        # If specific criterion given for EDP optimization, run only one optimization
        else:
            if criterion in deterministic_criteria_list:
                self.find_spb_deterministic(criterion=criterion)

            elif criterion == 'stochastic':
                self.find_spb_stochastic(stochastic_criteria=self.stochastic_criteria)

            # Replace binding scenario
            self.binding_spb_target = self.spb_bca[self.adjustment_end]
            self.spb_target_dict['edp'] = self.spb_bca[self.adjustment_end]
            self.pb_target_dict['edp'] = self.pb[self.adjustment_end]

    def _get_binding(self):
        """
        Get binding SPB target and scenario from dictionary with SPB targets.
        """
        # Get binding SPB target and scenario from dictionary with SPB targets
        self.binding_spb_target = np.max(list(self.spb_target_dict.values()))
        self.binding_criterion = list(self.spb_target_dict.keys())[np.argmax(list(self.spb_target_dict.values()))]

        # Project under baseline assumptions
        self.project(spb_target=self.binding_spb_target, scenario=None)

    def _apply_edp(self):
        """
        Check if EDP is binding in binding scenario and apply if it is.
        """
        # Check if EDP binding, run DSA for periods after EDP and project new path under baseline assumptions
        self.find_edp(spb_target=self.binding_spb_target)

        if not np.all([np.isnan(self.edp_steps)]) and np.any([self.edp_steps >= self.spb_steps - 1e-8]):
            self.edp_binding = True
            self._run_dsa(criterion=self.binding_criterion)
            self.project(
                spb_target=self.binding_spb_target,
                edp_steps=self.edp_steps,
                scenario=None
                )
            if self.save_df:
                self.df_dict['edp'] = self.df(all=True)
        else:
            self.edp_binding = False

    def _apply_debt_safeguard(self):
        """
        Check if the debt sustainability safeguard is binding on the current path and apply it if it is.
        """
        # Keep the steps of the EDP period as minimum steps while re-optimizing the SPB target
        self.edp_steps[:self.edp_period] = self.spb_steps[:self.edp_period]

        # Debt safeguard binding for countries with high debt and insufficient average decline after EDP abrogation
        if (self.d[self.adjustment_start-1] >= 60
            and not self._debt_safeguard_criterion()):

            # Find the SPB target for the debt safeguard. Search above the current binding target: the safeguard is not monotonic in the SPB target, as a lower
            # target delays the abrogation of the EDP and hence the start of the safeguard
            self.spb_debt_safeguard_target = self.find_spb_deterministic(
                criterion='debt_safeguard', bounds=(self.binding_spb_target, 10))

            # If debt safeguard SPB target is higher than DSA target, save debt safeguard target
            # 1e-3 tolerance for edp search algo edge case, 1e-8 tolerance for floating point errors
            if self.spb_debt_safeguard_target > self.binding_spb_target + 1e-3 + 1e-8:
                self.debt_safeguard_binding = True
                self.binding_spb_target = self.spb_debt_safeguard_target
                self.spb_target = self.binding_spb_target
                self.binding_criterion = 'debt_safeguard'
                self.spb_target_dict['debt_safeguard'] = self.binding_spb_target
                self.pb_target_dict['debt_safeguard'] = self.pb[self.adjustment_end]
                if self.save_df:
                    self.df_dict['debt_safeguard'] = self.df(all=True)
            else:
                self.debt_safeguard_binding = False
        else:
            self.debt_safeguard_binding = False

    def _apply_deficit_resilience(self):
        """
        Apply deficit resilience safeguard after binding scenario.
        """
        # For countries with high deficit, find SPB target that brings and keeps deficit below 1.5%
        if (np.any(self.d[self.adjustment_start-1:self.adjustment_end+1] > 60)
            or self.ob[self.adjustment_start-1] < -3):
                self.find_spb_deficit_resilience()

        # Save results and print update
        if np.any([~np.isnan(self.deficit_resilience_steps)]):
            self.deficit_resilience_binding = True
            self.spb_target_dict['deficit_resilience'] = self.spb_bca[self.adjustment_end]
            self.pb_target_dict['deficit_resilience'] = self.pb[self.adjustment_end]
            self.binding_spb_target = self.spb_bca[self.adjustment_end]
            if self.save_df:
                self.df_dict['deficit_resilience'] = self.df(all=True)
        else:
            self.deficit_resilience_binding = False

    # ========================================================================================= #
    #   COMMISSION RULES (prior guidance calculation sheets)                                  #
    # ========================================================================================= #

    def _find_spb_binding_rules(self, stochastic, print_results, save_df, bounds=(-3, 3), tol=1e-4):
        """
        Binding SPB target for a constant annual adjustment: the DSA-based adjustment is the smallest constant annual
        step meeting the DSA criteria; the binding adjustment is the smallest step above it for which the path,
        including EDP and deficit resilience steps, meets the debt safeguard. Rules as set in self.rules.
        """
        rules = self.rules
        n, s, e = self.adjustment_period, self.adjustment_start, self.adjustment_end
        if hasattr(self, 'predefined_spb_steps'):
            raise NotImplementedError('predefined_spb_steps are only supported with the default rules')
        spb_start = self.spb_bca[s - 1]
        edp = rules['edp'] is not None
        debt_safeguard = rules['debt_safeguard'] is not None
        deficit_resilience = rules['deficit_resilience'] is not None
        self.save_df = save_df
        self.spb_target_dict, self.pb_target_dict, self.binding_parameter_dict = {}, {}, {}
        if self.save_df:
            self.df_dict = {}

        # 1. DSA-based annual adjustment
        a = self._dsa_annual_adjustments(stochastic, bounds, tol)
        a_dsa, criterion = self._combine_dsa_criteria(a)
        a_dsa_exact = a_dsa
        if rules['grid']:
            a_dsa = np.ceil(round(a_dsa / rules['grid'], 6)) * rules['grid']
        for k, v in a.items():
            self.spb_target_dict[k] = spb_start + n * v
        self.binding_spb_target = spb_start + n * a_dsa
        self.binding_criterion = criterion

        # 2. EDP with default rule: minimum steps from the EDP optimiser, kept fixed below
        self.edp_default_steps = np.full(n, np.nan)
        self.edp_period, self.edp_end = 0, s - 2
        if rules['edp'] == 'default' and (self.edp_status == 'infer' or self._in_edp_T()):
            self.find_edp(spb_target=self.binding_spb_target)
            self.edp_default_steps = np.copy(self.edp_steps)

        # 3. Debt safeguard on the path including EDP and deficit resilience steps. The safeguard is not monotonic in
        # the adjustment (years drop out of debt bands), so search upwards on a grid
        a_binding = a_dsa
        if debt_safeguard and self.reference_trajectory:
            step = rules['grid'] or 0.01
            while not self._debt_safeguard_met(self._rules_path(a_binding)) and a_binding < bounds[1]:
                a_binding = np.round(a_binding + step, 6)
            if not self._debt_safeguard_met(self._rules_path(a_binding)):
                warnings.warn(f'{self.country}: debt safeguard not met at the upper bound of {bounds[1]} pp. per year')
        self.debt_safeguard_binding = bool(a_binding > a_dsa)
        if self.debt_safeguard_binding:
            self.binding_criterion = 'debt_safeguard'
            self.spb_target_dict['debt_safeguard'] = spb_start + n * a_binding
        self._rules_path(a_binding)

        # Record which minimum steps were binding
        self.edp_min_steps = np.copy(self.edp_steps)
        self.edp_binding = bool(np.any(self.edp_steps > a_binding + 1e-8))
        self.deficit_resilience_binding = bool(np.any(self.deficit_resilience_steps > a_binding + 1e-8))
        self.spb_target = self.spb_bca[e]
        self.binding_spb_target = spb_start + n * a_binding
        self.guidance = 'reference trajectory' if self.reference_trajectory else 'technical information'
        self.binding_parameter_dict.update({
            'annual_adjustment_dsa': a_dsa,
            'annual_adjustment_dsa_exact': a_dsa_exact,
            'annual_adjustment': a_binding,
            'guidance': self.guidance,
        })
        self._save_binding(edp, debt_safeguard, deficit_resilience, print_results)

    def _dsa_annual_adjustments(self, stochastic, bounds, tol):
        """
        Smallest constant annual SPB step meeting each DSA criterion, from the deterministic and stochastic optimizers
        (find_spb_deterministic, find_spb_stochastic). Bounds and tolerance refer to the annual step.
        """
        n, s = self.adjustment_period, self.adjustment_start
        commission = self.rules['dsa_criteria'] == 'commission'
        self.reference_trajectory = bool(self.d[s - 1] > 60 or self.ob[s - 1] < -3) or not commission
        scenarios = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']
        spb_start = self.spb_bca[s - 1]
        target_bounds = (spb_start + n * bounds[0], spb_start + n * bounds[1])
        self.project(spb_target=None)  # linear path without EDP or deficit resilience minimum steps

        def step(criterion, debt_condition='declines_or_below_60'):
            try:
                target = self.find_spb_deterministic(criterion, bounds=target_bounds, tol=n * tol,
                                                     debt_condition=debt_condition)
            except ValueError as err:
                warnings.warn(f'{self.country}: {err}')
                target = target_bounds[1]
            return (target - spb_start) / n

        a = {'deficit_reduction': step('deficit_reduction')}
        if self.reference_trajectory:
            for sc in scenarios:
                a[f'debt_declines_{sc}'] = step(sc, 'declines')
                a[f'debt_below_60_{sc}'] = step(sc, 'below_60')
            if stochastic:
                try:
                    self.find_spb_stochastic(bounds=target_bounds, stochastic_criteria=self.rules['stochastic_criteria'])
                    a['stochastic'] = (self.spb_target - spb_start) / n
                except Exception as e:
                    warnings.warn(f'{self.country}: stochastic criterion skipped ({e})')
        else:
            a['debt_below_60_main_adjustment'] = step('main_adjustment', 'below_60')
        return a

    def _combine_dsa_criteria(self, a):
        """
        DSA-based annual adjustment from the criteria-specific adjustments. Returns (adjustment, binding criterion).
        """
        scenarios = ['main_adjustment', 'lower_spb', 'financial_stress', 'adverse_r_g']
        if not self.reference_trajectory:
            candidates = {k: a[k] for k in ['deficit_reduction', 'debt_below_60_main_adjustment']}
            candidates['floor'] = -1
        elif self.rules['dsa_criteria'] == 'commission':
            option_a = {k: a[k] for k in ['deficit_reduction', 'stochastic'] + [f'debt_declines_{sc}' for sc in scenarios]
                        if k in a}
            option_b = {k: a[k] for k in ['deficit_reduction'] + [f'debt_below_60_{sc}' for sc in scenarios]}
            candidates = option_a if max(option_a.values()) <= max(option_b.values()) else option_b
            candidates = {**candidates, 'floor': 0}
        else:
            # Per scenario, debt declines or stays below 60%
            candidates = {sc: min(a[f'debt_declines_{sc}'], a[f'debt_below_60_{sc}']) for sc in scenarios}
            candidates['deficit_reduction'] = a['deficit_reduction']
            if 'stochastic' in a:
                candidates['stochastic'] = a['stochastic']
        criterion = max(candidates, key=candidates.get)
        return candidates[criterion], criterion

    def _rules_path(self, a):
        """
        Project the path for a constant annual adjustment a including EDP and deficit resilience steps according to
        self.rules. Returns the EDP status by projection year (used by the debt safeguard).
        """
        n, s = self.adjustment_period, self.adjustment_start
        resilience_step = 0.4 if n <= 4 else 0.25
        spb_steps = np.full(n, a, dtype=np.float64)
        edp_steps = np.copy(self.edp_default_steps)
        resilience_steps = np.full(n, np.nan)
        self.project(spb_steps=spb_steps, edp_steps=edp_steps, deficit_resilience_steps=resilience_steps)

        # Minimum steps depending on the previous year (Commission rules), set year by year
        if 'commission' in (self.rules['edp'], self.rules['deficit_resilience']):
            for k in range(n):
                t = s + k
                status = self._edp_status()
                if self.rules['edp'] == 'commission' and status[t - 1] == 1 and self.ob[t - 1] <= -3.05:
                    edp_steps[k] = 0.5 + ((self.interest_ratio[t] - self.interest_ratio[t - 1])
                                          if self.start_year + t > EDP_SPB_TERMS_LAST_YEAR else 0)
                if self.rules['deficit_resilience'] == 'commission' and self.sb[t - 1] <= -1.55:
                    resilience_steps[k] = resilience_step
                self.project(spb_steps=spb_steps, edp_steps=edp_steps, deficit_resilience_steps=resilience_steps)

        # Default deficit resilience rule applied on the resulting path
        if (self.rules['deficit_resilience'] == 'default'
                and (np.any(self.d[s - 1:self.adjustment_end + 1] > 60) or self.ob[s - 1] < -3)):
            self.spb_target = self.spb_bca[self.adjustment_end]
            self.deficit_resilience_target = np.full(n, -1.5, dtype=float)
            self.deficit_resilience_step = resilience_step
            self.deficit_resilience_start = s
            self._deficit_resilience_loop_adjustment()
        return self._edp_status()

    def _edp_status(self):
        """
        EDP status by projection year. Commission rule: status in T from the input file, abrogated once the deficit is
        below 3% (tolerance 0.05) in the two preceding years. Default rule: in EDP until the year in which the EDP is
        projected to be abrogated (see DsaModel._debt_safeguard_start).
        """
        status = np.zeros(self.projection_period)
        if self.rules['edp'] == 'commission':
            edp_T = self.params.get('EXCESSIVE_DEFICIT_PROCEDURE', np.nan)
            if isinstance(self.edp_status, (bool, np.bool_)):
                edp_T = int(self.edp_status)
            elif self.edp_status == 'infer' or np.isnan(edp_T):
                if np.isnan(edp_T):
                    warnings.warn(f'{self.country}: EDP status missing in the input file, set from the deficit in T')
                edp_T = int(self.ob[0] <= -3.05)
            status[0] = int(edp_T)
            ob_before = self._fiscal_balance_before_start()
            for j in range(1, self.projection_period):
                ob_2 = ob_before if j == 1 else self.ob[j - 2]
                status[j] = 0 if (status[j - 1] == 0 or (ob_2 > -3.05 and self.ob[j - 1] > -3.05)) else 1
        elif self.rules['edp'] == 'default':
            abrogation = self._debt_safeguard_start()
            if abrogation >= self.adjustment_start:
                status[:abrogation + 1] = 1
        return status

    def _debt_safeguard_met(self, edp_status):
        """
        Debt sustainability safeguard for the current projection.

        Commission rule: over the adjustment years outside an EDP, debt declines on average by at least 1 pp. per year
        while above 90% of GDP and 0.5 pp. while between 60% and 90%, with debt bands classified by the debt ratio at
        the start of each year. Default rule: average decline from the year the EDP is projected to be abrogated (or T) to
        the end of the adjustment period by 1 pp. if debt in T exceeds 90% and 0.5 pp. otherwise, for debt above 60% in T
        (see DsaModel._debt_safeguard_start).
        """
        s, e = self.adjustment_start, self.adjustment_end
        if self.rules['debt_safeguard'] == 'default':
            return self.d[s - 1] < 60 or self._debt_safeguard_criterion()

        years = range(s, e + 1)
        change = np.array([self.d[t] - self.d[t - 1] for t in years])
        above90 = np.array([self.d[t - 1] > 90 for t in years])
        between = np.array([60 < self.d[t - 1] <= 90 for t in years])
        no_edp = np.array([edp_status[t] == 0 for t in years])
        if not above90.any() and not between.any():
            return True
        if not no_edp.any():
            return True
        ok = True
        if above90.any():
            sel = no_edp if not between.any() else (above90 & no_edp)
            ok &= change[sel].mean() < -1 if sel.any() else True
        if between.any():
            sel = between & no_edp
            ok &= change[sel].mean() < -0.5 if sel.any() else True
        return bool(ok)

    # ========================================================================================= #
    #   EXCESSIVE DEFICIT PROCEDURE                                                           #
    # ========================================================================================= #

    def _in_edp_T(self):
        """
        Whether the country is in EDP in T, depending on self.edp_status:
            'input' (default)  EDP status in the input file (EXCESSIVE_DEFICIT_PROCEDURE), as in the Commission prior
                               guidance; if missing (legacy workbooks), inferred as with 'infer'
            'infer'            predicted by the model: deficit above 3% of GDP in T, or EDP steps found by find_edp
            True / False       set by the user
        """
        status = getattr(self, 'edp_status', 'input')
        if isinstance(status, (bool, np.bool_)):
            return bool(status)
        flag = self.params.get('EXCESSIVE_DEFICIT_PROCEDURE', np.nan)
        if status == 'input' and not np.isnan(flag):
            return flag == 1
        return getattr(self, 'edp_period', 0) > 0 or self.ob[self.adjustment_start - 1] < -3

    def _edp_applies(self):
        """
        Whether the EDP applies: EDP rules active (see find_spb_binding) and country in EDP in T (see _in_edp_T).
        """
        return getattr(self, 'edp_active', True) and self._in_edp_T()

    def edp_abrogation_year(self):
        """
        Year in which the EDP is projected to be abrogated on the current path, None if no EDP applies or the EDP is
        not abrogated during the projection. Commission EDP rule: first year after the status turns to zero, minus one
        (see _edp_status); default rule: DsaModel._debt_safeguard_start.
        """
        rules = getattr(self, 'rules', BINDING_RULES)
        if rules.get('edp') == 'commission':
            status = self._edp_status()
            if status[0] == 0:
                return None
            ended = np.where(status == 0)[0]
            return int(self.start_year + ended[0] - 1) if len(ended) else None
        if rules.get('edp') is None or not self._edp_applies():
            return None
        last = self.projection_period - 1
        abrogation = self._edp_abrogation_index(last=last)
        return int(self.start_year + abrogation) if abrogation <= last else None

    def _edp_abrogation_index(self, last):
        """
        Index of the year A in which the EDP is projected to be abrogated: the first year from T on (T at the earliest)
        with a deficit below 3% in A-1 and A. Searched up to index last; returns last + 1 if not abrogated by then.
        """
        s = self.adjustment_start
        ob_before = self.ob[s - 2] if s >= 2 else self._fiscal_balance_before_start()
        for A in range(s - 1, last + 1):
            ob_previous = self.ob[A - 1] if A >= 1 else ob_before
            if self.ob[A] >= -3 and ob_previous >= -3:
                return A
        return last + 1

    def _fiscal_balance_before_start(self):
        """
        Fiscal balance in the year before the start year from the input data (checked in _load_input_data).
        """
        return float(self.df_deterministic_data.loc[self.start_year - 1, 'FISCAL_BALANCE'])

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

        # Define EDP threshold, set to 3% of GDP
        self.edp_target = -3

        # If deficit excessive, increase spb by 0.5 annually until deficit below 3%
        if self.ob[self.adjustment_start] < self.edp_target:

            # Set start indices for SPB and SB parts of the EDP: SPB terms until EDP_SPB_TERMS_LAST_YEAR, SB terms after
            self.edp_spb_index = 0
            self.edp_sb_index = int(np.clip(EDP_SPB_TERMS_LAST_YEAR + 1 - self.adjustment_start_year, 0, self.adjustment_period))

            # Calculate EDP adjustment steps for spb, sb, and final periods
            self._calc_edp_spb()
            self._calc_edp_sb()
            self._calc_edp_end(spb_target=spb_target)

        # If excessive deficit in year before adjustment start, set edp_end to year before adjustment start
        elif self.ob[self.adjustment_start - 1] < self.edp_target:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 1

        # If deficit not excessive, set EDP period to 0
        else:
            self.edp_period = 0
            self.edp_end = self.adjustment_start - 2

    def _save_edp_period(self):
        """
        Saves EDP period and end period
        """
        self.edp_period = np.where(~np.isnan(self.edp_steps))[0][-1] + 1
        self.edp_end = self.adjustment_start + self.edp_period - 1

    def _calc_edp_spb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral primary balance adjustment
        """
        # Loop for SPB part of EDP: min. 0.5 spb adjustment while deficit > 3 and in spb adjustmet period
        while (self.ob[self.adjustment_start + self.edp_spb_index] <= self.edp_target
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

    def _calc_edp_sb(self):
        """
        Calculate EDP adjustment steps ensuring minimum strucutral balance adjustment
        """
        # Loop for SB balance part of EDP: min. 0.5 ob adjustment while deficit > 3 and before last period
        while (self.ob[self.adjustment_start + self.edp_sb_index] <= self.edp_target
                and self.edp_sb_index + 1 <= self.adjustment_period):

            # If sb adjustment is less than 0.5, increase by 0.001
            while (self.sb[self.adjustment_start + self.edp_sb_index]
                   - self.sb[self.adjustment_start + self.edp_sb_index - 1] < 0.5):

                # Initiate sb step at current adjustment_step value, increase by 0.001
                self.edp_steps[self.edp_sb_index] = self.spb_steps[self.edp_sb_index]
                self.edp_steps[self.edp_sb_index] += 0.001

                # Project using last periods SPB as target, move to next period
                self.project(
                    spb_target=self.spb_target,
                    edp_steps=self.edp_steps
                )

            # If sb adjustment reaches min. 0.5, move to next period
            if self.sb[self.adjustment_start + self.edp_sb_index] - self.sb[self.adjustment_start + self.edp_sb_index - 1] >= 0.5:

                # set edp step to spb step in this period to ensure EDP recorded even in cases where step exceeds 0.5
                self.edp_steps[self.edp_sb_index] = self.spb_steps[self.edp_sb_index]
                self.edp_sb_index += 1
                self._save_edp_period()

    def _calc_edp_end(self, spb_target):
        """
        Calculate EDP adjustment steps or SPB target ensuring deficit below 3% at adjustment end
        """
        # If EDP lasts until penultimate adjustmet period, increase EDP steps to ensure deficit < 3
        if self.edp_period == self.adjustment_period:
            while self.ob[self.adjustment_end] < self.edp_target:

                # Aim for linear adjustment path by increasing smallest EDP steps first
                min_edp_steps = np.min(self.edp_steps[~np.isnan(self.edp_steps)])
                min_edp_indices = np.where(self.edp_steps == min_edp_steps)[0]
                self.edp_steps[min_edp_indices] += 0.0001
                self.project(
                    spb_target=self.spb_target,
                    edp_steps=self.edp_steps
                )
                self._save_edp_period()

        # If last EDP period has deficit < 3, we do not impose additional adjustment
        if self.ob[self.adjustment_start - 1 + self.edp_period] >= self.edp_target:
            self.edp_steps[self.edp_sb_index:] = np.nan
            self._save_edp_period()

        # If no spb_target was specified, calculate to ensure deficit < 3 until adjustment end
        if spb_target is None:
            print('No SPB target specified, calculating to ensure deficit < 3')
            while np.any(self.ob[self.edp_end + 1:self.adjustment_end + 1] <= self.edp_target):
                self.spb_target += 0.001
                self.project(spb_target=self.spb_target, edp_steps=self.edp_steps)

    # ========================================================================================= #
    #   DEBT SUSTAINABILITY SAFEGUARD                                                         #
    # ========================================================================================= #

    def _debt_safeguard_start(self):
        """
        Base year (index) of the debt safeguard, Article 7(2) of Regulation (EU) 2024/1263: the year in which the EDP is
        projected to be abrogated, or T if no EDP applies, whichever is later. The decline is measured from the debt ratio
        in this year to the end of the adjustment period.

        As in the Commission prior guidance sheets, a country is in EDP in T if flagged in the input file or if its deficit
        exceeds 3% of GDP, and the EDP is abrogated in the first year A in which the deficit was below 3% in A-1 (outturn)
        and remains below 3% in A. Example: deficit above 3% in T, below 3% from T+1: abrogation and base year T+2.
        """
        s, e = self.adjustment_start, self.adjustment_end
        start = self._edp_abrogation_index(last=e) if self._edp_applies() else s - 1
        if hasattr(self, 'predefined_spb_steps'):
            start = max(start, s + len(self.predefined_spb_steps) - 1)
        return start

    def _debt_safeguard_criterion(self):
        """
        Checks the debt safeguard criterion: average annual decline from the year the EDP is projected to be abrogated
        (or T) to the end of the adjustment period, 1 pp. if debt in T exceeds 90% and 0.5 pp. otherwise.
        """
        debt_safeguard_decline = 1 if self.d[self.adjustment_start - 1] > 90 else 0.5
        debt_safeguard_start = self._debt_safeguard_start()
        if debt_safeguard_start >= self.adjustment_end:
            return True

        return (self.d[debt_safeguard_start] - self.d[self.adjustment_end]
                >= debt_safeguard_decline * (self.adjustment_end - debt_safeguard_start))

    # ========================================================================================= #
    #   DEFICIT RESILIENCE SAFEGUARD                                                          #
    # ========================================================================================= #

    def find_spb_deficit_resilience(self):
        """
        Apply the deficit resilience targets that sets min. annual spb adjustment if structural deficit exceeds 1.5%.
        """
        # Initialize deficit_resilience_steps
        self.deficit_resilience_steps = np.full((self.adjustment_period,), np.nan, dtype=np.float64)

        # Define structural deficit target
        self.deficit_resilience_target = np.full(self.adjustment_period, -1.5, dtype=float)

        # Define deficit resilience step size
        if self.adjustment_period <= 4:
            self.deficit_resilience_step = 0.4
        else:
            self.deficit_resilience_step = 0.25

        # Project baseline
        self.project(
            spb_target=self.spb_target,
            edp_steps=self.edp_steps,
            deficit_resilience_steps=self.deficit_resilience_steps
        )

        self.deficit_resilience_start = self.adjustment_start

        # Run deficit resilience loop
        self._deficit_resilience_loop_adjustment()

        return self.spb_bca[self.adjustment_end]

    def _deficit_resilience_loop_adjustment(self):
        """
        Loop for adjustment period violations of deficit resilience
        """
        for t in range(self.deficit_resilience_start, self.adjustment_end + 1):
            if ((self.d[t] > 60 or self.ob[t] < -3)
                and self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start]
                and self.spb_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8):  # 1e-8 tol for floating point errors
                self.deficit_resilience_steps[t - self.adjustment_start] = self.spb_steps[t - self.adjustment_start]
                while (self.sb[t] <= self.deficit_resilience_target[t - self.adjustment_start]
                       and self.deficit_resilience_steps[t - self.adjustment_start] < self.deficit_resilience_step - 1e-8):  # 1e-8 tol for floating point errors
                    self.deficit_resilience_steps[t - self.adjustment_start] += 0.001
                    self.project(
                        spb_target=self.spb_target,
                        edp_steps=self.edp_steps,
                        deficit_resilience_steps=self.deficit_resilience_steps
                    )
