# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Group DSA Class          #
# ========================================================================================= #
#
# The GroupDsaModel class collects DSA models for several countries, runs them (in parallel using
# concurrent.futures for the computationally demanding optimizers) and collects and saves the results:
#   find_spb_binding / find_spb_stochastic  run the optimizers for all countries
#   summary                                  results for all countries as one table
#   save_results                             one readable Excel workbook (see ResultsTables)
#   save_dfs                                 all raw model variables (one sheet per country and scenario)
#   df_avg                                   GDP-weighted averages across countries
#
# Author: Lennard Welslau
# Updated: 2026-09-29
# ========================================================================================= #

# Import libraries and modules
import os
import time
import warnings
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from data_pipeline import REPO_ROOT, EU27
from data_pipeline.sources import EURO_ADOPTION
from classes.ResultsTables import display_tables, key_variables, write_results_workbook


class GroupDsaModel:
    def __init__(self, countries, **dsa_params):
        """
        Initialize the GroupDsaModel instance.

        Parameters:
            countries (list): List of country codes.
            **dsa_params: Additional keyword arguments to pass to the DSA model.
        """

        # Store input parameters
        self.countries = countries
        self.dsa_params = dsa_params
        self._today = time.strftime('%Y_%m_%d')
        self.failed = {}  # countries for which a task failed, with the error message

        # Dictionaries to hold DSA model instances and results by country
        self.models = {c: {} for c in countries}
        self.results = {c: {} for c in countries}

        # Initialize DSA models for each country
        self._init_models()
        self.input_file = next(iter(self.models.values())).input_file

    def _init_models(self):
        """
        Instantiate a DSA model for each country.

        A local import is used here to avoid circular import issues.
        """
        from classes import StochasticDsaModel as DSA
        for country in self.countries:
            self.models[country] = DSA(country=country, **self.dsa_params)

    def update_params(self, update_params):
        """
        Update the attributes of each DSA model.

        Parameters:
            **update_params: Dictionaries of parameters to update for each country.
        """
        for country in self.countries:
            for attr, value in update_params.items():
                setattr(self.models[country], attr, value)

    def project(self, store_as=False, discard_models=False, **project_params):
        """
        Run the projection step for each DSA model.

        Parameters:
            store_as (str): Key to use when storing to the result dictionary.
            discard_models (bool): If True, delete the model from memory after processing.
            **project_params: dictionaries of parameters to pass to the project method
            for each country.
        """
        for country in self.countries:
            # Get the parameters for the current country
            params = project_params.get(country, {})
            self.models[country].project(**params)

            # Store results if required
            if store_as:
                self.results[country]['spb_target_dict'] = {store_as: self.models[country].spb_target}
                self.results[country]['df_dict'] = {store_as: self.models[country].df(all=True)}
            if discard_models:
                del self.models[country]

    def find_spb_binding(self, edp_countries=None, parallel=True, max_workers=None, discard_models=False, **find_binding_params):
        """
        Run the binding SPB analysis for each country.

        Parameters:
            edp_countries (list or str): Countries for which the EDP is applied. 'input': countries with
                             EXCESSIVE_DEFICIT_PROCEDURE = 1 in the input workbook. None (default): all countries,
                             the EDP then applies wherever the deficit exceeds 3% of GDP.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes (default: number of CPUs).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_binding_params: dict of additional parameters for find_spb_binding.
        """
        find_binding_params.setdefault('print_results', False)
        if edp_countries == 'input':
            edp_countries = [c for c, m in self.models.items() if m.params.get('EXCESSIVE_DEFICIT_PROCEDURE', 0) == 1]
            print(f'Countries in EDP according to input data: {edp_countries}')
        elif edp_countries is None:
            edp_countries = list(self.models)
        tasks = [(country, model, edp_countries, find_binding_params) for country, model in list(self.models.items())]
        print(f'Running find_spb_binding for {len(tasks)} countries (parallel={parallel})')
        self._run_tasks(_find_spb_binding_task, tasks, parallel, max_workers, discard_models)

    def _run_tasks(self, task_function, tasks, parallel, max_workers, discard_models):
        """
        Run task_function on each task, in parallel or sequentially, and store the returned results by country.
        Countries for which the task fails are listed in self.failed and reported at the end.
        """
        def store(country, results):
            self.results[country].update(results)
            self.failed.pop(country, None)
            if discard_models:
                del self.models[country]

        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(task_function, task): task[0] for task in tasks}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        store(*future.result())
                    except Exception as e:
                        self.failed[futures[future]] = repr(e)
        else:
            for task in tqdm(tasks):
                try:
                    store(*task_function(task))
                except Exception as e:
                    self.failed[task[0]] = repr(e)
        if self.failed:
            warnings.warn(f'Tasks failed for {len(self.failed)} countries (see .failed), results exclude them: '
                          + '; '.join(f'{c}: {e}' for c, e in self.failed.items()))

    def find_spb_stochastic(self, store_as='stochastic', parallel=True, max_workers=None, discard_models=False, **find_stochastic_params):
        """
        Run the stochastic SPB analysis for each country.

        Parameters:
            store_as (str): Key to use when storing the result.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes (default: number of CPUs).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_stochastic_params: dict of additional parameters for find_spb_stochastic.
        """
        tasks = [(country, model, store_as, find_stochastic_params) for country, model in list(self.models.items())]
        print(f'Running find_spb_stochastic for {len(tasks)} countries (parallel={parallel})')
        self._run_tasks(_find_spb_stochastic_task, tasks, parallel, max_workers, discard_models)

    def project_fr(self, store_as=False, discard_models=False, **fr_params):
        """
        Run the fiscal rule analysis for each DSA model.

        Parameters:
            store_as (str): Key to use when storing to the results dictionary.
            discard_models (bool): If True, delete the model from memory after processing.
            **fr_params: dictionaries of parameters to pass to the project_fr method
            for each country.
        """
        for country in self.countries:
            # Get the parameters for the current country
            params = fr_params.get(country, {})
            self.models[country].project_fr(**params)

            # Store results if required
            if store_as:
                self.results[country]['df_dict'] = {store_as: self.models[country].df(all=True)}
            if discard_models:
                del self.models[country]

    def summary(self, table='Summary', display=False):
        """
        Results of find_spb_binding for all countries as one table.

        Parameters:
            table (str): 'Summary' (binding target, adjustment and criterion by country),
                         'SPB targets' (SPB at the end of adjustment required by each criterion) or
                         'Adjustment path' (annual path by country and year).
            display (bool): If True, display the table (HTML in notebooks).
        """
        tables = {c: self.results[c]['tables'][table] for c in self.countries if 'tables' in self.results[c]}
        if not tables:
            raise ValueError('No results: run find_spb_binding first')
        if table == 'Summary':
            df = pd.concat(tables.values(), axis=1).T
        elif table == 'SPB targets':
            df = pd.concat({c: t['SPB at end of adjustment'] for c, t in tables.items()}, axis=1).T
        else:
            df = pd.concat(tables, names=['Country'])
        df.index.name = df.index.name or 'Country'
        if display:
            display_tables({table: df})
        return df

    def save_results(self, folder=None, file=None, scenarios=('binding', 'no_policy_change'), variables=None):
        """
        Save results to one readable Excel workbook (see ResultsTables.write_results_workbook):
            README, Summary, SPB targets, Adjustment paths, Debt <scenario> and one sheet per country
            with key variables by year for each of the given scenarios.

        Parameters:
            folder (str): Folder under output/. Defaults to today's date.
            file (str): File name. Defaults to results_<adjustment period>y.xlsx.
            scenarios (tuple): Scenarios from the stored model DataFrames to include in the country sheets.
            variables (list): Model variables for the country sheets (default: ResultsTables.KEY_VARIABLES).
        """
        folder_path = os.path.join(REPO_ROOT, 'output', folder or self._today)
        os.makedirs(folder_path, exist_ok=True)
        period = self.dsa_params.get('adjustment_period', 4)
        file_path = os.path.join(folder_path, file or f'results_{period}y.xlsx')

        results = {}
        for country in self.countries:
            res = self.results[country]
            tables = res.get('tables', {})
            results[country] = {
                'summary': tables['Summary'].iloc[:, 0] if 'Summary' in tables else None,
                'targets': tables.get('SPB targets'),
                'path': tables.get('Adjustment path'),
                'scenarios': {s: key_variables(res['df_dict'][s], variables)
                              for s in scenarios if s in res.get('df_dict', {})},
            }
            results[country] = {k: v for k, v in results[country].items() if v is not None}

        meta = {
            'Created': time.strftime('%Y-%m-%d %H:%M'),
            'Countries': ', '.join(self.countries),
            'Adjustment period': f'{period} years',
            'Input file': os.path.basename(str(self.input_file)),
            'Model parameters': ', '.join(f'{k}={v}' for k, v in self.dsa_params.items()),
        }
        write_results_workbook(file_path, results, meta)
        print(f'Results saved to {os.path.relpath(file_path, REPO_ROOT)}')
        return file_path

    def save_dfs(self, folder=None, file=None):
        """
        Save all raw model DataFrames stored in the results (all model variables, one sheet per
        country and scenario). For a readable summary of results use save_results.

        Parameters:
            folder (str): Output folder path. If None, a folder based on the current date is created.
            file (str): Filename for the Excel output.
        """
        folder_path = os.path.join(REPO_ROOT, 'output', folder or self._today)
        os.makedirs(folder_path, exist_ok=True)
        period = self.dsa_params.get('adjustment_period', 4)
        file_path = os.path.join(folder_path, file or f'timeseries_{period}y.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            for country, res in self.results.items():
                for scenario, df in res.get('df_dict', {}).items():
                    # Limit the sheet name to 31 characters
                    df.to_excel(writer, sheet_name=f'{country}_{period}_{scenario}'[:31])
        print(f'DataFrames saved to {os.path.relpath(file_path, REPO_ROOT)}')

    def get_country_model(self, country):
        """
        Return the dictionary of DSA models for the given country.

        Parameters:
            country (str): The country code (e.g., 'AUT').

        Returns:
            dict: Dictionary of DSA models for the specified country.
        """
        return self.models.get(country.upper(), {})

    def df_avg(self, countries=None, scenario='binding'):
        """
        Calculate average of model DataFrames:
        - For non-absolute (lowercase) attributes: compute the weighted average
            using (ngdp * exr_eur) as weights (row-wise).
        - For absolute (non-lowercase) attributes: simply compute the row-wise sum.

        Parameters:
            countries (list or str): Countries to average, 'EU', 'EA' (euro area members in the first
                                     projection year) or None (default: all countries of the group).
            scenario (str): The scenario key to extract from each country's df_dict.

        Returns:
            avg_df (pd.DataFrame): Aggregated DataFrame with the same index and columns,
                                   where each cell is the weighted average or sum.
        """
        # Specify the list of countries to process.
        if countries == 'EU':
            countries = EU27
        elif countries == 'EA':
            first_year = next(iter(self.results.values()))['df_dict'][scenario].index.get_level_values('y')[0]
            countries = [c for c in EU27 if EURO_ADOPTION.get(c, 9999) <= first_year]
        elif countries is None:
            countries = self.countries

        # Countries with results for the scenario
        countries = [c for c in countries if scenario in self.results.get(c, {}).get('df_dict', {})]
        scenario_df_dict, weight_series_dict = {}, {}

        # Loop over each country to extract the desired scenario DataFrame and compute weights.
        for country in countries:
            # Access the dictionary containing DataFrames for the current country and period.
            df_dict = self.results[country]['df_dict']
            # Check if the specified scenario exists for this country.
            if scenario in df_dict:
                df = df_dict[scenario]
                # Store the DataFrame for this country.
                scenario_df_dict[country] = df
                # Compute the weight for each row as the product of 'ngdp' and 'exr_eur'.
                weight_series_dict[country] = df['ngdp'] * df['exr_eur']

        # Compute the total weight per row across all countries.
        gdp_total = sum(weight_series_dict[country] for country in weight_series_dict)

        # Create an empty DataFrame for the aggregated results, using the index from one of the DataFrames.
        avg_df = pd.DataFrame(index=df.index)

        # Iterate over each column in the DataFrame.
        for attr in df.columns:
            try:
                if attr.islower():
                    # For lowercase columns, compute the weighted average row-wise.
                    # Multiply each country's column values by its corresponding weight,
                    # then sum these products for each row, and finally divide by the total weight.
                    avg_df[attr] = sum(
                        scenario_df_dict[country][attr] * weight_series_dict[country]
                        for country in scenario_df_dict
                        ) / gdp_total
                else:
                    # For non-lowercase columns, compute the row-wise sum across countries.
                    avg_df[attr] = sum(
                        scenario_df_dict[country][attr]
                        for country in scenario_df_dict
                        )
            except Exception as e:
                # Print an error message if any issue occurs during the calculation for the column.
                print(f"Error in calculating average for attribute: {attr}: {e}")

        return avg_df

# ========================================================================================= #
#                     MODULE-LEVEL TASKS (picklable for parallel processing)               #
# ========================================================================================= #

def _find_spb_binding_task(args):
    """
    Helper function to run the binding SPB analysis for one model.

    Expected arguments:
    - country: the country code (string)
    - model: the model instance
    - edp_countries: list of countries for which EDP should be applied
    - find_binding_params: dict of additional parameters for find_spb_binding
    """
    country, model, edp_countries, find_binding_params = args

    # Run the binding SPB analysis, applying the EDP only to the given countries
    model.find_spb_binding(save_df=True, edp=country in edp_countries, **find_binding_params)

    # Extract the results from the model, with fanchart data for the binding path if stochastic
    results = {
        'spb_target_dict': model.spb_target_dict,
        'df_dict': model.df_dict,
        'binding_parameter_dict': model.binding_parameter_dict,
        'tables': model.binding_tables,
    }
    if find_binding_params.get('stochastic', True):
        model.fanchart(plot=False)
        results['df_fanchart'] = model.df_fanchart

    # Add the 'no_policy_change' projection, then return the model to the binding path
    model.project()
    results['df_dict']['no_policy_change'] = model.df(all=True)
    model.project(spb_steps=model.binding_parameter_dict['spb_steps'], scenario=None)

    return country, results

def _find_spb_stochastic_task(args):
    """
    Helper function to run the stochastic SPB analysis for one model.

    Expected arguments:
    - country: the country code (string)
    - model: the model instance
    - store_as: key to use for saving the result in the dictionary
    - find_stochastic_params: dict of additional parameters for find_spb_stochastic
    """
    country, model, store_as, find_stochastic_params = args
    model.find_spb_stochastic(**find_stochastic_params)
    model.fanchart(plot=False)
    return country, {
        'spb_target_dict': {store_as: model.spb_target},
        'df_dict': {store_as: model.df(all=True)},
        'df_fanchart': model.df_fanchart,
    }

