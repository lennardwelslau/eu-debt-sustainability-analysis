# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Group DSA Class          #
# ========================================================================================= #
#
# This class collects several DSA model instances for different countries. Class methods 
# run the DSA models and save the results. Computationally demanding tasks can be run in 
# parallel using concurrent.futures.
#
# Author: Lennard Welslau
# Updated: 2025-02-10
# ========================================================================================= #

# Import libraries and modules
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# Set plotting style
plt.rcParams.update({
    'axes.grid': True,
    'grid.color': 'black',
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'font.size': 14
})

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

        # Dictionaries to hold DSA model instances and results by country
        self.models = {c: {} for c in countries}
        self.results = {c: {} for c in countries}

        # Initialize DSA models for each country 
        self._init_models()

    def _init_models(self):
        """
        Instantiate a DSA model for each country.

        A local import is used here to avoid circular import issues.
        """
        from classes import StochasticDsaModel as DSA
        for country in self.countries:
                # Create a copy of the DSA parameters and update with country
                model_params = self.dsa_params.copy()
                model_params['country'] = country
                self.models[country] = DSA(**model_params)

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
                    self.results[country]['spb_target_dict'] = {
                        store_as: self.models[country].spb_target
                        }
                    self.results[country]['df_dict'] = {
                        store_as: self.models[country].df(all=True)
                        }
                if discard_models:
                    del self.models[country]

    def find_spb_binding(self, edp_countries=[], parallel=True, max_workers=None, discard_models=False, **find_binding_params):
        """
        Run the binding SPB analysis for each country.

        Parameters:
            edp_countries (list): List of countries for which EDP should be applied.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes to use (default is the number of CPUs*5).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_binding_params: dict of additional parameters for find_spb_binding.
        """
        tasks = []
        print(f'Running find_spb_binding for {len(self.countries)} countries (parallel={parallel})')
        for country, model in list(self.models.items()):
            tasks.append((country, model, edp_countries, find_binding_params))
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_find_spb_binding_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        country, spb_dict, df_dict, binding_params, df_fanchart = future.result()
                        self.results[country]['spb_target_dict'] = spb_dict
                        self.results[country]['df_dict'] = df_dict
                        self.results[country]['binding_parameter_dict'] = binding_params
                        self.results[country]['df_fanchart'] = df_fanchart
                        if discard_models:
                            del self.models[country]
                    except Exception as e:
                        print(f"Error processing binding tasks {e}")
        else:
            # Sequential processing using a simple loop.
            for task in tqdm(tasks):
                try:
                    country, spb_dict, df_dict, binding_params, df_fanchart = _find_spb_binding_task(task)
                    self.results[country]['spb_target_dict'] = spb_dict
                    self.results[country]['df_dict'] = df_dict
                    self.results[country]['binding_parameter_dict'] = binding_params
                    self.results[country]['df_fanchart'] = df_fanchart
                    if discard_models:
                        del self.models[country]
                except Exception as e:
                    print(f"Error processing binding task for {country}: {e}")

    def find_spb_stochastic(self, store_as='stochastic', parallel=True, max_workers=None, discard_models=False, **find_stochastic_params):
        """
        Run the stochastic SPB analysis for each country.

        Parameters:
            store_as (str): Key to use when storing the result.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes to use (default is the number of CPUs*5).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_stochastic_params: dict of additional parameters for find_spb_stochastic.
        """
        tasks = []
        print(f'Running find_spb_stochastic for {len(self.countries)} countries (parallel={parallel})')
        for country, model in list(self.models.items()):
            tasks.append((country, model, store_as, find_stochastic_params))
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_find_spb_stochastic_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        country, spb_dict, df_dict, df_fanchart = future.result()
                        self.results[country]['spb_target_dict'] = spb_dict
                        self.results[country]['df_dict'] = df_dict
                        self.results[country]['df_fanchart'] = df_fanchart
                        if discard_models:
                            del self.models[country]
                    except Exception as e:
                        print(f"Error processing stochastic tasks {e}")
        else:
            # Sequential processing using a simple loop.
            for task in tqdm(tasks):
                try:
                    country, spb_dict, df_dict, df_fanchart = _find_spb_stochastic_task(task)
                    self.results[country]['spb_target_dict'] = spb_dict
                    self.results[country]['df_dict'] = df_dict
                    self.results[country]['df_fanchart'] = df_fanchart
                    if discard_models:
                        del self.models[country]
                except Exception as e:
                    print(f"Error processing stochastic task for {country}: {e}")

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
                    self.results[country]['df_dict'] = {
                        store_as: self.models[country].df(all=True),
                        }
                if discard_models:
                    del self.models[country]

    def save_spb(self, folder=None, file=None):
        """
        Save the SPB targets for each instance to an Excel file.

        This method builds an SPB table from the results, calculates binding scenarios,
        maps country codes to full names, and saves the table.

        Parameters:
            folder (str): Output folder path. If None, a folder based on the current date is created.
            file (str): Filename for the Excel output.
            save (bool): Whether to save the file. If False, only the DataFrame is returned.
        
        Returns:
            DataFrame: The SPB table.
        """
        # Build the SPB table from the results
        self.df_spb = pd.DataFrame()
        for country in self.results:
            spb_target_dict = self.results[country]['spb_target_dict']
            for scenario, spb_val in spb_target_dict.items():
                temp_df = pd.DataFrame({
                    'country': [country],
                    'adjustment_period': [self.dsa_params['adjustment_period']],
                    'scenario': [scenario],
                    'spbstar': [spb_val]
                })
                self.df_spb = pd.concat([self.df_spb, temp_df], ignore_index=True)

        # Pivot to have one row per (country, adjustment_period)
        self.df_spb = self.df_spb.pivot(
            index=['country', 'adjustment_period'],
            columns='scenario',
            values='spbstar'
        ).reset_index()

        # Calculate the binding DSA scenario
        dsa_col_list = ['main_adjustment', 'adverse_r_g', 'lower_spb', 'financial_stress', 'stochastic']
        dsa_col_list = [col for col in dsa_col_list if col in self.df_spb.columns]
        self.df_spb['binding_dsa'] = self.df_spb[dsa_col_list].max(axis=1) if dsa_col_list else np.nan

        # Calculate the binding safeguard scenario
        safeguard_col_list = ['deficit_reduction', 'debt_safeguard', 'deficit_resilience']
        safeguard_col_list = [col for col in safeguard_col_list if col in self.df_spb.columns]
        self.df_spb['binding_safeguard'] = self.df_spb[safeguard_col_list].max(axis=1) if safeguard_col_list else np.nan

        # Rename column if necessary
        self.df_spb.rename(columns={'main_adjustment_deficit_reduction': 'deficit_reduction'}, inplace=True)

        # Map country codes to full country names
        self.df_spb['iso'] = self.df_spb['country'].copy()
        self.df_spb['country'] = self.df_spb['country'].apply(self._get_country_name)

        # Define the desired column order; fill missing columns with NaN
        col_order = ['country', 'iso', 'adjustment_period',
                     'main_adjustment', 'adverse_r_g', 'lower_spb', 'financial_stress',
                     'stochastic', 'binding_dsa', 'deficit_reduction', 'edp',
                     'debt_safeguard', 'deficit_resilience', 'binding_safeguard', 'binding']
        
        # add columns that are not in col order to the end of the dataframe
        col_order += [col for col in self.df_spb.columns if col not in col_order]
        for col in col_order:
            if col not in self.df_spb.columns:
                self.df_spb[col] = np.nan
        self.df_spb = self.df_spb[col_order].sort_values(['adjustment_period', 'country']).round(3)

        # Save the DataFrame to Excel if required
        if folder is None:
            folder = f'{self._today}/'
        folder_path = os.path.join('../output/', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if file is None:
            file = f"spb_targets_{self.dsa_params['adjustment_period']}y.xlsx"
        file_path = os.path.join(folder_path, file)
        self.df_spb.to_excel(file_path, index=False)
        print(f"SPB table saved to {file_path}")

        return self.df_spb.dropna(axis=1, how='all')

    def save_dfs(self, folder=None, file=None):
        """
        Save the time-series DataFrames stored in the results to an Excel file,
        with one sheet per country/scenario.

        Parameters:
            folder (str): Output folder path. If None, a folder based on the current date is created.
            file (str): Filename for the Excel output.
        """
        if folder is None:
            folder = f'{self._today}/'
        folder_path = os.path.join('../output/', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if file is None:
            file = f"timeseries_{self.dsa_params['adjustment_period']}y.xlsx"
        file_path = os.path.join(folder_path, file)
        with pd.ExcelWriter(file_path) as writer:
            for country, res in self.results.items():
                df_dict = res['df_dict']
                for scenario, df in df_dict.items():
                    # Limit the sheet name to 31 characters
                    sheet_name = f"{country}_{self.dsa_params['adjustment_period']}_{scenario}"[:31]
                    df.to_excel(writer, sheet_name=sheet_name)
        print(f"DataFrames saved to {file_path}")

    def get_country_model(self, country):
        """
        Return the dictionary of DSA models for the given country.

        Parameters:
            country (str): The country code (e.g., 'AUT').

        Returns:
            dict: Dictionary of DSA models for the specified country.
        """
        return self.models.get(country.upper(), {})

    def _get_country_name(self, country):
        """
        Get the full country name corresponding to a given country code.

        Parameters:
            country (str): The country code (e.g., 'AUT').

        Returns:
            str or None: The full country name if found; otherwise, None.
        """
        country_code_dict = {
            'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
            'CYP': 'Cyprus', 'CZE': 'Czechia', 'DNK': 'Denmark', 'EST': 'Estonia',
            'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
            'HUN': 'Hungary', 'IRL': 'Ireland', 'ITA': 'Italy', 'LVA': 'Latvia',
            'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta', 'NLD': 'Netherlands',
            'POL': 'Poland', 'PRT': 'Portugal', 'ROU': 'Romania', 'SVK': 'Slovakia',
            'SVN': 'Slovenia', 'ESP': 'Spain', 'SWE': 'Sweden'
        }
        return country_code_dict.get(country.upper(), None)

    def df_avg(self, countries=None, scenario='binding'):
        """
        Calculate average of model DataFrames:
        - For non-absolute (lowercase) attributes: compute the weighted average 
            using (ngdp * exr_eur) as weights (row-wise).
        - For absolute (non-lowercase) attributes: simply compute the row-wise sum.
        
        Parameters:
        countries (list): List of countries to process. Defaults to self.countries.
        scenario (str): The scenario key to extract from each country's df_dict.
        
        Returns:
        avg_df (pd.DataFrame): Aggregated DataFrame with the same index and columns,
                                where each cell is the weighted average or sum.
        """
        # Specify the list of countries to process.
        if countries == 'EU':
            countries = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 
                        'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 
                        'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']
        elif countries == 'EA':
            countries = ['AUT', 'BEL', 'HRV', 'CYP', 'EST', 'FIN', 
                        'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 
                        'MLT', 'NLD', 'PRT', 'SVK', 'SVN', 'ESP']
        elif countries is None:
            countries = self.countries
        
        # Initialize dictionaries to store each country's DataFrame and its corresponding weight series.
        scenario_df_dict = {c: None for c in countries}
        weight_series_dict = {c: None for c in countries}
        
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

# Module-level helper function for binding SPB
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
    # Determine whether to apply EDP for this country
    edp = country in edp_countries

    # Run the binding SPB analysis on the model
    model.find_spb_binding(save_df=True, edp=edp, **find_binding_params)

    # get fanchart
    model.fanchart(plot=False)

    # Extract the results from the model
    spb_dict = model.spb_target_dict
    df_dict = model.df_dict
    binding_params = model.binding_parameter_dict
    df_fanchart = model.df_fanchart

    # Run the projection step and add the 'no_policy_change' results
    model.project()
    df_dict['no_policy_change'] = model.df(all=True)

    return country, spb_dict, df_dict, binding_params, df_fanchart

# Module-level helper function for stochastic SPB
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
    spb_dict = {store_as: model.spb_target}
    df_dict = {store_as: model.df(all=True)}
    df_fanchart = model.df_fanchart
    return country, spb_dict, df_dict, df_fanchart

