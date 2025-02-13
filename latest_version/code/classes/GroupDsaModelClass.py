# ========================================================================================= #
#               European Commission Debt Sustainability Analysis - Group DSA Class         #
# ========================================================================================= #
#
# This class collects several DSA model instances for different countries and 
# adjustment periods. Its methods run the DSA models and save the results. 
# Computationally demanding tasks are run in parallel using concurrent.futures,
# but can be deactivated via an optional argument.
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
    def __init__(self, countries, adjustment_periods, **dsa_params):
        """
        Initialize the GroupDsaModel instance.

        Parameters:
            countries (list): List of country codes.
            adjustment_periods (list): List of adjustment periods.
            **dsa_params: Additional keyword arguments to pass to the DSA model.
        """

        # Store input parameters
        self.countries = countries
        self.adjustment_periods = adjustment_periods
        self.dsa_params = dsa_params
        self._today = time.strftime('%Y_%m_%d')

        # Dictionary to hold DSA model instances by country and adjustment period
        self.models = {c: {adj: None for adj in adjustment_periods} for c in countries}

        # Dictionary to store results for each country and adjustment period
        self.results = {c: {adj: None for adj in adjustment_periods} for c in countries}

        # Initialize DSA models for each country and adjustment period
        self._init_models()

    def _init_models(self):
        """
        Instantiate a DSA model for each country and adjustment period.

        A local import is used here to avoid circular import issues.
        """
        from classes import StochasticDsaModel as DSA
        for country in self.countries:
            for adj in self.adjustment_periods:
                # Create a copy of the DSA parameters and update with country and adjustment period
                model_params = self.dsa_params.copy()
                model_params['adjustment_period'] = adj
                model_params['country'] = country
                self.models[country][adj] = DSA(**model_params)

    def update_params(self, update_params):
        """
        Update the attributes of each DSA model.

        Parameters:
            **update_params: Dictionaries of parameters to update for each country and adjustment period.
        """
        for country in self.countries:
            for adj in self.adjustment_periods:
                for attr, value in update_params.items():
                    setattr(self.models[country][adj], attr, value)
                
    def project(self, store_as=False, discard_models=False, **project_params):
        """
        Run the projection step for each DSA model.

        Parameters:
            store_as (str): Key to use when storing to the result dictionary.
            discard_models (bool): If True, delete the model from memory after processing.
            **project_params: dictionaries of parameters to pass to the project method 
            for each country and adjustment period.
        """
        for country in self.countries:
            for adj in self.adjustment_periods:

                # Get the parameters for the current country and adjustment period
                params = project_params.get(country, {}).get(adj, {})
                self.models[country][adj].project(**params)
        
                # Store results if required
                if store_as:
                    self.results[country][adj] = {
                        'spb_target_dict': {store_as: self.models[country][adj].spb_target},
                        'df_dict': {store_as: self.models[country][adj].df(all=True)},
                    }
                if discard_models:
                    del self.models[country][adj]

    def find_spb_binding(self, edp_countries=[], parallel=True, max_workers=None, discard_models=False, **find_binding_params):
        """
        Run the binding SPB analysis for each (country, adjustment_period) pair.

        Parameters:
            edp_countries (list): List of countries for which EDP should be applied.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes to use (default is the number of CPUs*5).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_binding_params: dict of additional parameters for find_spb_binding.
        """
        tasks = []
        print(f'Running find_spb_binding for {len(self.countries)*len(self.adjustment_periods)} country-period pairs (parallel={parallel})')
        for country in self.countries:
            # Ensure the results dictionary exists for this country.
            self.results[country] = {}
            for adj, model in list(self.models[country].items()):
                tasks.append((country, adj, model, edp_countries, find_binding_params))
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_find_spb_binding_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        country, adj, spb_dict, df_dict, binding_params = future.result()
                        self.results[country][adj] = {
                            'spb_target_dict': spb_dict,
                            'df_dict': df_dict,
                            'binding_parameter_dict': binding_params
                        }
                        if discard_models:
                            del self.models[country][adj]
                    except Exception as e:
                        print(f"Error processing binding task for {country} {adj}: {e}")
        else:
            # Sequential processing using a simple loop.
            for task in tqdm(tasks):
                try:
                    country, adj, spb_dict, df_dict, binding_params = _find_spb_binding_task(task)
                    self.results[country][adj] = {
                        'spb_target_dict': spb_dict,
                        'df_dict': df_dict,
                        'binding_parameter_dict': binding_params
                    }
                    if discard_models:
                        del self.models[country][adj]
                except Exception as e:
                    print(f"Error processing binding task for {country} {adj}: {e}")

    def find_spb_stochastic(self, store_as='stochastic', parallel=True, max_workers=None, discard_models=False, **find_stochastic_params):
        """
        Run the stochastic SPB analysis for each (country, adjustment_period) pair.

        Parameters:
            store_as (str): Key to use when storing the result.
            parallel (bool): If True (default), run tasks in parallel using ProcessPoolExecutor;
                             if False, process tasks sequentially.
            max_workers (int): Maximum number of worker processes to use (default is the number of CPUs*5).
            discard_models (bool): If True, delete the model from memory after processing.
            **find_stochastic_params: dict of additional parameters for find_spb_stochastic.
        """
        tasks = []
        print(f'Running find_spb_stochastic for {len(self.countries)*len(self.adjustment_periods)} country-period pairs (parallel={parallel})')
        for country in self.countries:
            for adj, model in list(self.models[country].items()):
                tasks.append((country, adj, model, store_as, find_stochastic_params))
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_find_spb_stochastic_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        country, adj, spb_dict, df_dict = future.result()
                        self.results[country][adj] = {
                            'spb_target_dict': spb_dict,
                            'df_dict': df_dict,
                        }
                        if discard_models:
                            del self.models[country][adj]
                    except Exception as e:
                        print(f"Error processing stochastic task for {country} {adj}: {e}")
        else:
            # Sequential processing using a simple loop.
            for task in tqdm(tasks):
                try:
                    country, adj, spb_dict, df_dict = _find_spb_stochastic_task(task)
                    self.results[country][adj] = {
                        'spb_target_dict': spb_dict,
                        'df_dict': df_dict,
                    }
                    if discard_models:
                        del self.models[country][adj]
                except Exception as e:
                    print(f"Error processing stochastic task for {country} {adj}: {e}")

    def project_fr(self, store_as=False, discard_models=False, **fr_params):
        """
        Run the fiscal rule analysis for each DSA model.

        Parameters:
            store_as (str): Key to use when storing to the results dictionary.
            discard_models (bool): If True, delete the model from memory after processing.
            **fr_params: dictionaries of parameters to pass to the project_fr method 
            for each country and adjustment period.
        """
        for country in self.countries:
            for adj in self.adjustment_periods:

                # Get the parameters for the current country and adjustment period
                params = fr_params.get(country, {}).get(adj, {})
                self.models[country][adj].project_fr(**params)

                # Store results if required
                if store_as:
                    self.results[country][adj] = {
                        'df_dict': {store_as: self.models[country][adj].df(all=True)},
                    }
                if discard_models:
                    del self.models[country][adj]

    def save_spb(self, folder=None, file='spb_targets.xlsx', save=True):
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
        for country in self.results.keys():
            for adj in self.results[country].keys():
                spb_target_dict = self.results[country][adj]['spb_target_dict']
                for scenario, spb_val in spb_target_dict.items():
                    temp_df = pd.DataFrame({
                        'country': [country],
                        'adjustment_period': [adj],
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
        output_path = '../output/'
        folder_path = os.path.join(output_path, folder)
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.df_spb.to_excel(file_path, index=False)
        print(f"SPB table saved to {file_path}")

        return self.df_spb.dropna(axis=1, how='all')

    def save_dfs(self, folder=None, file='timeseries.xlsx'):
        """
        Save the time-series DataFrames stored in the results to an Excel file,
        with one sheet per country/adjustment period/scenario.

        Parameters:
            folder (str): Output folder path. If None, a folder based on the current date is created.
            file (str): Filename for the Excel output.
        """
        if folder is None:
            folder = f'{self._today}/'
        output_path = '../output/'
        folder_path = os.path.join(output_path, folder)
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with pd.ExcelWriter(file_path) as writer:
            for country, adj_dict in self.results.items():
                for adj, res in adj_dict.items():
                    df_dict = res['df_dict']
                    for scenario, df in df_dict.items():
                        # Limit the sheet name to 31 characters
                        sheet_name = f'{country}_{adj}_{scenario}'[:31]
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

    def df_avg(self, countries=None, adjustment_period=None, scenario='binding'):
        """
        Calculate average of model DataFrames:
        - For non-absolute (lowercase) attributes: compute the weighted average 
            using (ngdp * exr_eur) as weights (row-wise).
        - For absolute (non-lowercase) attributes: simply compute the row-wise sum.
        
        Parameters:
        countries (list): List of countries to process. Defaults to self.countries.
        adjustment_period (str): The adjustment period to use. Defaults to the first period.
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
        
        # Use first adjustment period in list if none provided.
        if adjustment_period is None:
            adjustment_period = self.adjustment_periods[0]
        
        # Initialize dictionaries to store each country's DataFrame and its corresponding weight series.
        scenario_df_dict = {c: None for c in countries}
        weight_series_dict = {c: None for c in countries}
        
        # Loop over each country to extract the desired scenario DataFrame and compute weights.
        for country in countries:
            # Access the dictionary containing DataFrames for the current country and period.
            df_dict = self.results[country][adjustment_period]['df_dict']
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
    - adj: the adjustment period (could be a number or string)
    - model: the model instance
    - edp_countries: list of countries for which EDP should be applied
    - find_binding_params: dict of additional parameters for find_spb_binding
    """
    country, adj, model, edp_countries, find_binding_params = args
    # Determine whether to apply EDP for this country
    edp = country in edp_countries

    # Run the binding SPB analysis on the model
    model.find_spb_binding(save_df=True, edp=edp, **find_binding_params)
    spb_dict = model.spb_target_dict
    df_dict = model.df_dict
    binding_params = model.binding_parameter_dict

    # Run the projection step and add the 'no_policy_change' results
    model.project()
    df_dict['no_policy_change'] = model.df(all=True)

    return country, adj, spb_dict, df_dict, binding_params

# Module-level helper function for stochastic SPB
def _find_spb_stochastic_task(args):
    """
    Helper function to run the stochastic SPB analysis for one model.
    
    Expected arguments:
    - country: the country code (string)
    - adj: the adjustment period identifier
    - model: the model instance
    - store_as: key to use for saving the result in the dictionary
    - find_stochastic_params: dict of additional parameters for find_spb_stochastic
    """
    country, adj, model, store_as, find_stochastic_params = args
    model.find_spb_stochastic(**find_stochastic_params)
    spb_dict = {store_as: model.spb_target}
    df_dict = {store_as: model.df(all=True)}
    return country, adj, spb_dict, df_dict