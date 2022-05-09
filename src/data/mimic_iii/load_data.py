import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from omegaconf import ListConfig

from src import ROOT_PATH

logger = logging.getLogger(__name__)


def process_static_features(static_features: pd.DataFrame, drop_first=False) -> pd.DataFrame:
    """
    Global standard normalisation of static features & one hot encoding
    Args:
        static_features: pd.DataFrame with unprocessed static features
        drop_first: Dropping first class of one-hot-encoded features

    Returns: pd.DataFrame with pre-processed static features

    """
    processed_static_features = []
    for feature in static_features.columns:
        if isinstance(static_features[feature].iloc[0], float):
            mean = np.mean(static_features[feature])
            std = np.std(static_features[feature])
            processed_static_features.append((static_features[feature] - mean) / std)
        else:
            one_hot = pd.get_dummies(static_features[feature], drop_first=drop_first)
            processed_static_features.append(one_hot.astype(float))

    static_features = pd.concat(processed_static_features, axis=1)
    return static_features


def load_mimic3_data_processed(data_path: str,
                               min_seq_length: int = None,
                               max_seq_length: int = None,
                               treatment_list: List[str] = None,
                               outcome_list: List[str] = None,
                               vital_list: List[str] = None,
                               static_list: List[str] = None,
                               max_number: int = None,
                               data_seed: int = 100,
                               drop_first=False,
                               **kwargs) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict):
    """
    Load and pre-process MIMIC-3 hourly averaged dataset (for real-world experiments)
    :param data_path: Path with MIMIC-3 dataset (HDFStore)
    :param min_seq_length: Min sequence lenght in cohort
    :param min_seq_length: Max sequence lenght in cohort
    :param treatment_list: List of treaments
    :param outcome_list: List of outcomes
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: tuple of DataFrames and params (treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params)
    """

    logger.info(f'Loading MIMIC-III dataset from {data_path}.')

    h5 = pd.HDFStore(data_path, 'r')
    if treatment_list is None:
        treatment_list = ['vaso', 'vent']
    if outcome_list is None:
        outcome_list = [
            'diastolic blood pressure',
            'oxygen saturation'
        ]
    else:
        outcome_list = ListConfig([outcome.replace('_', ' ') for outcome in outcome_list])
    if vital_list is None:
        vital_list = [
            'heart rate',
            'red blood cell count',
            'sodium',
            'mean blood pressure',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'positive end-expiratory pressure set',
            'respiratory rate',
            'prothrombin time pt',
            'cholesterol',
            'hemoglobin',
            'creatinine',
            'blood urea nitrogen',
            'bicarbonate',
            'calcium ionized',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
            'calcium urine'
        ]
    if static_list is None:
        static_list = [
            'gender',
            'ethnicity',
            'age'
        ]

    treatments = h5['/interventions'][treatment_list]
    all_vitals = h5['/vitals_labs_mean'][outcome_list + vital_list]
    static_features = h5['/patients'][static_list]

    treatments = treatments.droplevel(['hadm_id', 'icustay_id'])
    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names
    static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

    # Filling NA
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')

    # Filtering longer then min_seq_length and cropping to max_seq_length
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)
    treatments = treatments.loc[filtered_users]
    all_vitals = all_vitals.loc[filtered_users]
    if max_seq_length is not None:
        treatments = treatments.groupby('subject_id').head(max_seq_length)
        all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
    static_features = static_features[static_features.index.isin(filtered_users)]
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')

    # Global scaling (same as with semi-synthetic)
    outcomes_unscaled = all_vitals[outcome_list].copy()
    mean = np.mean(all_vitals, axis=0)
    std = np.std(all_vitals, axis=0)
    all_vitals = (all_vitals - mean) / std

    # Splitting outcomes and vitals
    outcomes = all_vitals[outcome_list].copy()
    vitals = all_vitals[vital_list].copy()
    static_features = process_static_features(static_features, drop_first=drop_first)
    scaling_params = {
        'output_means': mean[outcome_list].to_numpy(),
        'output_stds': std[outcome_list].to_numpy(),
    }

    h5.close()
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params


def load_mimic3_data_raw(data_path: str,
                         min_seq_length: int = None,
                         max_seq_length: int = None,
                         max_number: int = None,
                         vital_list: List[str] = None,
                         static_list: List[str] = None,
                         data_seed: int = 100,
                         drop_first=False,
                         **kwargs) -> (pd.DataFrame, pd.DataFrame):
    """
    Load MIMIC-3 hourly averaged dataset, without preprocessing (for semi-synthetic experiments)
    :param data_path: Path with MIMIC-3 dataset (HDFStore)
    :param min_seq_length: Min sequence lenght in cohort
    :param max_seq_length: Max sequence length in cohort
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: Tuple of DataFrames (all_vitals, static_features)
    """
    logger.info(f'Loading MIMIC-III dataset from {data_path}.')

    h5 = pd.HDFStore(data_path, 'r')
    if vital_list is None:
        vital_list = [
            'heart rate',
            'red blood cell count',
            'sodium',
            'mean blood pressure',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'positive end-expiratory pressure set',
            'respiratory rate',
            'prothrombin time pt',
            'cholesterol',
            'hemoglobin',
            'creatinine',
            'blood urea nitrogen',
            'bicarbonate',
            'calcium ionized',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
            'calcium urine'
        ]
    if static_list is None:
        static_list = [
            'gender',
            'ethnicity',
            'age'
        ]

    all_vitals = h5['/vitals_labs_mean'][vital_list]
    static_features = h5['/patients'][static_list]

    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names
    static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

    # Filling NA
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')

    # Filtering longer then min_seq_length and cropping to max_seq_length
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)
    all_vitals = all_vitals.loc[filtered_users]
    static_features = static_features.loc[filtered_users]
    if max_seq_length is not None:
        all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')

    # Global Mean-Std Normalisation
    mean = np.mean(all_vitals, axis=0)
    std = np.std(all_vitals, axis=0)
    all_vitals = (all_vitals - mean) / std

    static_features = process_static_features(static_features, drop_first=drop_first)

    h5.close()
    return all_vitals, static_features


if __name__ == "__main__":
    data_path = ROOT_PATH + '/' + 'data/processed/all_hourly_data.h5'
    treatments, outcomes, vitals, stat_features, outcomes_unscaled, scaling_params = \
        load_mimic3_data_processed(data_path, min_seq_length=100, max_seq_length=100)
