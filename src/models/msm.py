from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import math
from typing import Union
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from copy import deepcopy
from pytorch_lightning import Trainer
import ray
from ray import tune
from ray import ray_constants
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models import TimeVaryingCausalModel
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, clip_normalize_stabilized_weights
from src.models.utils_lstm import VariationalLSTM


logger = logging.getLogger(__name__)


class MSM(TimeVaryingCausalModel):
    """
    Pytorch-Lightning implementation of Marginal Structural Models (MSMs) (https://pubmed.ncbi.nlm.nih.gov/10955408/)
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'msm_regressor', 'propensity_treatment', 'propensity_history'}
    tuning_criterion = None

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.lag_features = args.model.lag_features

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            assert self.hparams.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
            self.dataset_collection.process_data_multi()

    def get_exploded_dataset(self, dataset: Dataset, min_length: int, only_active_entries=True, max_length=None) -> Dataset:
        exploded_dataset = deepcopy(dataset)
        if max_length is None:
            max_length = max(exploded_dataset.data['sequence_lengths'][:])
        if not only_active_entries:
            exploded_dataset.data['active_entries'][:, :, :] = 1.0
            exploded_dataset.data['sequence_lengths'][:] = max_length
        exploded_dataset.explode_trajectories(min_length)
        return exploded_dataset

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        logger.info(f'Propensity scores for {dataset.subset_name}.')
        exploded_dataset = self.get_exploded_dataset(dataset, min_length=self.lag_features, only_active_entries=False)

        inputs = self.get_inputs(exploded_dataset)
        classifier = getattr(self, self.model_type)

        propensity_scores = np.stack(classifier.predict_proba(inputs), 1)[:, :, 1]
        propensity_scores = propensity_scores.reshape(dataset.data['active_entries'].shape[0],
                                                      dataset.data['active_entries'].shape[1] - self.lag_features,
                                                      self.dim_treatments)
        propensity_scores = np.concatenate([0.5 * np.ones((propensity_scores.shape[0], self.lag_features, self.dim_treatments)),
                                            propensity_scores], axis=1)
        return propensity_scores


class MSMPropensityTreatment(MSM):

    model_type = 'propensity_treatment'

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None, has_vitals: bool = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)

        self.input_size = self.dim_treatments
        logger.info(f'Input size of {self.model_type}: {self.input_size}')
        self.output_size = self.dim_treatments

        self.propensity_treatment = MultiOutputClassifier(LogisticRegression(penalty='none', max_iter=args.exp.max_epochs))
        self.save_hyperparameters(args)

    def get_inputs(self, dataset: Dataset) -> np.array:
        active_entries = dataset.data['active_entries']
        prev_treatments = dataset.data['prev_treatments']
        inputs = (prev_treatments * active_entries).sum(1)
        return inputs

    def fit(self):
        self.prepare_data()
        train_f = self.get_exploded_dataset(self.dataset_collection.train_f, min_length=self.lag_features)
        active_entries = train_f.data['active_entries']
        last_entries = active_entries - \
            np.concatenate([active_entries[:, 1:, :], np.zeros((active_entries.shape[0], 1, 1))], axis=1)

        # Inputs
        inputs = self.get_inputs(train_f)

        # Outputs
        current_treatments = train_f.data['current_treatments']
        outputs = (current_treatments * last_entries).sum(1)

        self.propensity_treatment.fit(inputs, outputs)


class MSMPropensityHistory(MSM):

    model_type = 'propensity_history'

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None, has_vitals: bool = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)

        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0

        logger.info(f'Input size of {self.model_type}: {self.input_size}')
        self.output_size = self.dim_treatments

        self.propensity_history = MultiOutputClassifier(LogisticRegression(penalty='none', max_iter=args.exp.max_epochs))
        self.save_hyperparameters(args)

    def get_inputs(self, dataset: Dataset, projection_horizon=0) -> np.array:
        active_entries = dataset.data['active_entries']
        lagged_entries = active_entries - \
            np.concatenate([active_entries[:, self.lag_features + 1:, :],
                            np.zeros((active_entries.shape[0], self.lag_features + 1, 1))], axis=1)
        if projection_horizon > 0:
            lagged_entries = np.concatenate([lagged_entries[:, projection_horizon:, :],
                                             np.zeros((active_entries.shape[0], projection_horizon, 1))], axis=1)

        active_entries_before_proection = np.concatenate([active_entries[:, projection_horizon:, :],
                                                          np.zeros((active_entries.shape[0], projection_horizon, 1))], axis=1)

        prev_treatments = dataset.data['prev_treatments']
        inputs = [(prev_treatments * active_entries_before_proection).sum(1)]
        if self.has_vitals:
            vitals = dataset.data['vitals']
            inputs.append(vitals[np.repeat(lagged_entries, self.dim_vitals, 2) == 1.0].reshape(vitals.shape[0],
                                                                                               (self.lag_features + 1) *
                                                                                               self.dim_vitals))
        if self.autoregressive:
            prev_outputs = dataset.data['prev_outputs']
            inputs.append(prev_outputs[np.repeat(lagged_entries, self.dim_outcome, 2) == 1.0].reshape(prev_outputs.shape[0],
                                                                                                      (self.lag_features + 1) *
                                                                                                      self.dim_outcome))
        static_features = dataset.data['static_features']
        inputs.append(static_features)
        return np.concatenate(inputs, axis=1)

    def fit(self):
        self.prepare_data()
        train_f = self.get_exploded_dataset(self.dataset_collection.train_f, min_length=self.lag_features)
        active_entries = train_f.data['active_entries']
        last_entries = active_entries - \
            np.concatenate([active_entries[:, 1:, :], np.zeros((active_entries.shape[0], 1, 1))], axis=1)

        # Inputs
        inputs = self.get_inputs(train_f)

        # Outputs
        current_treatments = train_f.data['current_treatments']
        outputs = (current_treatments * last_entries).sum(1)

        self.propensity_history.fit(inputs, outputs)


class MSMRegressor(MSM):

    model_type = 'msm_regressor'

    def __init__(self,
                 args: DictConfig,
                 propensity_treatment: MSMPropensityTreatment = None,
                 propensity_history: MSMPropensityHistory = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None, has_vitals: bool = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)

        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0

        logger.info(f'Input size of {self.model_type}: {self.input_size}')
        self.output_size = self.dim_outcome

        self.propensity_treatment = propensity_treatment
        self.propensity_history = propensity_history

        self.msm_regressor = \
            [MultiOutputRegressor(LinearRegression()) for _ in range(self.dataset_collection.projection_horizon + 1)]
        self.save_hyperparameters(args)

    def get_inputs(self, dataset: Dataset, projection_horizon=0, tau=0) -> np.array:
        active_entries = dataset.data['active_entries']
        lagged_entries = active_entries - np.concatenate([active_entries[:, self.lag_features + 1:, :],
                                                          np.zeros((active_entries.shape[0], self.lag_features + 1, 1))], axis=1)
        if projection_horizon > 0:
            lagged_entries = np.concatenate([lagged_entries[:, projection_horizon:, :],
                                             np.zeros((active_entries.shape[0], projection_horizon, 1))], axis=1)

        active_entries_before_proection = np.concatenate([active_entries[:, projection_horizon:, :],
                                                          np.zeros((active_entries.shape[0], projection_horizon, 1))], axis=1)

        prev_treatments = dataset.data['prev_treatments']
        inputs = [(prev_treatments * active_entries_before_proection).sum(1)]
        if self.has_vitals:
            vitals = dataset.data['vitals']
            inputs.append(vitals[np.repeat(lagged_entries, self.dim_vitals, 2) == 1.0].reshape(vitals.shape[0],
                                                                                               (self.lag_features + 1) *
                                                                                               self.dim_vitals))
        if self.autoregressive:
            prev_outputs = dataset.data['prev_outputs']
            inputs.append(
                prev_outputs[np.repeat(lagged_entries, self.dim_outcome, 2) == 1.0].reshape(prev_outputs.shape[0],
                                                                                            (self.lag_features + 1) *
                                                                                            self.dim_outcome))
        static_features = dataset.data['static_features']
        inputs.append(static_features)

        # Adding current actions
        current_treatments = dataset.data['current_treatments']
        prediction_entries = active_entries - np.concatenate(
            [active_entries[:, tau + 1:, :], np.zeros((active_entries.shape[0], tau + 1, 1))], axis=1)
        prediction_entries = np.concatenate([prediction_entries[:, projection_horizon - tau:, :],
                                             np.zeros((prediction_entries.shape[0], projection_horizon - tau, 1))], axis=1)
        inputs.append((current_treatments * prediction_entries).sum(1))

        return np.concatenate(inputs, axis=1)

    def get_sample_weights(self, dataset: Dataset, tau=0) -> np.array:
        active_entries = dataset.data['active_entries']
        stabilized_weights = dataset.data['stabilized_weights']

        prediction_entries = active_entries - np.concatenate(
            [active_entries[:, tau + 1:, :], np.zeros((active_entries.shape[0], tau + 1, 1))],
            axis=1)
        stabilized_weights = stabilized_weights[np.squeeze(prediction_entries) == 1.0].reshape(stabilized_weights.shape[0],
                                                                                               tau + 1)
        sw = np.prod(stabilized_weights, axis=1)
        sw_tilde = np.clip(sw, np.nanquantile(sw, 0.01), np.nanquantile(sw, 0.99))
        return sw_tilde

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.dataset_collection is not None and 'stabilized_weights' not in self.dataset_collection.train_f.data:
            self.dataset_collection.process_propensity_train_f(self.propensity_treatment, self.propensity_history)

    def fit(self):
        self.prepare_data()
        for tau in range(self.dataset_collection.projection_horizon + 1):

            train_f = self.get_exploded_dataset(self.dataset_collection.train_f, min_length=self.lag_features + tau)
            active_entries = train_f.data['active_entries']
            last_entries = active_entries - \
                np.concatenate([active_entries[:, 1:, :], np.zeros((active_entries.shape[0], 1, 1))], axis=1)

            # Inputs
            inputs = self.get_inputs(train_f, projection_horizon=tau, tau=tau)

            # Stabilized weights
            sw = self.get_sample_weights(train_f, tau=tau)

            # Outputs
            outputs = train_f.data['outputs']
            outputs = (outputs * last_entries).sum(1)

            self.msm_regressor[tau].fit(inputs, outputs, sample_weight=sw)

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        batch_size = 10000
        outcome_pred = np.zeros_like(dataset.data['outputs'])
        for batch in range(len(dataset) // batch_size + 1):
            subset = deepcopy(dataset)
            for (k, v) in subset.data.items():
                subset.data[k] = v[batch * batch_size:(batch + 1) * batch_size]

            exploded_dataset = self.get_exploded_dataset(subset, min_length=self.lag_features, only_active_entries=False,
                                                         max_length=max(dataset.data['sequence_lengths'][:]))
            inputs = self.get_inputs(exploded_dataset, projection_horizon=0, tau=0)
            outcome_pred_batch = self.msm_regressor[0].predict(inputs)

            outcome_pred_batch = outcome_pred_batch.reshape(subset.data['active_entries'].shape[0],
                                                            subset.data['active_entries'].shape[1] - 1,
                                                            self.dim_outcome)
            # First time-step requires two previous outcomes -> duplicating the next prediction
            outcome_pred_batch = np.concatenate([outcome_pred_batch[:, :1, :], outcome_pred_batch], axis=1)
            outcome_pred[batch * batch_size:(batch + 1) * batch_size] = outcome_pred_batch
        return outcome_pred

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))

        for t in range(1, self.dataset_collection.projection_horizon + 1):
            inputs = self.get_inputs(dataset, projection_horizon=self.dataset_collection.projection_horizon - 1, tau=t - 1)
            outcome_pred = self.msm_regressor[t].predict(inputs)
            predicted_outputs[:, t - 1] = outcome_pred

        return predicted_outputs
