import numpy
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pytorch_lightning import Trainer
import ray
from ray import tune
from ray import ray_constants
from typing import Union, List

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, ROutcomeVitalsHead
from src.models.utils_lstm import VariationalLSTM


logger = logging.getLogger(__name__)


class GNet(TimeVaryingCausalModel):
    """
    Pytorch-Lightning implementation of G-Net (https://proceedings.mlr.press/v158/li21a/li21a.pdf)
    """

    model_type = 'g_net'
    possible_model_types = {'g_net'}
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        assert self.autoregressive  # Works only in autoregressive regime

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        self.input_size += self.dim_vitals if self.has_vitals else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_vitals + self.dim_outcome

        self._init_specific(args.model.g_net)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network (only g_net)
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            self.dropout_rate = sub_args.dropout_rate
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.r_size = sub_args.r_size
            self.num_layer = sub_args.num_layer
            self.comp_sizes = sub_args.comp_sizes
            self.num_comp = sub_args.num_comp
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.mc_samples = sub_args.mc_samples

            # Params for Representation network
            if self.seq_hidden_units is None or self.r_size is None or self.dropout_rate is None or self.fc_hidden_units is None:
                raise MissingMandatoryValue()

            # Params for Conditional distribution networks
            assert len(self.comp_sizes) == self.num_comp
            assert sum(self.comp_sizes) == self.output_size

            # Representation network init + Conditional distribution networks init
            self.repr_net = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)
            self.r_outcome_vitals_head = ROutcomeVitalsHead(self.seq_hidden_units, self.r_size, self.fc_hidden_units,
                                                            self.dim_outcome, self.dim_vitals, self.num_comp, self.comp_sizes)

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.dataset_collection is not None:
            self.dataset_collection.split_train_f_holdout(self.hparams.dataset.holdout_ratio)
            self.dataset_collection.explode_cf_treatment_seq(self.mc_samples)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
        sub_args.r_size = int(input_size * new_args['r_size'])
        sub_args.fc_hidden_units = int(sub_args.seq_hidden_units * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']

    def forward(self, batch, sample=False):
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']

        r = self.build_r(curr_treatments, vitals, prev_outputs, static_features)
        vitals_outcome_pred = self.r_outcome_vitals_head.build_outcome_vitals(r)
        return vitals_outcome_pred

    def build_r(self, curr_treatments, vitals, prev_outputs, static_features):
        # Concatenation of input
        vitals_prev_outputs = []
        vitals_prev_outputs.append(vitals) if self.has_vitals else None
        vitals_prev_outputs.append(prev_outputs) if self.autoregressive else None
        vitals_prev_outputs = torch.cat(vitals_prev_outputs, dim=-1)

        x = torch.cat((curr_treatments, vitals_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.repr_net(x)
        r = self.r_outcome_vitals_head.build_r(x)
        return r

    def training_step(self, batch, batch_ind):
        outcome_next_vitals_pred = self(batch)  # By convention order is (outcomes, vitals)
        outcome_pred = outcome_next_vitals_pred[:, :, :self.dim_outcome]
        next_vitals_pred = outcome_next_vitals_pred[:, :, self.dim_outcome:]

        outcome_mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        # batch['next_vitals'] is shorter by one timestep
        vitals_mse_loss = F.mse_loss(next_vitals_pred[:, :-1, :], batch['next_vitals'], reduce=False) if self.has_vitals else 0.0

        # Masking for shorter sequences
        # Attention! Averaging across all the active entries (= sequence masks) for full batch
        mse_loss_outcome = (batch['active_entries'] * outcome_mse_loss).sum() / batch['active_entries'].sum()
        if self.hparams.model.g_net.fit_vitals:
            mse_loss_vitals = \
                (batch['active_entries'][:, 1:, :] * vitals_mse_loss).sum() / batch['active_entries'][:, 1:, :].sum()
        else:
            mse_loss_vitals = 0.0
        mse_loss = mse_loss_outcome + mse_loss_vitals

        self.log(f'{self.model_type}_train_mse_loss_outcomes', mse_loss_outcome, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_mse_loss_vitals', mse_loss_vitals, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return mse_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        return self(batch).cpu()

    def on_fit_end(self) -> None:
        if self.dataset_collection is not None and hasattr(self.dataset_collection, 'train_f_holdout') \
                and len(self.dataset_collection.train_f_holdout) > 0:
            logger.info('Fitting residuals based on train_f_holdout.')
            self.eval()
            outcome_next_vitals_pred = self.get_predictions(self.dataset_collection.train_f_holdout, vitals=True)

            outcomes_next_vitals = self.dataset_collection.train_f_holdout.data['outputs']
            if self.has_vitals:
                # No ground truth for the last next_vitals
                outcome_next_vitals_pred = outcome_next_vitals_pred[:, :-1, :]
                outcomes_next_vitals = outcomes_next_vitals[:, :-1, :]

                vitals = self.dataset_collection.train_f_holdout.data['next_vitals']
                outcomes_next_vitals = np.concatenate((outcomes_next_vitals, vitals), axis=-1)

            self.holdout_resid = outcomes_next_vitals - outcome_next_vitals_pred
            self.holdout_resid_len = self.dataset_collection.train_f_holdout.data['sequence_lengths']
            if self.has_vitals:
                # No ground truth for the last next_vitals
                self.holdout_resid_len = self.holdout_resid_len - 1
        else:  # Without MC-sampling of residuals
            self.holdout_resid = self.holdout_resid_len = None

    def get_predictions(self, dataset: Union[Dataset, List[Dataset]], vitals=False) -> np.array:
        if not isinstance(dataset, list):
            logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        if isinstance(dataset, list):
            data_loader = \
                [DataLoader(d, batch_size=self.hparams.dataset.val_batch_size, shuffle=False, num_workers=2) for d in dataset]
        else:
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)

        outcome_vitals_pred = self.trainer.predict(self, data_loader)

        if isinstance(dataset, list):
            outcome_vitals_pred = np.stack([torch.cat(pred).numpy() for pred in outcome_vitals_pred], axis=0)
            if vitals:
                return outcome_vitals_pred
            else:
                return outcome_vitals_pred[:, :, :, :self.dim_outcome]

        else:
            outcome_vitals_pred = torch.cat(outcome_vitals_pred).numpy()
            if vitals:
                return outcome_vitals_pred
            else:
                return outcome_vitals_pred[:, :, :self.dim_outcome]

    def get_autoregressive_predictions(self, datasets: List[Dataset]) -> np.array:
        assert hasattr(self, 'holdout_resid') and hasattr(self, 'holdout_resid_len')
        assert len(datasets) == self.mc_samples
        logger.info(f'Autoregressive Prediction for {datasets[0].subset_name} with MC-sampling of trajectories.')

        predicted_outputs = np.zeros((self.mc_samples, len(datasets[0]), self.hparams.dataset.projection_horizon,
                                      self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')

            # MC-sampling of trajectories
            for m in range(self.mc_samples):
                outputs_next_vitals_scaled = self.get_predictions(datasets[m], vitals=True)
                split = datasets[m].data['future_past_split'].astype(int)

                if t > 0:  # Tau >= 2
                    predicted_outputs[m, :, t - 1, :] = \
                        outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :self.dim_outcome]

                # Adding noise from empirical distribution of residuals
                if self.holdout_resid is not None:
                    rand_resid_ind = np.random.randint(len(self.holdout_resid), size=len(datasets[m]))
                    resid_len = self.holdout_resid_len[rand_resid_ind].astype(int)
                    resid_at_split = self.holdout_resid[rand_resid_ind, np.minimum(split - 1 + t, resid_len - 1), :]
                    outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :] += resid_at_split

                # Autoregressive feeding of predicted outcomes and vitals
                if t < self.hparams.dataset.projection_horizon:
                    datasets[m].data['prev_outputs'][range(len(datasets[m])), split + t, :] = \
                        outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :self.dim_outcome]

                    if self.has_vitals:
                        datasets[m].data['vitals'][range(len(datasets[m])), split + t, :] = \
                            outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, self.dim_outcome:]

        predicted_outputs = predicted_outputs.mean(0)  # Averaging over mc_samples
        return predicted_outputs
